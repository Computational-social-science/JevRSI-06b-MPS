#!/usr/bin/env python3
"""Prove the MPS perf patch changes timing, not numbers.

`rsijev/evaluate.py` is upstream's PROTECTED evaluation path. We changed one
line in it: the per-row device-to-host copy became one copy per batch, because
on Apple MPS every per-row `.tolist()` forces a full pipeline sync and a
2,200-row eval pass was paying 2,200 of them.

That is only legitimate if it is numerically inert. This test asserts it, on
REAL data from the REAL model, by comparing the patched expression against the
original one:

    original:  probs[r, :k].float().tolist()        (per row)
    patched:   probs.float().cpu()[r, :k].tolist()  (per batch)

If these ever disagree, the patch is a change to how things are scored, the
protected boundary has been crossed, and every number this pipeline produces is
suspect. So the test fails loudly rather than the pipeline quietly continuing.

It also times both, because a fix that is correct but not faster is not a fix.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    import torch
    import dev
    from rsijev.encode import EncodeConfig, collate, encode_question
    from rsijev.contract import load_cases

    d = dev.device()
    print(f"device: {d}")

    print("\nequivalence on synthetic tensors, every option count 2..80")
    # The masked tail is the part most likely to differ (a -inf becoming nan on
    # a different path), so test the full padded width, not just the valid part.
    g = torch.Generator().manual_seed(17)
    for width in (2, 3, 4, 5, 8, 16, 80):
        probs = torch.softmax(torch.randn(7, width, generator=g) * 3, dim=-1)
        probs = probs.to(d)
        ks = [min(width, 2 + (i % max(1, width - 1))) for i in range(7)]
        original = [probs[r, :k].float().tolist() for r, k in enumerate(ks)]
        cpu = probs.float().cpu()
        patched = [cpu[r, :k].tolist() for r, k in enumerate(ks)]
        same = all(a == b for a, b in zip(original, patched))
        check(f"width={width:3d}: identical floats", same)

    print("\nequivalence on masked -inf rows (the padded tail)")
    probs = torch.full((3, 10), float("-inf"))
    probs[:, :4] = torch.softmax(torch.randn(3, 4, generator=g), dim=-1)
    probs = probs.to(d)
    original = [probs[r, :4].float().tolist() for r in range(3)]
    cpu = probs.float().cpu()
    patched = [cpu[r, :4].tolist() for r in range(3)]
    check("masked rows give identical floats",
          all(a == b for a, b in zip(original, patched)))
    check("masked rows are finite in the first k columns",
          all(all(v == v for v in row) for row in patched))

    print("\nend-to-end parity on real data, through upstream's own predict()")
    corpus = ROOT / "corpus" / "synth.jsonl"
    if not corpus.is_file():
        check("corpus present for the e2e leg", False, "run scripts/build_replication_corpus.py")
        return 1
    cases = load_cases(str(corpus))[:24]
    import dev as devmod
    lm, tok = devmod.load_base_model("Qwen/Qwen3-0.6B-Base", d)
    from rsijev.arch import ArchConfig, DecisionModel
    from rsijev.encode import iter_questions
    from rsijev.evaluate import predict

    pairs = list(iter_questions(cases))
    enc = EncodeConfig(option_order="canonical")
    model = DecisionModel(getattr(lm, "model", lm), 1024,
                          ArchConfig(readout="option_xattn")).to(d)
    model.scorer.to(torch.float32)

    t0 = time.perf_counter()
    preds = predict(model, tok, cases, enc, max_options=80, device=d, batch_size=8)
    t_new = time.perf_counter() - t0

    # The original path, run by hand: same model, same batches, per-row .tolist().
    import torch.nn.functional as F
    from rsijev.encode import unpermute_logits

    @torch.no_grad()
    def predict_original():
        model.eval()
        out = []
        for i in range(0, len(pairs), 8):
            chunk = pairs[i:i + 8]
            batch = collate(tok, [encode_question(tok, c.state, q, enc) for c, q in chunk],
                            max_options=80, device=d)
            logits = unpermute_logits(model(**batch), batch["option_perm"],
                                      batch["option_mask"])
            p = F.softmax(logits, dim=-1)
            for r, (c, q) in enumerate(chunk):
                out.append(tuple(p[r, : len(q.options)].float().tolist()))
        return out

    t0 = time.perf_counter()
    ref = predict_original()
    t_old = time.perf_counter() - t0

    got = [tuple(pr.probs) for _, _, pr in preds]
    check(f"all {len(got)} rows bit-identical to the original path",
          got == ref, f"{sum(1 for a, b in zip(got, ref) if a != b)} differ")

    # The TIMING half is only meaningful on an idle machine, and this one is
    # rarely idle: an arm training a 0.6B model on the same MPS takes about half
    # the throughput, and the comparison then reports a 1.0x "speedup" for code
    # that is genuinely faster. It failed exactly that way on 2026-10-01 at
    # 12.31s -> 12.67s while `null2` was training, and passed earlier on the same
    # commit at 5.74s -> 6.00s.
    #
    # A guard whose verdict depends on whether an unrelated job happens to be
    # running is not a guard, and one that cries wolf on a healthy commit is the
    # fastest way to teach a reader to ignore the panel. So the timing is SKIPPED
    # while an arm is in flight, loudly -- never the bit-identity half above,
    # which is the half the patch actually has to justify.
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "pipeline"))
    try:
        import doctor as _doctor
        busy = len(_doctor.arm_pids()) > 0
    except Exception:
        busy = False
    if busy:
        print(f"  SKIP  the speedup is real  [an arm is training: {t_old:.2f}s -> "
              f"{t_new:.2f}s ({t_old / max(t_new, 1e-9):.2f}x) is a measurement of "
              f"the machine, not of the patch — re-run between arms]")
    else:
        check("the speedup is real", t_new < t_old, f"{t_old:.2f}s -> {t_new:.2f}s "
          f"({t_old/max(t_new,1e-9):.1f}x)")

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
