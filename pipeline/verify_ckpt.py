"""Verify a checkpoint reproduces the run that produced it. Or do not publish it.

EDITABLE — ours. Closes the third gap found in the rule-1 audit.

Upstream, in `CONTRIBUTING.md`:

    **Artifacts are verified.** A checkpoint is reloaded from disk, re-scored,
    and published only if it reproduces its training run's per-question
    predictions exactly. Both v1.0 checkpoints agree at **1.0000**, and v2.0 and
    v2.1 at 1.0000 on typed-decisions in both option orders. Their MMLU-Pro
    agreement is 0.999 or better but not always exactly 1, because a few of that
    benchmark's answers turn on a logit margin below 0.001 and flip with
    floating-point summation order -- reported rather than rounded.

Two things worth copying from that paragraph, both of which are about honesty
rather than engineering:

  * **the exact number, not a rounded one** — and
  * **a disagreement is reported, not rounded away.** The MMLU-Pro figure is
    0.999 rather than 1.0 and they say so, with the mechanism. A check that
    rounds a disagreement into a pass is worse than no check, because it
    manufactures the confidence it was built to test.

So this returns the raw agreement, names the failures, and refuses to certify a
checkpoint whose agreement is below `min_agreement`. It does not distinguish
"1.0" from "0.9999" in the pass/fail sense — it reports both, and the caller
decides. The floor is 1.0 by default, so a real disagreement fails loudly, and
the diagnostic is the per-question detail the reader needs to judge it.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

import dev as devmod          # noqa: E402


@dataclass
class Verification:
    ckpt: str = ""
    ok: bool = False
    min_agreement: float = 1.0
    per_target: dict[str, float] = field(default_factory=dict)
    per_target_n: dict[str, int] = field(default_factory=dict)
    disagreements: list[str] = field(default_factory=list)
    n_disagreements: int = 0
    error: str = ""
    reason: str = ""

    def to_json(self) -> dict:
        return self.__dict__.copy()

    def one_line(self) -> str:
        if self.error:
            return f"NOT VERIFIED: {self.error}"
        if self.ok:
            return (f"verified: " + ", ".join(
                f"{t} {a:.4f} on {self.per_target_n.get(t, 0)} questions"
                for t, a in sorted(self.per_target.items())))
        return f"FAILED: {self.reason}"


def _load_items(items_path: Path) -> dict:
    """The training run's own per-question predictions, keyed to compare against.

    `run_arm_lib.item_rows` already writes `target`, `role`, `case_id`, `key` and
    `pred` for every question, so this compares like with like rather than
    re-deriving either side.
    """
    out: dict[tuple, str] = {}
    for line in items_path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("role") != "candidate":
            continue
        out[(r["target"], r.get("option_order", "canonical"),
             r["case_id"], r["key"])] = r["pred"]
    return out


def verify(ckpt: Path, items_path: Path, *, model_id: str,
           min_agreement: float = 1.0) -> Verification:
    """Reload the checkpoint from disk, re-score, compare per question."""
    v = Verification(ckpt=str(ckpt), min_agreement=min_agreement)
    if not ckpt.is_dir():
        v.error = f"no checkpoint at {ckpt}"
        return v
    if not items_path.is_file():
        v.error = f"no training-run items at {items_path}; there is nothing to "
        v.error += "reproduce against"
        return v

    try:
        from rsijev.arch import ArchConfig, DecisionModel
        from rsijev.contract import load_cases
        from rsijev.encode import EncodeConfig
        from rsijev.evaluate import predict
        from rsijev.targets import load_mmlu_pro_1k, load_typed_decisions
        import torch
        from safetensors.torch import load_file
    except Exception as exc:
        v.error = f"import failed: {type(exc).__name__}: {exc}"
        return v

    try:
        # Rebuild the model from the SAVED weights and the SAVED spec, not from
        # the live one. A verification that reconstructs the model any other way
        # is not verifying the artifact.
        meta = json.loads((ckpt / "meta.json").read_text())
        spec = meta.get("spec", {})
        d = devmod.device()
        lm, tok = devmod.load_base_model(meta.get("base_model") or model_id, d)

        from run_arm_lib import text_hidden_size
        arch = ArchConfig(readout=spec.get("readout", "option_xattn"),
                          readout_layer=spec.get("readout_layer", -1),
                          max_options=spec.get("max_options", 80),
                          freeze_base=spec.get("freeze_base", False),
                          option_pool=spec.get("option_pool", "mean"),
                          **dict(spec.get("arch_extra") or {}))
        model = DecisionModel(getattr(lm, "model", lm), text_hidden_size(lm), arch).to(d)
        model.scorer.to(torch.float32)
        miss, unexp = model.tower.load_state_dict(
            load_file(str(ckpt / "tower.safetensors")), strict=False)
        miss = [k for k in miss if "embed_tokens" not in k]
        if miss or unexp:
            v.error = f"tower mismatch: missing {miss[:3]}, unexpected {list(unexp)[:3]}"
            return v
        model.scorer.load_state_dict(load_file(str(ckpt / "scorer.safetensors")),
                                     strict=True)

        expected = _load_items(items_path)
        if not expected:
            v.error = "the training run recorded no candidate rows to compare against"
            return v
        wanted = {t for (t, _, _, _) in expected}
        targets = {}
        if "typed_decisions" in wanted:
            targets["typed_decisions"] = load_typed_decisions("test")
        if "mmlu_pro_guard" in wanted:
            targets["mmlu_pro_guard"] = load_mmlu_pro_1k()[:200]

        for tname, cases in targets.items():
            for order in ("canonical", "reversed"):
                enc = EncodeConfig(layout=spec.get("layout", "state_first"),
                                   option_pool=spec.get("option_pool", "mean"),
                                   option_order=order)
                preds = predict(model, tok, cases, enc,
                                max_options=spec.get("max_options", 80),
                                device=d, batch_size=spec.get("eval_batch_size", 16))
                hit = tot = 0
                for c, q, pr in preds:
                    key = (tname, order, c.case_id, q.key)
                    if key not in expected:
                        continue
                    got = q.options[max(range(len(pr.probs)), key=pr.probs.__getitem__)]
                    tot += 1
                    if got == expected[key]:
                        hit += 1
                    elif len(v.disagreements) < 12:
                        v.disagreements.append(
                            f"{tname}/{order} {c.case_id}/{q.key}: "
                            f"run={expected[key]!r} reload={got!r}")
                if tot:
                    k = f"{tname}/{order}"
                    v.per_target[k] = hit / tot
                    v.per_target_n[k] = tot
    except Exception as exc:
        v.error = f"{type(exc).__name__}: {exc}"
        return v

    v.n_disagreements = sum(v.per_target_n.get(k, 0) - round(v.per_target[k] * v.per_target_n[k])
                            for k in v.per_target)
    worst = min(v.per_target.values()) if v.per_target else 0.0
    v.ok = bool(v.per_target) and worst >= min_agreement
    if v.ok:
        v.reason = (f"worst agreement {worst:.4f} >= floor {min_agreement:.4f}; "
                    f"{len(v.disagreements)} disagreement(s) listed")
    else:
        v.reason = (f"worst agreement {worst:.4f} < floor {min_agreement:.4f}. "
                    f"A checkpoint is published only if it reproduces its training "
                    f"run; this one does not. First disagreements: "
                    f"{v.disagreements[:3]}")
    return v


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--min-agreement", type=float, default=1.0)
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    v = verify(Path(a.ckpt), Path(a.items), model_id=a.model,
               min_agreement=a.min_agreement)
    print(v.one_line())
    for x in v.disagreements[:12]:
        print("   " + x)
    if a.save:
        Path(a.save).write_text(json.dumps(v.to_json(), indent=2) + "\n")
        print(f"-> {a.save}")
    raise SystemExit(0 if v.ok else 1)
