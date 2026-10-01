"""One arm, end to end, on our backbone and our device.

EDITABLE — ours, and deliberately thin. Upstream's `scripts/run_arm_lib.py` is
already "one experiment end to end"; its own words: "run_arm_lib.py — one
experiment end to end. release_train.py and the internal search loop both call
it, so a release is the same code path as an experiment." So we do not
reimplement the arm — we call it.

What this file adds, and nothing more:

  * device + dtype. Upstream's entry scripts hardcode `dev = "cuda"` and
    `dtype=torch.bfloat16`; the library takes `device` as a parameter, so this is
    the whole adaptation. CUDA keeps the published stack. MPS gets fp32, because
    bf16-on-MPS has a different reduction order from bf16-on-CUDA and running it
    would create a third numeric stack rather than a second.
  * target and corpus loading, from upstream's own loaders
    (`rsijev.targets`, `rsijev.contract.load_cases`) — the same calls
    `release_train.py` makes, plus a guard-target subsample and a stated fallback
    so a missing dataset is visible in the record rather than silently absent.
  * reading the top-1 number out of upstream's OWN `item_rows` output. Upstream
    already writes one row per question with `pred` and `gold` resolved
    (`scripts/load_release.py` re-derives pooled top-1 from exactly those two
    fields, in exactly that way). We read the same two fields rather than
    inventing a metric, so our number and upstream's number are the same
    measurement by construction.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))     # run_arm_lib, load_release
sys.path.insert(0, str(ROOT / "pipeline"))    # dev, loop

import dev as devmod              # noqa: E402
import contamination as contam    # noqa: E402
from loop import ArmResult, Prereg  # noqa: E402


def load_targets(guard_n: int | None = 200) -> dict:
    """Evaluation targets, via upstream's own loaders.

    1. typed-decisions test split — upstream's primary, with soft gold.
    2. mmlu_pro_1k — upstream's general-knowledge GUARD, subsampled. Its job is
       to catch forgetting, not to be a headline, and 200 rows is enough to see
       a 0.03 move against a ~0.011 seed sd. Upstream scores the full 1000; we
       name the subsample in the record so the two are not confused.

    A dataset that will not load is reported and the loop continues with what
    loaded. A missing guard is a weaker claim, not a false one — provided the
    record says the guard was absent, which it does.
    """
    from rsijev.targets import load_mmlu_pro_1k, load_typed_decisions

    targets: dict[str, list] = {}
    try:
        targets["typed_decisions"] = load_typed_decisions("test")
    except Exception as exc:
        print(f"  [targets] typed-decisions UNAVAILABLE: {type(exc).__name__}: {exc}")
    try:
        mm = load_mmlu_pro_1k()
        targets["mmlu_pro_guard"] = mm[:guard_n] if guard_n else mm
    except Exception as exc:
        print(f"  [targets] mmlu_pro UNAVAILABLE: {type(exc).__name__}: {exc}")
    if not targets:
        raise RuntimeError("no evaluation target loaded; refusing to run blind")
    for k, v in targets.items():
        print(f"  [targets] {k}: {len(v)} cases, "
              f"{sum(len(c.questions) for c in v)} questions")
    return targets


def load_corpus(corpus_dir: Path) -> dict[str, list]:
    """Corpus, via upstream's own `load_cases` — the reader it wrote them with."""
    from rsijev.contract import load_cases
    out: dict[str, list] = {}
    for f in sorted(Path(corpus_dir).glob("*.jsonl")):
        out[f.stem] = load_cases(str(f))
        print(f"  [corpus] {f.stem}: {len(out[f.stem])} cases, "
              f"{sum(len(c.questions) for c in out[f.stem])} questions")
    if not out:
        raise RuntimeError(f"no corpus files in {corpus_dir}")
    return out


def top1_from_items(items_path: Path) -> dict[str, dict[str, float]]:
    """Pooled top-1 per (target, role), read off upstream's `item_rows` output.

    Identical to `scripts/load_release.py --verify`:
        pred = argmax over p.probs; ok = pred == gold_label(q, c.gold[q.key])
    which is the definition of top-1 everywhere in this project. We do not
    define a metric; we read the one the arm runner already wrote per question,
    so there is no second implementation to disagree with the first.
    """
    agg: dict[tuple[str, str], list[int]] = defaultdict(list)
    with open(items_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["target"], r["role"])
            # canonical order only: reversed is a robustness diagnostic, and
            # pooling it in would score a model twice under two presentations.
            if r.get("option_order", "canonical") != "canonical":
                continue
            agg[key].append(int(r["pred"] == r["gold"]))
    return {f"{t}/{role}": {"top1": sum(v) / len(v), "n": len(v)}
            for (t, role), v in agg.items() if v}


def run_arm(*, arm: str, spec: dict, prereg: Prereg, corpus_dir: Path,
            model_id: str, save_dir: Path | None = None,
            guard_n: int | None = 200) -> ArmResult:
    """Train and score one arm. Never raises: an error is a record.

    An arm that crashes is still data. Upstream registers an out-of-the-ordinary
    outcome rather than losing it, and a crash mid-run says the recipe is
    unstable, which is one of the things the loop exists to find.
    """
    import run_arm_lib as lib

    res = ArmResult(arm=arm, axis=prereg.axis, change=prereg.change,
                    champion=prereg.null_floor_vs, prereg=prereg.to_json())
    t0 = time.perf_counter()
    out_dir = ROOT / "records" / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        dev = devmod.device()
        print(f"\n=== ARM {arm} ({prereg.axis}) ===")
        print(f"  change    : {prereg.change}")
        print(f"  prereg    : {prereg.prediction}")
        print(f"  floor     : +{prereg.delta_floor} vs {prereg.null_floor_vs}")
        print(f"  device    : {dev}  kernel={devmod.kernel_stamp()}")

        lm, tok = devmod.load_base_model(model_id, dev)
        print(f"  backbone  : {model_id}  "
              f"{sum(p.numel() for p in lm.parameters())/1e6:.1f}M params")
        targets = load_targets(guard_n)
        corpus = load_corpus(corpus_dir)

        # Contamination: VERIFIED here, not asserted at corpus-build time.
        # Upstream: "no release trains on any benchmark's test split, and no
        # training document shares a state with one ... verified by the arm
        # runner rather than asserted." It runs on the objects this arm is about
        # to train on -- not on a re-read of the files, and not on a snapshot
        # taken when the corpus was built.
        crep = contam.check(corpus, targets)
        contam.save(crep, arm)
        print(f"  contam    : {crep.verdict or crep.one_line()}")
        if not crep.clean:
            # A contaminated arm's numbers are not weaker, they are about a
            # different thing, so this is an error verdict and not a warning.
            res.error = crep.verdict
            res.verdict = "error"
            res.reason = "refused to run: the corpus is not clean against the targets"
            print(f"\n!!! ARM {arm} REFUSED: {crep.verdict}")
            return res
        print(f"             corpus files {crep.corpus_files} vs targets {crep.eval_targets}")

        full_spec = {**spec, "seed": spec.get("seed", 17)}
        if save_dir:
            full_spec["save_dir"] = str(save_dir)

        items_path = out_dir / f"{arm}.items.jsonl"
        recs = lib.run_arm(lm=lm, tok=tok, targets=targets, corpus=corpus, device=dev,
                           spec=full_spec, name=arm, items_path=items_path)
        (out_dir / f"{arm}.rows.json").write_text(json.dumps(recs, indent=1, default=str))

        # top-1 from upstream's own per-question rows
        tops = top1_from_items(items_path)
        primary = "typed_decisions" if "typed_decisions" in targets else next(iter(targets))
        ck, kk = tops.get(f"{primary}/control"), tops.get(f"{primary}/candidate")
        if not ck or not kk:
            res.error = f"no primary rows for {primary}"
            res.verdict = "error"
            res.reason = res.error
            return res
        res.pooled_top1_control = ck["top1"]
        res.pooled_top1_candidate = kk["top1"]
        res.n_questions = kk["n"]
        res.targets_used = sorted({k.split("/")[0] for k in tops})

        for k, v in tops.items():
            t, role = k.split("/")
            (res.per_target_top1 if role == "candidate" else res.per_target_top1_control)[t] = v["top1"]

        # the rest of the numbers, straight out of upstream's as_record rows
        for r in recs:
            if r.get("role") == "candidate" and r.get("target") == primary \
                    and r.get("option_order", "canonical") == "canonical":
                res.min_decision_score_candidate = r.get("min_decision_score")
                res.pooled_aurc_candidate = r.get("pooled_aurc")
                res.ece_candidate = r.get("pooled_ece", r.get("choice.ece"))
                res.final_loss = r.get("final_loss")
                res.train_seconds = r.get("train_seconds", 0.0) or 0.0
            if r.get("role") == "control" and r.get("target") == primary \
                    and r.get("option_order", "canonical") == "canonical":
                res.min_decision_score_control = r.get("min_decision_score")
                res.pooled_aurc_control = r.get("pooled_aurc")
                res.ece_control = r.get("pooled_ece", r.get("choice.ece"))
        res.contamination = (f"clean: {crep.n_train_cases} train vs "
                             f"{crep.n_eval_cases} eval, 0 exact / 0 near")
        res.checkpoint = str(save_dir) if save_dir else ""
        res.train_seconds = res.train_seconds or (time.perf_counter() - t0)
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
        res.verdict = "error"
        res.reason = "arm raised; see the arm log"
        res.train_seconds = time.perf_counter() - t0
        print(f"\n!!! ARM {arm} FAILED: {res.error}")
        traceback.print_exc()
    return res
