#!/usr/bin/env python3
"""A PARTIAL suite mean, by upstream's own rules, over the benchmarks we can load.

## Why this exists

Upstream's headline from v2.0 on is a weighted mean over many benchmarks, and
BENCHMARKS.md says why in one sentence: "v1.0 was judged on one benchmark; that
turned out to be too narrow to tell a real gain from a benchmark-specific one, so
from v2.0 the headline is a weighted mean over twelve. From v3.0 it is over
fifteen."

This project gates on ONE benchmark, which is upstream's v1.0 protocol — the one
they abandoned, for a stated reason that applies to us exactly: a gain on one
benchmark cannot be distinguished from a gain on that benchmark. And it is not a
theoretical worry here, because the recipe's one measurable cost is on a DIFFERENT
benchmark: general knowledge falls -0.0413 +/- 0.0202 on MMLU-Pro while the
primary target rises. A single-benchmark gate cannot see a trade-off between two
benchmarks, which is why every arm currently returns `needs_repair` on a guard the
design cannot resolve, and why the one arm whose null could be informative cannot
produce one.

The full suite needs thirteen more upstream checkouts (`KEV_ROOT`, `NIMBLE_ROOT`,
`JEVBENCH_ROOT`, ...), none of which has a default because "a default would be one
author's filesystem". Upstream also supports a partial run, and states its rule:
"With `--only`, the mean is renormalised over the weight scored, and the script
names the benchmarks it left out."

So this is that rule, as a callable, over the two benchmarks that need no checkout:

  * the measurement is UPSTREAM'S. `rsijev/targets_suite.load_suite` builds the
    cases, `SUITES["v3"]` supplies the weights, and the mean is theirs. Nothing
    here re-implements a loader, re-derives a weight, or re-scores anything.
  * the incompleteness is REPORTED, not absorbed. `named_benchmarks()` lists every
    benchmark this partial run does not cover, with its weight, so a number from
    here can never be read as a suite number.
  * `decontam=False` is deliberate and stated: `SUITE_DECONTAM` names eval cases
    that overlap the training data UPSTREAM held. Our corpus is not their corpus,
    so their report does not describe our overlap and applying it would be wrong in
    the dangerous direction. Our own authority is `pipeline/contamination.py`, which
    re-checks the LOADED corpus against the LOADED targets before any compute, per
    arm. Upstream's own words for the alternative: compare the two numbers before
    trusting either.

No network is required beyond what the two loaders already fetch, and nothing here
writes to `ckpt/` or `records/`.

    python pipeline/partial_suite.py --list
    python pipeline/partial_suite.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The two benchmarks whose splits come from the HuggingFace Hub rather than a local
# checkout, so they load with no root configured. Their v3 weights are 0.20 and
# 0.081 -- 28.1% of the suite, and between them they are the whole trade-off this
# project cannot currently see: the primary target and the general-knowledge guard.
LOCAL_LOADABLE = ("typed_decisions_test", "mmlu_pro_1k")


def named_benchmarks() -> list[tuple[str, float, str]]:
    """(name, weight, why it is not in this partial run), for every benchmark left out."""
    from rsijev.targets_suite import SUITES
    out = []
    for name, w in sorted(SUITES["v3"].items(), key=lambda kv: -kv[1]):
        if name in LOCAL_LOADABLE:
            continue
        if name == "tasksource_jev_test":
            why = "held out: upstream scores it once per release, never during the search"
        elif name in ("semif_external", "scienthoon_ood"):
            why = "held out for the same reason, and we have not spent it either"
        else:
            why = "needs a local upstream checkout (KEV_ROOT / NIMBLE_ROOT / JEVBENCH_ROOT)"
        out.append((name, w, why))
    return out


def partial_mean() -> dict:
    from rsijev import targets_suite as ts
    got = ts.load_suite(list(LOCAL_LOADABLE), decontam=False, suite="v3")
    w = ts.SUITES["v3"]
    scored = {n: w[n] for n in got}
    total = sum(scored.values())
    return {
        "suite": "v3",
        "scored": {n: {"weight_upstream": scored[n],
                       "weight_renormalised": scored[n] / total if total else 0.0,
                       "cases": len(got[n]),
                       "questions": sum(len(getattr(c, "questions", []) or []) for c in got[n])}
                   for n in got},
        "weight_covered": total,
        "coverage": f"{len(got)} of {len(w)} benchmarks, {total * 100:.1f}% of the suite weight",
        "omitted": named_benchmarks(),
        "decontam": "upstream's SUITE_DECONTAM report describes THEIR training data; "
                    "ours differs, so it is not applied. Our authority is "
                    "pipeline/contamination.py, per arm, on the loaded corpus.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="print the benchmarks this partial run does NOT cover, and stop")
    a = ap.parse_args()
    if a.list:
        print(f"NOT covered by pipeline/partial_suite.py "
              f"(renormalised over what is):\n")
        for n, wt, why in named_benchmarks():
            print(f"  {n:22s} w={wt:.3f}   {why}")
        print(f"\ncovered: {', '.join(LOCAL_LOADABLE)}")
        return 0
    r = partial_mean()
    print(f"PARTIAL SUITE  ({r['suite']} weighting, renormalised over what was scored)")
    print(f"  coverage: {r['coverage']}\n")
    for n, d in r["scored"].items():
        print(f"  {n:22s} w {d['weight_upstream']:.3f} -> {d['weight_renormalised']:.4f}   "
              f"{d['cases']} cases, {d['questions']} questions")
    print(f"\n  decontamination: {r['decontam']}")
    print(f"\n  NOT covered — a number from here is not a suite number:")
    for n, wt, why in r["omitted"]:
        print(f"    {n:22s} w={wt:.3f}   {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
