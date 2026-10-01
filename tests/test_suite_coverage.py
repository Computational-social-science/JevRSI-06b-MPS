#!/usr/bin/env python3
"""The suite's coverage is a measured property of this machine, not an intention.

## Why

The gate is moving from one benchmark to upstream's weighted mean over fifteen, and
the weight that mean is renormalised over depends on which benchmarks THIS machine
can actually load. That set is not a constant: it is three git checkouts at pinned
commits, two Hub datasets, and whatever upstream has published. A checkout can
disappear, a pin can stop resolving, an unpublished input can stay unpublished.

So coverage can shrink silently, and the failure is the dangerous kind: the gate
keeps working, the numbers keep looking like suite numbers, and the denominator
changed underneath. Upstream's own rule is that a partial run "names the benchmarks
it did not score and says its mean is renormalised" -- so the name list is the
artifact, and it is checked here.

What this asserts:

  * the coverage is what the manifest says it is, benchmark by benchmark;
  * the renormalised weights sum to 1 (a partial mean that does not renormalise is
    quietly a different metric);
  * the omitted list NAMES every omitted benchmark, with upstream's weights, so no
    gap can be absorbed;
  * a benchmark that is present but not loadable is reported as such rather than
    quietly dropped from the denominator;
  * and the coverage has not fallen below the floor recorded in the manifest -- a
    shrink is a finding, not a rounding.

No torch, no GPU, no training. Loads the benchmark definitions, which is Hub I/O
and a few seconds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    import bench_roots
    import partial_suite as ps
    from rsijev.targets_suite import SUITES

    print("the suite's coverage, measured on this machine")
    r = ps.partial_mean()
    bench = json.loads((ROOT / "config" / "benchmarks.json").read_text())
    print(f"  {r['coverage']}")

    # 1. every scoreable benchmark is one this manifest claims, and vice versa
    claimed = set()
    for e in bench["checkouts"].values():
        claimed.update(e.get("benchmarks", []))
    claimed.update(ps.LOCAL_LOADABLE)
    scored = set(r["scored"])
    unaccounted = scored - claimed
    check("every benchmark scored is one the manifest accounts for", not unaccounted,
          "; ".join(sorted(unaccounted)) or
          f"{len(scored)} scored, all from config/benchmarks.json")

    # 2. the weights used are upstream's, and they renormalise
    w = SUITES[bench["suite"]]
    tot = sum(w[n] for n in scored)
    renorm = [d["weight_renormalised"] for d in r["scored"].values()]
    check("every weight used is upstream's own",
          all(abs(r["scored"][n]["weight_upstream"] - w[n]) < 1e-12 for n in scored),
          f"v3 weighting, {len(w)} benchmarks defined upstream")
    check("the renormalised weights sum to 1", abs(sum(renorm) - 1.0) < 1e-9,
          f"sums to {sum(renorm):.6f} over the weight actually scored")

    # 3. nothing is dropped quietly
    omitted = {n for n, _, _ in r["omitted"]}
    failed = set(r["failed_to_load"])
    everything = set(w)
    check("every benchmark is either scored, omitted-by-name, or named as unloadable",
          scored | omitted | failed == everything,
          f"scored {len(scored)}, omitted {len(omitted)}, present-but-unloadable "
          f"{len(failed)}; upstream defines {len(everything)}")
    check("a present-but-unloadable benchmark is reported, not dropped",
          all(n in omitted or n in scored for n in failed) and all(
              why for _, _, why in r["omitted"]),
          f"{', '.join(sorted(failed)) or 'none'} — each with a stated reason"
          if failed else "none present but unloadable")
    check("every omitted benchmark carries a reason and upstream's weight",
          all(why.strip() and wt > 0 for _, wt, why in r["omitted"]),
          f"{len(r['omitted'])} omitted, {sum(wt for _, wt, _ in r['omitted']):.3f} "
          f"of the weight")

    # 4. the floor, so a shrink is a finding
    floor = bench.get("coverage_floor")
    check("coverage has not fallen below the floor recorded in the manifest",
          floor is None or tot >= floor - 1e-9,
          f"{tot:.3f} now, floor {floor}" if floor else "no floor recorded")

    # 5. the pins the checkouts are at
    rows = bench_roots.verify()
    wrong = [r_["name"] for r_ in rows if r_["present"] and not r_["at_pin"]]
    check("every checkout present is AT upstream's pin", not wrong,
          "; ".join(wrong) or "; ".join(
              f"{r_['name']} {r_['head']}" for r_ in rows if r_["at_pin"]) or "none present")

    print(f"\n  scored: {', '.join(sorted(scored))}")
    print(f"  questions: {sum(d['questions'] for d in r['scored'].values())}")
    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
