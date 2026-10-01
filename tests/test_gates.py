#!/usr/bin/env python3
"""Gate and proposer tests. No torch, no GPU, no network — runs in <1s.

These are the parts of the loop that decide whether a number means anything, so
they are tested on their own. Each case is a bar condition from upstream's
BENCHMARKS.md, expressed as a test.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from agenda import AGENDA, CHAMPION_SPEC, next_prereg, propose  # noqa: E402
from loop import ArmResult, gate  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def arm(cand_top1, ctrl_top1, *, cand=None, ctrl=None, prereg=None, **kw) -> ArmResult:
    return ArmResult(
        arm=kw.pop("arm", "t"), axis=kw.pop("axis", "training"), change="c",
        pooled_top1_candidate=cand_top1, pooled_top1_control=ctrl_top1,
        per_target_top1=cand if cand else {}, per_target_top1_control=ctrl if ctrl else {},
        prereg=prereg if prereg is not None else next_prereg(AGENDA[0], None).to_json(),
        **kw)


import loop as _L
B = float(_L.bar())


def main() -> int:
    print("preregistration")
    pre = next_prereg(AGENDA[0], None)
    import loop as _lg
    check("a prediction carries the MEASURED bar, not upstream's 0.006",
          pre.delta_floor >= _lg.bar() and pre.delta_floor != 0.006,
          f"prereg={pre.delta_floor} measured={_lg.bar()}")
    check("a prediction names what it is measured against", bool(pre.null_floor_vs))
    check("a prediction states a direction", pre.direction in ("up", "down"))
    pre_c = next_prereg(AGENDA[0], {"arm": "x", "pooled_top1": 0.70})
    check("a floor against a champion names it", "x" in pre_c.null_floor_vs,
          pre_c.null_floor_vs)

    print(f"\nthe bar on the primary (measured {B:.4f})")
    p, f, _ = gate(arm(0.600 + B + 0.01, 0.600, cand={"td": 0.600+B+0.01}, ctrl={"td": 0.600}))
    check(f"clearing {B+0.01:.4f} passes", p and not f, str(f))
    p, f, _ = gate(arm(0.600 + B, 0.600, cand={"td": 0.600+B}, ctrl={"td": 0.600}))
    check(f"clearing {B:.4f} exactly passes", p, str(f))
    p, f, _ = gate(arm(0.6055, 0.600, cand={"td": 0.6055}, ctrl={"td": 0.600}))
    check("missing by 0.0005 fails on the bar alone", (not p) and f == ["bar"], str(f))
    p, f, _ = gate(arm(0.600, 0.600, cand={"td": 0.60}, ctrl={"td": 0.60}))
    check("no movement is not a keeper", (not p) and f == ["bar"], str(f))
    p, f, _ = gate(arm(0.500, 0.600, cand={"td": 0.50}, ctrl={"td": 0.60}))
    check("a loss fails", not p, str(f))

    print("\nguard: no target down by more than its own seed noise (0.011)")
    p, f, _ = gate(arm(0.600 + B + 0.01, 0.600,
                       cand={"td": 0.600 + B + 0.01, "other": 0.50},
                       ctrl={"td": 0.60, "other": 0.52}))
    check("a 0.020 drop on one target is caught", not p and "regression:other" in f, str(f))
    p, f, _ = gate(arm(0.600 + B + 0.01, 0.600,
                       cand={"td": 0.600 + B + 0.01, "other": 0.515},
                       ctrl={"td": 0.60, "other": 0.520}))
    check("a 0.005 drop is inside the noise", p, str(f))

    print("\nguard: general knowledge is not bought with forgetting (0.030)")
    p, f, _ = gate(arm(0.720, 0.600,
                       cand={"td": 0.600 + B + 0.01, "mmlu_pro_guard": 0.50},
                       ctrl={"td": 0.60, "mmlu_pro_guard": 0.62}))
    check("a 0.12 guard loss fails", not p, str(f))
    check("  ... and fails it as its own named guard",
          any(g.startswith("guard:") for g in f), str(f))
    p, f, _ = gate(arm(0.600 + B + 0.01, 0.600,
                       cand={"td": 0.600 + B + 0.01, "mmlu_pro_guard": 0.58},
                       ctrl={"td": 0.60, "mmlu_pro_guard": 0.60}))
    check("a 0.02 guard move is tolerated", p, str(f))

    print("\nverdicts: one failed guard is a repair, two is a death")
    # A guard-target loss alone is exactly ONE failed guard, because the guard
    # carries the wider 0.030 tolerance instead of the 0.011 regression rule.
    _, f1, _ = gate(arm(0.600 + B + 0.01, 0.600,
                        cand={"td": 0.600 + B + 0.01, "mmlu_pro_guard": 0.50},
                        ctrl={"td": 0.60, "mmlu_pro_guard": 0.60}))
    check("clears the bar + forgets general knowledge -> 1 guard -> needs_repair",
          len(f1) == 1 and f1[0].startswith("guard:"), str(f1))
    # A genuine two-guard failure: the guard AND a decision benchmark.
    _, f2, _ = gate(arm(0.720, 0.600,
                        cand={"td": 0.600 + B + 0.01, "other": 0.50, "mmlu_pro_guard": 0.50},
                        ctrl={"td": 0.60, "other": 0.60, "mmlu_pro_guard": 0.62}))
    check("clears the bar + forgets AND regresses -> 2 guards -> rejected",
          len(f2) == 2, str(f2))

    print("\nmeasurement with no number is an error, not a pass")
    p, f, why = gate(arm(None, None))
    check("no primary metric never passes", (not p) and "no_measurement" in f, str(f))

    print("\nthe proposer")
    check("is deterministic", propose([]).name == propose([]).name)
    check("starts at the NULL arm, not a hypothesis",
          propose([]).name == "null_floor", propose([]).name)
    check("... because the noise floor must be measured before any bar is set",
          propose([]).expected_effect == 0.0)
    nulls = [h.name for h in AGENDA if h.name.startswith("null")]
    check("the null series comes first, before any hypothesis",
          [h.name for h in AGENDA[:4]] == nulls, str(nulls))
    check("a floor needs >= 2 points, so there are 4 null arms",
          len(nulls) >= 2, f"{len(nulls)} null arms")
    check("the null repeats use different seeds, so they sample the seed axis",
          len({h.spec.get("seed") for h in AGENDA if h.name.startswith("null")})
          == len(nulls), "same seed would measure only the process axis")
    check("the champion comes after the null series",
          AGENDA[4].name == "champion_base", AGENDA[4].name)
    check("skips arms already run",
          propose([{"arm": h.name} for h in AGENDA[:2]]).name == AGENDA[2].name)
    check("an arm with a prereg but no record is re-run, not skipped",
          propose([{"arm": "champion_base"}]).name == "null_floor",
          "a prediction the log never resolved has to be re-run to close out")
    check("an arm WITH a record is not re-run",
          propose([{"arm": h.name} for h in AGENDA[:3]]).name == AGENDA[3].name)
    check("re-measures the champion once the agenda is exhausted",
          propose([{"arm": h.name} for h in AGENDA]).name == "null_floor")
    axes = Counter(h.axis for h in AGENDA)
    check("covers all three axes", set(axes) == {"model", "data", "training"}, str(dict(axes)))
    check("every arm names one change", all(h.change for h in AGENDA))
    check("every arm states a falsifiable prediction", all(h.prediction for h in AGENDA))
    check("every arm names the source of its expected effect",
          all(h.source for h in AGENDA),
          "an unattributed effect size is not a prediction")
    # The bug this catches: the daemon used to apply the CONFIG seed
    # unconditionally, so all four nulls ran at seed 17 and came back
    # bit-identical -- a "measured" noise sd of exactly 0.0000, which then set
    # the bar at +0.011 and would have been reported as a real resolution limit.
    # Three copies of one run is not a floor.
    nulls = [h for h in AGENDA if h.name.startswith("null")]
    check("each null arm names its OWN seed in its spec",
          all("seed" in h.spec for h in nulls),
          "the config seed must be a default, not an override")
    check("the null seeds are all DIFFERENT",
          len({h.spec["seed"] for h in nulls}) == len(nulls),
          f"seeds: {[h.spec['seed'] for h in nulls]}")
    # null_floor deliberately shares the champion's seed 17: it IS the
    # champion's own first run, the reference point the other three are spread
    # against. What must hold is that the series has four distinct seeds, so
    # three of the four are independent of it.
    champ_seed = AGENDA[4].spec.get("seed")
    check("three of the four nulls are on seeds the champion never ran",
          sum(1 for h in nulls if h.spec["seed"] != champ_seed) >= 3,
          f"champion seed {champ_seed}, null seeds {[h.spec['seed'] for h in nulls]}")
    check("a null arm does not claim its delta is zero",
          not any("delta against its own control is 0" in h.prediction
                  for h in nulls),
          "it trains, so its delta is the effect; the SPREAD is the floor")
    check("every null arm says what the series is for",
          all("SPREAD" in h.prediction or "spread" in h.prediction
              for h in nulls))
    check("the champion spec is upstream v1.0's",
          CHAMPION_SPEC["readout"] == "option_xattn"
          and CHAMPION_SPEC["objective"] == "soft_ce"
          and CHAMPION_SPEC["option_order"] == "shuffled"
          and CHAMPION_SPEC["steps"] == 1500,
          str({k: CHAMPION_SPEC[k] for k in ("readout", "objective", "option_order", "steps")}))

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
