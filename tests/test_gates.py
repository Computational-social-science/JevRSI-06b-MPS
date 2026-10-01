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
    # A 0.02 guard move is INSIDE upstream's 0.030 tolerance and outside nothing
    # else -- under the plain rule it passes. It no longer passes, and that is the
    # correction of 2026-10-01 rather than a regression: the guard's own measured
    # spread over four replicas of this recipe is 0.0202, so 0.02 sits inside
    # 0.030 +/- 0.0202. A tolerance checked against a difference smaller than the
    # measurement's own noise is a coin flip wearing a threshold's clothes, and it
    # had already crowned one keeper on a lucky seed. So the band turns "tolerated"
    # into "unresolved", which routes to the repair path -- upstream's own
    # disposition for an arm that fails exactly one guard. Both readings are
    # asserted here, because the interesting claim is not the verdict but the fact
    # that the verdict is now a function of the measured band.
    p, f, _ = gate(arm(0.600 + B + 0.01, 0.600,
                       cand={"td": 0.600 + B + 0.01, "mmlu_pro_guard": 0.58},
                       ctrl={"td": 0.60, "mmlu_pro_guard": 0.60}))
    check("a 0.02 guard move is no longer silently tolerated", not p, str(f))
    check("  ... it is UNRESOLVED, not a failure: inside the measured band",
          any(g.endswith(":unresolved") for g in f), str(f))
    p, f, _ = gate(arm(0.600 + B + 0.01, 0.600,
                       cand={"td": 0.600 + B + 0.01, "mmlu_pro_guard": 0.595},
                       ctrl={"td": 0.60, "mmlu_pro_guard": 0.60}))
    check("a 0.005 guard move, inside the noise, still passes outright", p, str(f))

    print("\nverdicts: one failed guard is a repair, two is a death")
    # A guard-target loss alone is exactly ONE failed guard, because the guard
    # carries the wider 0.030 tolerance instead of the 0.011 regression rule.
    _, f1, _ = gate(arm(0.600 + B + 0.01, 0.600,
                        cand={"td": 0.600 + B + 0.01, "mmlu_pro_guard": 0.50},
                        ctrl={"td": 0.60, "mmlu_pro_guard": 0.60}))
    check("clears the bar + forgets general knowledge -> 1 guard -> needs_repair",
          len(f1) == 1 and f1[0].startswith("guard:"), str(f1))
    # A genuine two-guard failure: the guard AND a decision benchmark.
    _, f2, _ = gate(arm(0.600 + B + 0.01, 0.600,
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

    # ---------------------------------------------------------------- the bar
    #
    # Everything above tests that the null series is built correctly. These test
    # the thing the series is FOR, which is where the pipeline was wrong for a
    # day: the bar stood at +0.0450 while doing nothing was worth +0.0680, so the
    # first null arm cleared it and the search recorded plain training as an
    # improvement. A bar that an arm clearing by training alone can pass is not a
    # threshold.
    print("\nthe bar excludes what doing nothing buys")
    import confirm as _cf
    import power
    measured = [0.0665, 0.0435, 0.0940]          # the three measured replicas
    decisive_seeds = _cf.seeds_for("decisive")
    pw = power.analyse(measured, planned_seeds=decisive_seeds)
    check("the bar is derived at the seed budget a DECISIVE arm actually runs",
          pw.planned_seeds == decisive_seeds,
          f"bar at {pw.planned_seeds}, decisive arms run {decisive_seeds}")
    check("the bar sits ABOVE the do-nothing mean",
          pw.recommended_bar > pw.do_nothing_mean,
          f"bar +{pw.recommended_bar:.4f} vs doing nothing +{pw.do_nothing_mean:.4f}")
    check("no arm that changed NOTHING can clear the bar",
          pw.recommended_bar > max(measured),
          f"largest null +{max(measured):.4f} — a bar at or below this is a "
          f"rubber stamp")
    check("the margin sits on the VARIANCE, not on the effect",
          abs((pw.recommended_bar - pw.do_nothing_mean)
              - power.increment_resolvable(pw.paired_sd, decisive_seeds)) < 1e-6,
          "bar = do-nothing mean + resolvable increment, and nothing else")
    check("the bar names the rule that produced it",
          pw.bar_rule == "do_nothing_mean + resolvable increment", pw.bar_rule)
    check("the rationale says what doing nothing is worth",
          f"{pw.do_nothing_mean:+.4f}" in pw.bar_rationale,
          "a bar whose reason cannot be stated is a number someone typed")

    # A hypothesis arm's own effect must never be able to inflate the floor it is
    # judged against. The daemon used to feed EVERY recorded delta into `analyse`,
    # which made the bar a ratchet: each result raised the threshold for the next.
    from agenda import is_null_series, null_deltas
    mixed = [{"arm": "null1", "pooled_top1_candidate": 0.5585, "pooled_top1_control": 0.4645},
             {"arm": "confirm__null_floor", "pooled_top1_candidate": 0.508,
              "pooled_top1_control": 0.4645},
             {"arm": "champion_base", "pooled_top1_candidate": 0.90,
              "pooled_top1_control": 0.4645}]
    check("a hypothesis arm's delta cannot raise the floor",
          [round(d, 4) for d in null_deltas(mixed)] == [0.094, 0.0435],
          f"null series from a mixed log: {[round(d, 4) for d in null_deltas(mixed)]} "
          f"— champion_base's +0.4355 is excluded")
    check("a record with no control is skipped, not defaulted",
          null_deltas([{"arm": "null9", "pooled_top1_candidate": 0.5}]) == [],
          "a delta invented from a missing control is not a measurement")

    # The membership test that keeps the floor honest has to be ONE definition.
    check("the null series is identified the same way everywhere",
          is_null_series("null2") and is_null_series("confirm__null_floor")
          and not is_null_series("champion_base")
          and not is_null_series("model_layer_mix"),
          "a producer and a checker that each guess 'is this a null?' will "
          "disagree silently")

    # ------------------------------------------------------- the seed promise
    print("\nthe seed is part of the prediction, and it is on disk")
    for h in nulls:
        pre_h = next_prereg(h, None)
        check(f"{h.name} pins seed {h.spec.get('seed')} in its preregistration",
              pre_h.seed == h.spec.get("seed"),
              f"prereg says {pre_h.seed}, spec says {h.spec.get('seed')}")

    # ------------------------------------------- the duplicate-run detector
    #
    # The general form of the bug that cost D1 its null series, and that came back
    # as `records/null2`: two arm names, one measurement. `null2` ran at seed 17
    # while 37 was written down and returned rows bit-identical to `null_floor`.
    # A measured sd of exactly zero is the signature; this reads the artifacts
    # themselves so the signature cannot sit unnoticed in `records/`.
    import hashlib
    import json as _json
    root = Path(__file__).resolve().parent.parent
    sigs: dict[str, list[str]] = {}
    for d in sorted((root / "records").glob("*/")):
        for f in sorted(d.glob("*.rows.json")):
            body = [r for r in _json.loads(f.read_text())
                    if not isinstance(r, dict) or r.get("arm") in (None, d.name)]
            sig = hashlib.sha256(
                _json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
            sigs.setdefault(sig, []).append(d.name)
    dupes = {k: v for k, v in sigs.items() if len(v) > 1}
    check("no two recorded arms are the same run",
          not dupes, str(dupes) if dupes else
          f"{len(sigs)} distinct measurement(s) across {len(list((root / 'records').glob('*/')))} arm dirs")

    # ------------------------------------------------- the state schema gate
    #
    # `power.load()` is `PowerResult(**raw)`, so a field added to the dataclass
    # and not to the file on disk raises TypeError -- and on 2026-10-01 that
    # killed the doctor for roughly three hours (14:37, 14:52, 15:07, 17:56, all
    # "could not run: TypeError"), silently, with the watchdog logging the
    # failure and carrying on. A round-trip through save/load is the cheapest
    # possible proof that the state file and the dataclass still agree, and it
    # belongs in the build rather than in a log nobody reads.
    print("\nthe power state round-trips (a schema drift must fail here, not at 3am)")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        orig = power.STATE
        try:
            power.STATE = Path(td)
            power.save(pw)
            back = power.load()
        finally:
            power.STATE = orig
    check("save() then load() returns the same result",
          back == pw,
          "" if back == pw else
          f"mde[1] survives as {pw.mde.get(1)} -> "
          f"{back.mde.get(1) if back else 'unloadable'}")

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
