#!/usr/bin/env python3
"""The gate must measure arms against the CHAMPION, not against the control.

This file exists because the gate had a real logical defect, found by reading
the code rather than by a failing number.

WHAT THE DEFECT WAS

    delta = candidate - control          # control = the zero-shot logprob readout
    pass if delta >= 0.006

Every trained arm beats the zero-shot control by a wide margin, because training
is the whole point (+0.0665 on the very first null). So a fixed floor on
`delta` is cleared by essentially every arm that trains at all, and an arm that
is *far worse than the incumbent* sails through. With a champion already at
+0.13, an arm landing at +0.05 would pass and be promoted — replacing a much
better model with a much worse one, and moving the bar DOWN.

WHY THE CHAMPION IS THE RIGHT REFERENCE

The control is a constant: it is the same frozen model's logprob readout on the
same evaluation set, measured in the same call, so it is identical for every arm.
Subtracting it is therefore a monotone transform of the absolute score, and
ordering by `delta` is the same as ordering by score — which is why the
trajectory's champion line was right even while the gate was not.

But the THRESHOLD has to move. Upstream's bar is "+0.006 on the suite mean"
against the thing every later arm has to beat, and its own table shows the bar
rising as the champion does: v1.0 0.662, then a keeper at 0.796, then 0.7056,
then 0.7291. Those are absolute scores, and each release had to clear the one
before it. A fixed floor cannot express that.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from loop import ArmResult, gate  # noqa: E402

FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def arm(delta, *, champion_delta=None, arm_name="x", per=None, per_c=None):
    """An arm with `delta` over the control, measured against a champion."""
    r = ArmResult(arm=arm_name, axis="training", change="c",
                  pooled_top1_candidate=0.5 + delta,
                  pooled_top1_control=0.5,
                  per_target_top1=per or {"typed_decisions": 0.5 + delta},
                  per_target_top1_control=per_c or {"typed_decisions": 0.5},
                  prereg={"delta_floor": 0.006, "direction": "up"})
    r.champion_delta = champion_delta
    return r


def main() -> int:
    print("THE DEFECT: a fixed floor on delta-vs-control is cleared by anything")
    print("that trains, so a much worse arm can be promoted over a much better one.")
    print()
    a = arm(0.05, champion_delta=0.13, arm_name="worse_but_trains")
    p, f, why = gate(a)
    check("an arm at +0.05 does NOT clear a champion at +0.13",
          not p, f"pass={p} failed={f}")
    check("... and the reason names the champion",
          "champion" in why.lower(), why[:80])

    print("\nTHE FIX: the bar is champion + noise, so it rises as the champion does")
    # "clearly above" has to clear champion + floor = 0.130 + 0.006 = 0.136.
    a = arm(0.150, champion_delta=0.13, arm_name="better")
    p, f, _ = gate(a)
    check("an arm clearly above the champion passes", p, str(f))

    a = arm(0.1305, champion_delta=0.13, arm_name="hair_over")
    p, f, _ = gate(a)
    check("just above champion + 0.0005 does NOT pass (floor is 0.006)",
          not p and "bar" in f, str(f))

    a = arm(0.136, champion_delta=0.13, arm_name="just_over")
    p, f, _ = gate(a)
    check("champion + 0.006 exactly passes", p, str(f))

    print("\nTHE FIRST ARM still establishes the bar, as it must")
    a = arm(0.0665, champion_delta=None, arm_name="first")
    p, f, _ = gate(a)
    check("with no champion, the first arm is judged on delta vs control",
          p, str(f))
    a = arm(0.001, champion_delta=None, arm_name="first_but_bad")
    p, f, _ = gate(a)
    check("... and a first arm that barely trains is still rejected",
          not p and "bar" in f, str(f))

    print("\nA BAR THAT CANNOT MOVE DOWN")
    a = arm(0.10, champion_delta=0.20, arm_name="regression_attempt")
    p, f, _ = gate(a)
    check("a well-trained arm worse than a strong champion is rejected",
          not p and "bar" in f, str(f))

    print("\nTHE OTHER GUARDS ARE UNCHANGED")
    a = arm(0.20, champion_delta=0.13,
            per={"typed_decisions": 0.70, "other": 0.40},
            per_c={"typed_decisions": 0.62, "other": 0.52})
    p, f, _ = gate(a)
    check("a per-target regression still fails even when the bar is cleared",
          not p and "regression:other" in f, str(f))
    a = arm(0.20, champion_delta=0.13,
            per={"typed_decisions": 0.70, "mmlu_pro_guard": 0.50},
            per_c={"typed_decisions": 0.62, "mmlu_pro_guard": 0.60})
    p, f, _ = gate(a)
    check("losing the general-knowledge guard still fails",
          not p and any(x.startswith("guard:") for x in f), str(f))

    print("\nNO MEASUREMENT IS STILL AN ERROR, NOT A PASS")
    r = ArmResult(arm="x", axis="training", change="c",
                  prereg={"delta_floor": 0.006, "direction": "up"})
    p, f, _ = gate(r)
    check("no primary metric never passes", (not p) and "no_measurement" in f, str(f))

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
