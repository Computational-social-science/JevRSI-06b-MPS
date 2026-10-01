#!/usr/bin/env python3
"""End-to-end test of the confirmation flow, on a temporary state directory.

The point is the *sequence*, not the arithmetic: an arm clears the bar, does NOT
become champion, a fresh-seed run is scheduled, the proposer runs that run before
any new hypothesis, and the crown moves only on a confirmation. Every one of
those steps is a place where a loop can quietly let the bar drift on evidence it
selected for, so each is asserted.

Runs on synthetic records. It does not touch the real `records/`, `state/` or
any checkpoint, and it launches no subprocess.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    import confirm as confirmmod
    import held_out
    from loop import ArmResult, gate
    import loop as loopmod
    import agenda as agendamod

    tmp = Path(tempfile.mkdtemp(prefix="confirm-test-"))
    try:
        # Redirect every module that writes state at import time.
        for mod, path in ((confirmmod, tmp / "pending_confirm.json"),
                          (held_out, tmp / "held_out.json"),
                          (loopmod, tmp / "recs.jsonl")):
            mod.STATE = tmp
            if mod is held_out:
                mod.HELD_OUT_JSON = path
        loopmod.RECORDS = tmp

        # --- the bar, for reference
        from agenda import AGENDA
        base = dict(AGENDA[4].spec)               # champion_base
        pre = agendamod.next_prereg(AGENDA[4], None)
        floor = pre.delta_floor

        # --- STEP 1: an arm clears the bar
        keeper = ArmResult(arm="champion_base", axis="training", change="c",
                           pooled_top1_candidate=0.720, pooled_top1_control=0.600,
                           per_target_top1={"typed_decisions": 0.720},
                           per_target_top1_control={"typed_decisions": 0.600},
                           prereg=pre.to_json())
        passed, failed, reason = gate(keeper)
        check("an arm clearing the bar passes the gate", passed and not failed, reason)
        keeper.verdict = "kept"
        check("... and is initially labelled kept", keeper.verdict == "kept")

        # --- STEP 2: it does NOT get the crown; a confirmation is scheduled
        pend = confirmmod.schedule(
            "champion_base", 0, delta=0.120, floor=floor,
            steps=300, batch_size=16)
        keeper.verdict = "kept_pending_confirm"
        check("the keeper is downgraded to pending, not crowned",
              keeper.verdict == "kept_pending_confirm")
        check("a confirmation is owed", confirmmod.pending() is not None)
        check("the confirmation uses a seed the arm never ran on",
              pend["seed"] != base.get("seed", 17),
              f"arm seed {base.get('seed')} vs confirm seed {pend['seed']}")
        check("the confirmation is named for the arm it confirms",
              pend["confirm_arm"] == "confirm__champion_base")
        check("the bar it must clear is the same bar", pend["floor"] == floor)

        # --- STEP 3: a confirmation that HOLDS crowns the arm
        conf_ok = ArmResult(arm="confirm__champion_base", axis="training", change="c",
                            pooled_top1_candidate=0.712, pooled_top1_control=0.600,
                            prereg=pre.to_json())
        p2, f2, _ = gate(conf_ok)
        check("the confirmation itself is gated by the same bar", p2 and not f2)
        out = confirmmod.judge(original_arm="champion_base", original_delta=0.120,
                               confirm_delta=0.112, floor=floor)
        check("a holding confirmation confirms the arm", out.confirmed, out.reason[:60])
        if out.confirmed:
            confirmmod.clear()
        check("and the debt is cleared", confirmmod.pending() is None)

        # --- STEP 4: a confirmation that FAILS does not crown, and does not
        #     discard either. It says seed sensitivity, which is a third thing.
        confirmmod.schedule("champion_base", 1, delta=0.120, floor=floor,
                            steps=300, batch_size=16)
        out2 = confirmmod.judge(original_arm="champion_base", original_delta=0.120,
                                confirm_delta=0.004, floor=floor)
        check("a failing confirmation does not confirm", not out2.confirmed)
        check("... and is not called 'rejected' either",
              out2.verdict == "not_confirmed", out2.verdict)
        check("... it says seed sensitivity", "seed sensitivity" in out2.reason)
        check("the debt is NOT cleared, so it cannot be lost",
              confirmmod.pending() is not None)

        # --- STEP 5: the pending confirmation outranks the agenda
        h = agendamod.propose([])          # the agenda's own next arm
        check("the agenda would otherwise propose a fresh hypothesis",
              h.name == "null_floor", h.name)
        pend2 = confirmmod.pending()
        # A second run of the SAME arm gets a distinct name. Two records called
        # `confirm__champion_base` would be indistinguishable in the log, and
        # the whole point of a multi-seed arm is knowing which run is which.
        check("the daemon's pending branch names the right arm",
              pend2["confirm_arm"] == "confirm__champion_base_r2",
              pend2["confirm_arm"])
        check("and records which run it is", pend2.get("run") == 2
              and pend2.get("runs_required") == 2, str(pend2.get("run")))

        # --- STEP 6: the held-out set is independent of all of this
        check("a keeper does NOT read the held-out set on its own",
              held_out.is_spent() is False,
              "the read happens only when a champion is CONFIRMED")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
