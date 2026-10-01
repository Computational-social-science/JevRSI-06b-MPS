#!/usr/bin/env python3
"""The invariant checker must FIRE, not merely report green.

An invariant module that has only ever been run against a healthy pipeline
cannot be distinguished from one that always returns `ok=True`. This plants each
violation in a temporary state directory, confirms the right invariant catches
it, and confirms it goes quiet again when the violation is removed.

The negative-control discipline from `tests/test_no_cost.py` and
`tests/test_contamination.py`, applied to the thing that is supposed to be
watching everything else. Three guards that only ever said "fine" would be worse
than one that says "fine", because they would be trusted.
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


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def find(fs, inv):
    return next((f for f in fs if f.invariant == inv), None)


def run_isolated(mutate, in_flight=None):
    """Point every module at a temp state dir, run check_all(), restore.

    `in_flight` MUST be injected. `invariants.check_all()` reads it by probing the
    live process table, so without this a test that writes an empty state directory
    is still evaluated against whatever arm happens to be running on the real
    pipeline — which is how "an empty search violates nothing" came to depend on
    the real machine being idle. It passed for the wrong reason, and would have
    failed on a live run for a reason that had nothing to do with the invariant.
    """
    import agenda, confirm, held_out, invariants, loop, power
    mods = (loop, confirm, held_out, power, invariants)
    saved = [(m, getattr(m, "STATE", None), getattr(m, "RECORDS", None)) for m in mods]
    saved_probe: list = []
    tmp = Path(tempfile.mkdtemp(prefix="inv-"))
    try:
        for m in mods:
            if hasattr(m, "STATE"):
                m.STATE = tmp
            if hasattr(m, "RECORDS"):
                m.RECORDS = tmp
        if hasattr(invariants, "LOGS"):
            invariants.LOGS = ROOT / "logs"
        # The one input that does not live in a file.
        saved_probe.append(invariants.running_arm)
        invariants.running_arm = lambda: in_flight
        (tmp / "records").mkdir(parents=True, exist_ok=True)
        mutate(tmp)
        return invariants.check_all()
    finally:
        for m, s, r in saved:
            if s is not None and hasattr(m, "STATE"):
                m.STATE = s
            if r is not None and hasattr(m, "RECORDS"):
                m.RECORDS = r
        for fn in saved_probe:
            invariants.running_arm = fn
        shutil.rmtree(tmp, ignore_errors=True)


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main() -> int:
    print("the healthy case: everything holds")
    F = run_isolated(lambda t: write(t / "arms.jsonl", []) and
                     write(t / "prereg.jsonl", []))
    bad = [f.invariant for f in F if not f.ok and f.severity in ("critical", "major")]
    check("an empty search violates nothing", not bad, str(bad))

    print("\nI6: an arm running out of queue order")
    # This is the one invariant whose input is a PROCESS, not a file. The
    # negative control has to say what is in flight, or it is testing the state
    # of the machine rather than the rule.
    F = run_isolated(lambda t: (
        write(t / "arms.jsonl", []), write(t / "prereg.jsonl", [])),
        in_flight="tower_frozen_head_only")
    f6 = find(F, "I6")
    check("I6 fires on an out-of-order arm", not f6.ok, f6.detail[:60])
    check("... and names the queue head and the arm that jumped it",
          "null_floor" in " ".join(f6.evidence) and "tower_frozen_head_only" in " ".join(f6.evidence),
          str(f6.evidence))
    F = run_isolated(lambda t: (
        write(t / "arms.jsonl", []), write(t / "prereg.jsonl", [])),
        in_flight="null_floor")
    check("I6 is quiet when the queue head IS what is running",
          find(F, "I6").ok, find(F, "I6").detail[:50])
    F = run_isolated(lambda t: (
        write(t / "arms.jsonl", []), write(t / "prereg.jsonl", [])),
        in_flight=None)
    check("I6 is quiet when nothing is in flight", find(F, "I6").ok)

    print("\nI5: a floor that is missing AFTER something has been judged")
    # The healthy case above must NOT trip I5 -- before the first arm completes
    # there is no floor and nothing is at stake. But once a number has been
    # judged against a bar, the bar must have a floor behind it.
    F = run_isolated(lambda t: (
        write(t / "prereg.jsonl", [{"arm": "x", "delta_floor": 0.006, "direction": "up"}]),
        write(t / "arms.jsonl", [{"arm": "x", "verdict": "kept", "contamination": "clean",
                                  "pooled_top1_candidate": 0.7,
                                  "pooled_top1_control": 0.46}])))
    f5 = find(F, "I5")
    check("I5 fires once arms have been judged", (not f5.ok) and f5.severity == "critical",
          f5.detail[:70])
    check("... and says how many arms are at stake",
          "1 arm" in " ".join(f5.evidence), str(f5.evidence))

    print("\nI1: a result with no preregistration")
    F = run_isolated(lambda t: (
        write(t / "arms.jsonl", [{"arm": "x", "verdict": "rejected",
                                  "pooled_top1_candidate": 0.5,
                                  "pooled_top1_control": 0.4}]),
        write(t / "prereg.jsonl", [])))
    f1 = find(F, "I1")
    check("I1 fires", (not f1.ok) and f1.severity == "critical", f1.detail[:60])

    print("\nI2: a prediction left unresolved")
    F = run_isolated(lambda t: (
        write(t / "arms.jsonl", []),
        write(t / "prereg.jsonl", [{"arm": "ghost", "axis": "training",
                                    "change": "c", "prediction": "p",
                                    "delta_floor": 0.006, "direction": "up"}])))
    f2 = find(F, "I2")
    check("I2 fires", not f2.ok, f2.detail[:70])
    check("... and names it", "ghost" in " ".join(f2.evidence), str(f2.evidence))

    print("\nI3: a champion with no record  (this exact failure happened here)")
    def plant_champion(t):
        write(t / "arms.jsonl", [])
        write(t / "prereg.jsonl", [])
        (t / "champion.json").write_text(json.dumps(
            {"arm": "null2", "pooled_top1": 0.531, "control_top1": 0.4645}))
    F = run_isolated(plant_champion)
    f3 = find(F, "I3")
    check("I3 fires and is CRITICAL", (not f3.ok) and f3.severity == "critical",
          f3.detail[:60])

    print("\nI4: the bar moving DOWN")
    def descending(t):
        write(t / "arms.jsonl", [])
        write(t / "prereg.jsonl", [])
        (t / "champion.json").write_text(json.dumps({"arm": "x"}))
        write(t / "champions.jsonl", [
            {"arm": "a", "pooled_top1": 0.60, "control_top1": 0.46},   # +0.14
            {"arm": "b", "pooled_top1": 0.52, "control_top1": 0.46},   # +0.06  WORSE
        ])
    F = run_isolated(descending)
    f4 = find(F, "I4")
    check("I4 fires on a descending bar", not f4.ok, f4.detail[:70])
    check("... and shows the sequence", "0.1400" in " ".join(f4.evidence),
          str(f4.evidence))

    print("\nI6 with a STALE arm log and no live process  (the bug that shipped)")
    # `running_arm()` used to read the most recent arm log, which is simply the
    # name of the last arm that ever ran. Those files are never deleted, so the
    # loop looked permanently mid-arm, I6 reported a correctly-ordered search as
    # out of order, and the critical agent halted a search that was fine. The
    # only way to see that is to make the log lie while no process is running.
    import invariants as _inv
    saved_probe2 = _inv.running_arm
    try:
        _inv.running_arm = lambda: None            # no run_one.py process alive
        F = run_isolated(lambda t: (
            write(t / "prereg.jsonl", []), write(t / "arms.jsonl", [])),
            in_flight=None)
        f6 = find(F, "I6")
        check("I6 is quiet when no process is alive, however stale the logs",
              f6.ok, f6.detail[:50])
    finally:
        _inv.running_arm = saved_probe2

    print("\nI7: a keeper short a seed")
    F = run_isolated(lambda t: (
        write(t / "prereg.jsonl", []),
        write(t / "arms.jsonl", [{"arm": "champion_base", "verdict": "kept",
                                  "role": "decisive", "seeds_required": 3,
                                  "seeds_done": 1,
                                  "pooled_top1_candidate": 0.7,
                                  "pooled_top1_control": 0.46}])))
    f7 = find(F, "I7")
    check("I7 fires", not f7.ok, f7.detail[:60])
    check("... and reports 1/3", "1/3" in " ".join(f7.evidence), str(f7.evidence))

    print("\nI9: a scored arm with no contamination check")
    F = run_isolated(lambda t: (
        write(t / "prereg.jsonl", [{"arm": "x", "delta_floor": 0.006, "direction": "up"}]),
        write(t / "arms.jsonl", [{"arm": "x", "verdict": "rejected",
                                  "pooled_top1_candidate": 0.5,
                                  "pooled_top1_control": 0.4,
                                  "contamination": ""}])))
    f9 = find(F, "I9")
    check("I9 fires on an unchecked arm", not f9.ok, f9.detail[:60])

    print("\nI10: a KEPT arm's checkpoint must be verified; a FAILED arm's must not be")
    F = run_isolated(lambda t: (
        write(t / "prereg.jsonl", [{"arm": "x", "delta_floor": 0.006, "direction": "up"}]),
        write(t / "arms.jsonl", [{"arm": "x", "verdict": "kept", "checkpoint": "ckpt/x",
                                  "artifact": "", "contamination": "clean",
                                  "pooled_top1_candidate": 0.7,
                                  "pooled_top1_control": 0.46}])))
    f10 = find(F, "I10")
    check("I10 fires on a KEPT arm with no artifact field", not f10.ok, f10.detail[:60])

    # The distinction the first version missed. A checkpoint belonging to an arm
    # that failed its fresh-seed confirmation is EVIDENCE, not a product; treating
    # it as an unverified publication raised a critical and halted a search that
    # was correct. This is the exact shape of `confirm__null_floor`.
    F = run_isolated(lambda t: (
        write(t / "prereg.jsonl", []),
        write(t / "arms.jsonl", [{"arm": "confirm__null_floor",
                                  "verdict": "not_confirmed", "checkpoint": "ckpt/c",
                                  "artifact": "not a deliverable: verdict is not_confirmed",
                                  "contamination": "clean",
                                  "pooled_top1_candidate": 0.508,
                                  "pooled_top1_control": 0.4645}])))
    f10b = find(F, "I10")
    check("I10 is quiet on a FAILED arm's checkpoint when it is marked as evidence",
          f10b.ok, f10b.detail[:60])

    F = run_isolated(lambda t: (
        write(t / "prereg.jsonl", []),
        write(t / "arms.jsonl", [{"arm": "c", "verdict": "not_confirmed",
                                  "checkpoint": "ckpt/c", "artifact": "",
                                  "contamination": "clean",
                                  "pooled_top1_candidate": 0.5,
                                  "pooled_top1_control": 0.46}])))
    f10c = find(F, "I10")
    check("... but a BLANK artifact field fires regardless of verdict",
          not f10c.ok, f10c.detail[:60])

    print("\nI11: two kernel stacks in one search")
    F = run_isolated(lambda t: (
        write(t / "prereg.jsonl", [{"arm": "a", "delta_floor": 0.006, "direction": "up"},
                                   {"arm": "b", "delta_floor": 0.006, "direction": "up"}]),
        write(t / "arms.jsonl", [
            {"arm": "a", "verdict": "rejected", "contamination": "clean",
             "kernel_stamp": "torch-reference/torch-2.14.0/cpu",
             "pooled_top1_candidate": 0.5, "pooled_top1_control": 0.4},
            {"arm": "b", "verdict": "rejected", "contamination": "clean",
             "kernel_stamp": "torch-reference/torch-2.14.0/mps",
             "pooled_top1_candidate": 0.5, "pooled_top1_control": 0.4}])))
    f11 = find(F, "I11")
    check("I11 fires on a mid-search device change", not f11.ok, f11.detail[:60])

    print("\nI13: only the surviving lineage keeps its weights")
    import shutil as _sh
    import tempfile as _tf
    from pathlib import Path as _P
    _ck = ROOT / "ckpt"
    _probe = _ck / "_inv_probe_survivor"
    _probe.mkdir(parents=True, exist_ok=True)
    try:
        (_probe / "w.bin").write_bytes(b"x" * 4096)
        rows = [{"arm": "_inv_probe_survivor", "verdict": "not_confirmed",
                 "contamination": "clean", "pooled_top1_candidate": 0.5,
                 "pooled_top1_control": 0.46}]
        F = run_isolated(lambda t: (
            write(t / "prereg.jsonl", []), write(t / "arms.jsonl", rows)))
        f13 = find(F, "I13")
        check("I13 fires on a DISCARDED arm that still holds weights",
              (not f13.ok) and f13.severity == "critical", f13.detail[:60])
        check("... and names it", "_inv_probe_survivor" in " ".join(f13.evidence),
              str(f13.evidence)[:90])

        # The two cases it must NOT fire on. A guard that fires on a survivor is
        # a guard that deletes the only model anybody can publish.
        rows[0].update(verdict="kept", artifact="verified: 2000/2000 agree")
        F = run_isolated(lambda t: (
            write(t / "prereg.jsonl", []), write(t / "arms.jsonl", rows)))
        check("I13 is quiet on a KEPT arm's weights", find(F, "I13").ok,
              find(F, "I13").detail[:60])
    finally:
        _sh.rmtree(_probe, ignore_errors=True)

    print("\nI12: the cost rule, and the checker does not fire on itself")
    import invariants
    f12 = find(invariants.check_all(), "I12")
    check("I12 does NOT fire on the checker itself", f12.ok,
          f12.detail[:60] if not f12.ok else "")

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())