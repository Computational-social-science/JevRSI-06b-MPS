#!/usr/bin/env python3
"""Does the halt actually stop the search, and does it refuse to let itself off?

A stop mechanism that has never been exercised is a comment. And a stop mechanism
that can clear itself is worse than none, because it converts a real halt into a
delay: a flapping invariant would resume the search at 3:01 and the audit trail
would show a clean run.

So the tests here cover the three properties the design rests on, and each is
tested by planting the condition rather than by reading the code:

  1. a science-class verdict HALTS, and an operational one does not;
  2. a standing halt BLOCKS the daemon from proposing — checked by actually
     running the daemon, not by calling `status()`;
  3. clearing requires a reason, is attributed, and is appended to an audit log,
     and a corrupted halt file is treated as HALTED rather than as clear.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def isolated():
    """Point `halt` at a temp dir and return it plus a restore callable."""
    import halt
    tmp = Path(tempfile.mkdtemp(prefix="halt-"))
    saved = (halt.STATE, halt.HALT, halt.AUDIT)
    halt.STATE = tmp
    halt.HALT = tmp / "HALT.json"
    halt.AUDIT = tmp / "halt_audit.jsonl"
    return halt, tmp, saved


def restore(halt, saved, tmp):
    halt.STATE, halt.HALT, halt.AUDIT = saved
    shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("1. which verdicts halt")
    halt, tmp, saved = isolated()
    try:
        check("a clean run is not halted", not halt.status().active)
        h = halt.reconcile("healthy", True, "all auditors agree")
        check("healthy does NOT halt", not h.active)
        h = halt.reconcile("degraded", False, "1 major: vitals/V3")
        check("degraded does NOT halt — a dead dashboard is broken, not wrong",
              not h.active)
        h = halt.reconcile("broken", False, "the loop died")
        check("broken does NOT halt — operational failure is not a reason to stop "
              "burning science", not h.active)
        h = halt.reconcile("wrong", False, "bar untraceable", ["provenance/I4"])
        check("wrong HALTS", h.active, h.reason[:50])
        check("... and the halt names the finding",
              h.findings == ["provenance/I4"], str(h.findings))
    finally:
        restore(halt, saved, tmp)

    print("\n2. a standing halt blocks the daemon from proposing")
    # Run the REAL daemon in --dry-run. If the halt gate is not wired in, this
    # proposes an arm and says so; if it is, it refuses.
    halt, tmp, saved = isolated()
    try:
        halt.raise_halt("planted for the test", ["provenance/I4"])
        # The daemon's LOG directory goes with it. The test proves the gate is
        # wired in by running the REAL daemon binary, and that subprocess used to
        # append "HALT not proposing: planted for the test" to the live
        # `logs/daemon.log` -- which `status.py` then reads as a halted loop. The
        # halt file was already redirected; the log was the gap, and it is the
        # same class of bug twice: a test that writes into the live 24/7 record.
        r = subprocess.run(
            [sys.executable, str(ROOT / "pipeline" / "daemon.py"), "--dry-run"],
            cwd=ROOT, capture_output=True, text=True, timeout=180,
            env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
                 "RSIJEV_HALT": str(halt.HALT), "RSIJEV_LOGS": str(tmp)})
        out = r.stdout + r.stderr
        check("the daemon refuses while halted", "HALT" in out, f"rc={r.returncode}")
        check("... and does NOT propose an arm", "PROPOSED" not in out,
              out.strip()[-140:])
        check("... and explains how to clear it", "--clear" in out)
    finally:
        restore(halt, saved, tmp)

    print("\n3. clearing requires a reason, is attributed, and is audited")
    halt, tmp, saved = isolated()
    try:
        halt.raise_halt("planted", ["x"])
        try:
            halt.clear_halt("drhu", "   ")
            check("clearing with a blank reason is refused", False)
        except ValueError:
            check("clearing with a blank reason is refused", True)
        halt.clear_halt("drhu", "verified the bar by hand against the log")
        check("clearing with a reason works", not halt.status().active)
        audit = [json.loads(l) for l in halt.AUDIT.read_text().splitlines() if l.strip()]
        check("both the halt and the clear are in the audit log",
              [a["action"] for a in audit] == ["halted", "cleared"],
              str([a["action"] for a in audit]))
        check("the audit names who cleared it and why",
              "drhu" in json.dumps(audit[-1]) and "against the log" in json.dumps(audit[-1]),
              json.dumps(audit[-1])[:110])
    finally:
        restore(halt, saved, tmp)

    print("\n4. a halt that cannot be read is a halt, not a clearance")
    halt, tmp, saved = isolated()
    try:
        halt.HALT.write_text("{ this is not json")
        h = halt.status()
        check("a corrupt halt file reads as HALTED", h.active)
        check("... and says why", h.unreadable and "cannot be parsed" in h.reason,
              h.reason[:60])
    finally:
        restore(halt, saved, tmp)

    print("\n5. a halt does not clear itself when the condition goes away")
    halt, tmp, saved = isolated()
    try:
        halt.raise_halt("planted", ["x"])
        # The doctor now reports healthy. The halt must survive it.
        halt.reconcile("healthy", True, "all auditors agree")
        check("a green verdict does NOT clear a standing halt",
              halt.status().active,
              "a flapping invariant must not be able to resume the search")
    finally:
        restore(halt, saved, tmp)

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
