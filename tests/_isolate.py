"""Keep the tests off the live control plane.

Three separate test files have now reached the real `state/HALT.json` and the
real audit trail, each for a different reason and each with the same consequence:
the unattended loop stopped, or a reader saw a halt that no real violation
raised. The pattern recurred because the isolation was re-derived per file, so
this is the one place it is defined.

What reaches out, and why:

  * `doctor.arbitrate()` gained the authority to HALT, so a test that feeds it a
    synthetic critical finding halts the live search. That is correct behaviour
    and it is exactly what the test must not trigger on the real path.
  * `doctor.diagnose()` is the real diagnosis, so it reconciles against the real
    halt path.
  * `power.STATE` and `loop.RECORDS` are module globals that several suites
    redirect into a temp dir; a suite that forgets to restore them makes the NEXT
    suite see an empty state and conclude the noise floor was never measured.

Use it as a context manager and do not hand-roll the save/restore again.
"""
from __future__ import annotations

import contextlib
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))


@contextlib.contextmanager
def isolated_halt():
    """Redirect the halt file, its audit trail and the daemon's marker.

    All three, together. The marker was the subtle one: it used to live in the
    fixed STATE directory while the halt itself was redirected, so a test shared
    the live loop's "already explained this" flag — and because the signature is
    keyed on a one-second timestamp, two runs inside the same second produced a
    collision and the test failed intermittently.
    """
    import halt as haltmod
    tmp = Path(tempfile.mkdtemp(prefix="iso-halt-"))
    saved = (haltmod.STATE, haltmod.HALT, haltmod.AUDIT)
    try:
        haltmod.STATE = tmp
        haltmod.HALT = tmp / "HALT.json"
        haltmod.AUDIT = tmp / "halt_audit.jsonl"
        yield haltmod
    finally:
        haltmod.STATE, haltmod.HALT, haltmod.AUDIT = saved
        shutil.rmtree(tmp, ignore_errors=True)


@contextlib.contextmanager
def isolated_state():
    """Redirect the module globals that hold paths, and always restore them.

    `finally`, never a hand-placed restore: a suite that restores on the happy
    path only leaves the next suite reading a temp directory, and the symptom
    appears two files later in a place that has nothing to do with the cause.
    """
    import adversary, invariants, loop, power
    mods = (loop, confirm := __import__("confirm"), power, invariants, adversary)
    saved = [(m, getattr(m, "STATE", None), getattr(m, "RECORDS", None),
              getattr(m, "ROOT", None)) for m in mods]
    tmp = Path(tempfile.mkdtemp(prefix="iso-state-"))
    try:
        for m in mods:
            if hasattr(m, "STATE"):
                m.STATE = tmp
            if hasattr(m, "RECORDS"):
                m.RECORDS = tmp
        yield tmp
    finally:
        for m, s, r, rt in saved:
            if s is not None and hasattr(m, "STATE"):
                m.STATE = s
            if r is not None and hasattr(m, "RECORDS"):
                m.RECORDS = r
            if rt is not None and hasattr(m, "ROOT"):
                m.ROOT = rt
        shutil.rmtree(tmp, ignore_errors=True)
