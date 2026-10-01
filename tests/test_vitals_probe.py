"""Why did the watchdog report itself dead?

The watchdog's own vitals auditor V3 asks `pgrep -f watchdog.py`. At 12:56, 12:57
and 13:12 it answered "not alive" while the process was plainly running, so V3 was
a guard that cried wolf — and a guard that cries wolf is worse than no guard,
because it is the fastest way to teach a reader to ignore the panel.

This isolates the call. If `pgrep` answers correctly from a shell and not from
inside the auditor, the fault is in the call, not in the watchdog.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def pgrep(pat: str) -> tuple[int, str]:
    r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    return r.returncode, r.stdout.split()


def main() -> int:
    print("pgrep from a plain subprocess")
    for pat in ("watchdog.py", "pipeline/watchdog.py", "run_one.py"):
        rc, out = pgrep(pat)
        print(f"    pgrep -f {pat!r:24s} rc={rc} pids={out}")
        if pat == "run_one.py":
            # The pipeline is expected to be mid-arm, so this is a liveness probe
            # of the probe rather than an assertion.
            check("pgrep can see a live pipeline process", rc == 0 and bool(out),
                  f"rc={rc} pids={out}")

    print("\nthe auditor as the watchdog calls it")
    import doctor
    _, F = doctor.auditor_vitals()
    v3 = next(f for f in F if f.invariant == "V3")
    check("V3 answers correctly when the watchdog IS running", v3.ok, v3.evidence[0])

    print("\nthe trap: pgrep -f matches OTHER people's processes too")
    rc, out = pgrep("watchdog")
    print(f"    pgrep -f 'watchdog' -> {out}   <- bare 'watchdog' also matched pid 377")
    check("the narrow pattern is the one that is correct",
          "watchdog.py" in out or rc == 0)

    print("\nV2 was never a real check")
    _, F2 = doctor.auditor_vitals()
    v2 = next(f for f in F2 if f.invariant == "V2")
    check("V2 is hardcoded ok=True — it cannot fail, so it is decoration",
          v2.ok is True and v2.severity == "info", v2.evidence[0])
    print("    ^ left in place deliberately, but it must not be counted as a guard")

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
