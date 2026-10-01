"""Reproduce the watchdog's inability to see itself.

Observed: watchdog pid 93422, started 12:57:05, ticks every 900s. Its own doctor
tick at 13:12:06 reported `vitals/V3 — the watchdog is alive` as FAILED, from
inside 93422. The same `pgrep -f watchdog.py` answers correctly from a shell.

A liveness check that depends on an external process-listing tool matching a
substring of a command line is fragile by construction, whatever the root cause:
the process is asking a stranger to recognise it. This measures the failure so the
fix can be justified rather than guessed at.
"""
from __future__ import annotations

import os
import subprocess
import sys

print(f"  this process   pid={os.getpid()}")
print(f"  argv           {sys.argv}")
print(f"  /proc analogue not on macOS; using ps + pgrep\n")

for pat in ("watchdog.py", "this_probe_does_not_exist.py"):
    r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    seen_self = str(os.getpid()) in r.stdout.split()
    print(f"  pgrep -f {pat!r:34s} rc={r.returncode} pids={r.stdout.split()}"
          f"  sees-self={seen_self}")

print()
# The distinguishing variable: is the match done by a child of the target?
r = subprocess.run(["ps", "-o", "pid,args", "-p", str(os.getpid())],
                   capture_output=True, text=True)
print("  ps agrees about our own argv:")
for line in r.stdout.strip().split("\n"):
    print("     ", line.strip())

print()
print("  CONCLUSION: if `sees-self=True` above, the 13:12 failure was not pgrep")
print("  being unable to match — it was the doctor running while the watchdog was")
print("  mid-restart, i.e. a real absence that self-reports. Either way the fix")
print("  is the same: a process should not need a stranger to prove it exists.")
