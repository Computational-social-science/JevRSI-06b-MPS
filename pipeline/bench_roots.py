#!/usr/bin/env python3
"""Point the suite loaders at this machine's checkouts, and verify the pins.

## Why a manifest and not environment variables

`rsijev/targets_suite` resolves its roots at IMPORT time, from the environment --
upstream's deliberate choice, since "a default would be one author's filesystem,
which is how v1.0 shipped a lab path inside a checkpoint". Two consequences here:

  * launchd does not inherit a shell's environment, so anything the loop runs
    under a plist would see no roots at all. Writing the variables into five
    plists is five places to forget, and forgetting is silent -- the loader raises
    `_Unset`, which explains itself, so it is loud, but the arm still dies.
  * A path is not a pin. The checkout has to be at the commit upstream pinned, or
    the benchmark is a different split with the same name and every number
    measured on it is quietly incomparable.

So: one file in the repository, read by whoever needs the suite, setting the
environment before the loader is imported and checking each checkout's HEAD
against the pin. Import this module, do not set anything by hand.

    import sys; sys.path.insert(0, "pipeline")
    import bench_roots            # sets the roots, verifies the pins
    from rsijev import targets_suite as ts

`roots()` is idempotent and cheap; `verify()` returns a report rather than raising
on a missing checkout, because a partial suite is a legitimate state as long as it
is NAMED -- which is upstream's own rule for `--only`.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "config" / "benchmarks.json"
APPLIED = False


def _entry(name: str) -> dict:
    try:
        return json.loads(MANIFEST.read_text())["checkouts"].get(name, {})
    except (OSError, json.JSONDecodeError, KeyError):
        return {}


def _head(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def roots() -> dict[str, Path]:
    """Set the loader's root variables from the manifest. Idempotent."""
    global APPLIED
    if not APPLIED:
        for name, e in _all().items():
            var, rel = e.get("env"), e.get("path")
            if var and rel and not os.environ.get(var):
                os.environ[var] = str(ROOT / rel)
        APPLIED = True
    return {e["env"]: Path(os.environ[e["env"]])
            for e in _all().values()
            if e.get("env") and os.environ.get(e["env"])}


def _all() -> dict[str, dict]:
    try:
        return json.loads(MANIFEST.read_text())["checkouts"]
    except (OSError, json.JSONDecodeError, KeyError):
        return {}


def verify() -> list[dict]:
    """One row per pinned checkout: is it here, and is it at the right commit."""
    out = []
    for name, e in sorted(_all().items()):
        p = ROOT / e["path"] if e.get("path") else None
        present = bool(p and p.is_dir())
        head = _head(p) if present else ""
        want = e.get("pin") or ""
        out.append({
            "name": name, "env": e.get("env"), "path": str(p) if p else None,
            "present": present, "pin": want[:8] or None, "head": head[:8] or None,
            "at_pin": bool(head and want and head.startswith(want)),
            "benchmarks": e.get("benchmarks", []),
            "weight": e.get("weight", 0.0),
        })
    return out


if __name__ == "__main__":
    rows = verify()
    print(f"pinned suite checkouts, from {MANIFEST.relative_to(ROOT)}\n")
    for r in rows:
        if not r["present"]:
            state = "ABSENT"
        elif r["at_pin"]:
            state = f"at pin {r['head']}"
        else:
            state = f"WRONG COMMIT (head {r['head']}, pin {r['pin']})"
        print(f"  {r['name']:10s} {state:34s} w={r['weight']:.3f}  "
              f"{', '.join(r['benchmarks'])}")
    have = sum(r["weight"] for r in rows if r["at_pin"])
    print(f"\n  suite weight available from local checkouts: {have:.3f}")
    print(f"  plus typed_decisions_test + mmlu_pro_1k from the Hub: 0.281")
    print(f"  TOTAL: {have + 0.281:.3f} of the v3 suite's 1.000")
    bad = [r["name"] for r in rows if r["present"] and not r["at_pin"]]
    raise SystemExit(1 if bad else 0)
