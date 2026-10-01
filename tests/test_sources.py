#!/usr/bin/env python3
"""Does the domestic-mirror rule actually hold, on every path that can download?

A standing rule that lives only in a plist is a rule that holds for launchd and
for nothing else: a hand-run arm, a test, the corpus builder, a future entry
point somebody writes. All of those reach the Hub, and all of them would quietly
use the origin. So the rule is checked where it is easy to break rather than
where it is easy to see.

Two failure modes matter and they are opposite:

  * the endpoint is NOT applied, and the search silently depends on reaching
    huggingface.co — which is a availability question nobody asked;
  * the endpoint is applied but the cost guard then flags the mirror as an
    unexpected host. **A guard that punishes compliance is worse than a guard
    that is absent**, because the only rational response is to disable it.
"""
from __future__ import annotations

import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

MIRROR = "https://hf-mirror.com"
FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    print("1. the code path every entry point shares")
    for m in list(sys.modules):
        if m in ("dev", "huggingface_hub", "transformers"):
            del sys.modules[m]
    os.environ.pop("HF_ENDPOINT", None)
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, 'pipeline');"
         "import os, dev;"
         "print(os.environ.get('HF_ENDPOINT',''));"
         "print(dev.hf_endpoint())"],
        cwd=ROOT, capture_output=True, text=True)
    out = r.stdout.strip().splitlines()
    check("importing dev sets HF_ENDPOINT when unset",
          bool(out) and out[0] == MIRROR, f"got {out[0] if out else '(none)':40s} {r.stderr[-80:]}")
    check("hf_endpoint() reports the mirror", len(out) > 1 and out[1] == MIRROR,
          out[1] if len(out) > 1 else "")

    print("\n2. an explicit HF_ENDPOINT still wins (one-off origin fetches)")
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, 'pipeline');"
         "import dev; print(dev.hf_endpoint())"],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "HF_ENDPOINT": "https://huggingface.co"})
    check("an explicit endpoint is not overridden",
          r.stdout.strip() == "https://huggingface.co", r.stdout.strip())

    print("\n3. the corpus builder, which runs by hand and imports dev never")
    src = (ROOT / "scripts" / "build_corpus.py").read_text()
    check("build_corpus.py sets the endpoint itself", MIRROR in src)
    # Import it for real rather than exec'ing a slice: an `exec` of the first
    # `def main` gives a truncated module, so a failure here would be the probe's
    # fault and not the script's. The assertion is also the one that matters --
    # the endpoint is set at MODULE level, above every import of huggingface_hub,
    # so it cannot be set too late.
    env = {k: v for k, v in os.environ.items() if k != "HF_ENDPOINT"}
    r = subprocess.run(
        [sys.executable, "-c",
         "import runpy, os, sys;"
         "sys.argv=['build_corpus.py'];"
         "\ntry:\n runpy.run_path('scripts/build_corpus.py', run_name='not_main')\n"
         "except SystemExit:\n pass\n"
         "print('ENDPOINT=' + os.environ.get('HF_ENDPOINT',''))"],
        cwd=ROOT, capture_output=True, text=True, env=env, timeout=120)
    check("... and importing it sets the endpoint",
          ("ENDPOINT=" + MIRROR) in r.stdout,
          (r.stdout.strip().splitlines() or [""])[-1][:70] + r.stderr[-60:])
    # Match an actual import STATEMENT, not the word in a comment -- the previous
    # version searched for the bare string and found its own explanatory comment
    # above the assignment, so the check failed on correct code.
    import re as _re
    hub_import = _re.search(r"^\s*(?:import\s+huggingface_hub|from\s+huggingface_hub)",
                            src, _re.M)
    at = src.index("_os.environ.setdefault(\"HF_ENDPOINT\"")
    check("... and the assignment precedes every hub import",
          hub_import is None or at < hub_import.start(),
          "no hub import in this file" if hub_import is None
          else f"import at offset {hub_import.start()}, setdefault at {at}")

    print("\n4. launchd: launchd does not inherit a shell's environment")
    base = Path.home() / "Library" / "LaunchAgents"
    plists = sorted(base.glob("com.research.rsijev*.plist"))
    check("the launchd plists exist", bool(plists), f"{len(plists)} found")
    for p in plists:
        d = plistlib.loads(p.read_bytes())
        got = d.get("EnvironmentVariables", {}).get("HF_ENDPOINT")
        check(f"{p.stem.replace('com.research.rsijev.','') or 'daemon'}: endpoint set",
              got == MIRROR, str(got))

    print("\n5. the cost guard must ALLOW the mirror, not flag it")
    src = (ROOT / "tests" / "test_no_cost.py").read_text()
    check("hf-mirror.com is in the allowlist", "hf-mirror.com" in src)
    r = subprocess.run([sys.executable, "tests/test_no_cost.py"],
                       cwd=ROOT, capture_output=True, text=True)
    check("the cost guard passes", r.returncode == 0,
          [l for l in r.stdout.splitlines() if "FAIL" in l][:1] or "")

    print("\n6. and the invariant version of the same rule")
    import invariants
    f = next((x for x in invariants.check_all() if x.invariant == "I12"), None)
    check("I12 holds with the mirror configured", f is not None and f.ok,
          f.detail[:60] if f else "I12 missing")

    print("\n7. it is visible where a reader will look")
    src = (ROOT / "pipeline" / "dashboard.html").read_text()
    check("the dashboard shows the endpoint", "hf_endpoint" in src)
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0,'pipeline'); import dashboard;"
         "d = dashboard.collect(); print(d['sources']['hf_endpoint'])"],
        cwd=ROOT, capture_output=True, text=True)
    check("the collector reports it", MIRROR in r.stdout, r.stdout.strip()[:60])

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
