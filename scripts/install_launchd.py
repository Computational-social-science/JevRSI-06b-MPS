#!/usr/bin/env python3
"""Install / remove the launchd job that runs the loop unattended.

EDITABLE — ours.

`StartInterval`, not `KeepAlive`. The difference matters: KeepAlive restarts the
job the instant it exits, so a finished arm and a crash look identical and a
broken setup spins at full speed writing the same error. StartInterval runs one
invocation per period and lets a non-zero exit sit until the next tick, which is
what a research loop wants — a broken setup is retried loudly, a completed arm
is not re-run.

Exit codes, from pipeline/daemon.py: 0 = an arm ran (kept, rejected,
needs_repair or error are all successful loop iterations), 1 = the loop could
not start (no corpus, no backbone).
"""
from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABEL = "com.research.rsijev"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def build_plist(interval: int, model: str) -> bytes:
    py = str(ROOT / ".venv" / "bin" / "python")
    daemon = str(ROOT / "pipeline" / "daemon.py")
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
        "HOME": str(Path.home()),
        # launchd gives a minimal environment; without these the venv's python
        # cannot find its own site-packages and the job dies at import.
        "RSIJEV_MODEL": model,
        "PYTHONUNBUFFERED": "1",
    }
    d = {
        "Label": LABEL,
        "ProgramArguments": [py, daemon, "--model", model],
        "StartInterval": interval,
        "RunAtLoad": True,
        "StandardOutPath": str(logs / "launchd.out.log"),
        "StandardErrorPath": str(logs / "launchd.err.log"),
        "WorkingDirectory": str(ROOT),
        "EnvironmentVariables": env,
        # A research loop should not be killed for being idle-ish, and should not
        # survive logout. RunAtLoad + StartInterval is the whole policy.
        "ProcessType": "Background",
        "LowPriorityIO": True,
        "Nice": 5,
    }
    return plistlib.dumps(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["install", "uninstall", "status", "print"])
    ap.add_argument("--interval", type=int, default=14400,
                    help="seconds between arms; default 4h")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    a = ap.parse_args()

    if a.action == "print":
        sys.stdout.buffer.write(build_plist(a.interval, a.model))
        return 0

    if a.action == "install":
        PLIST.parent.mkdir(parents=True, exist_ok=True)
        PLIST.write_bytes(build_plist(a.interval, a.model))
        # bootout first: an existing job with an old plist keeps running the old
        # program until it is explicitly unloaded.
        subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"],
                       capture_output=True)
        r = subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(PLIST)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"bootstrap failed: {r.stderr.strip()}", file=sys.stderr)
            # fall back to the legacy per-user domain, which is what older macOS
            # and some sandboxes accept
            r2 = subprocess.run(["launchctl", "load", "-w", str(PLIST)],
                                capture_output=True, text=True)
            if r2.returncode != 0:
                print(f"load also failed: {r2.stderr.strip()}", file=sys.stderr)
                return 1
        print(f"installed {PLIST}")
        print(f"  interval: {a.interval}s ({a.interval/3600:.1f}h)")
        print(f"  model   : {a.model}")
        print(f"  logs    : {ROOT/'logs'}")
        return 0

    if a.action == "uninstall":
        subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"],
                       capture_output=True)
        subprocess.run(["launchctl", "unload", "-w", str(PLIST)], capture_output=True)
        if PLIST.exists():
            PLIST.unlink()
        print(f"removed {LABEL}")
        return 0

    # status
    r = subprocess.run(["launchctl", "list"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if LABEL in line:
            print("launchd:", line.strip())
            break
    else:
        print("launchd: not loaded")
    dl = ROOT / "logs" / "daemon.log"
    if dl.is_file():
        print("\n--- last 25 lines of daemon.log ---")
        for line in dl.read_text().splitlines()[-25:]:
            print(line)
    st = ROOT / "state" / "champion.json"
    if st.is_file():
        import json
        c = json.loads(st.read_text())
        print(f"\nchampion: {c.get('arm')}  top1={c.get('pooled_top1')}")
    ap_path = ROOT / "records" / "arms.jsonl"
    if ap_path.is_file():
        n = sum(1 for x in ap_path.read_text().splitlines() if x.strip())
        print(f"arms run: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
