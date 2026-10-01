#!/usr/bin/env python3
"""Install the dashboard as a launchd agent so it outlives a reboot.

Separate from the loop's own agent on purpose. They fail for different reasons
and must be repairable separately: the loop dies when an arm crashes or the
corpus goes missing, the dashboard dies when a port is taken or a file moves.
One label each means the watchdog can restore whichever is down without
restarting both, and a reader can tell from `launchctl list` which half of the
system is actually absent.

`KeepAlive`, unlike the loop's `StartInterval`: a dashboard has no natural
"completion", so there is no state in which not running is correct. It is meant
to be up, and a crash is a crash.

Binds 127.0.0.1 only. The loop's arms, their predictions and the research record
are not something to expose on a routable interface by default, and the page
carries paths and numbers that belong on the machine that produced them.
"""
from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABEL = "com.research.rsijev.dash"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
PORT = 8788


def build() -> bytes:
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    d = {
        "Label": LABEL,
        "ProgramArguments": [str(ROOT / ".venv" / "bin" / "python"),
                             str(ROOT / "pipeline" / "dashboard.py")],
        "RunAtLoad": True,
        "KeepAlive": True,
        "WorkingDirectory": str(ROOT),
        "StandardOutPath": str(logs / "dashboard.out.log"),
        "StandardErrorPath": str(logs / "dashboard.err.log"),
        "EnvironmentVariables": {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(Path.home()),
            "RSIJEV_DASH_PORT": str(PORT),
            "PYTHONUNBUFFERED": "1",
        },
        "ProcessType": "Background",
        "LowPriorityIO": True,
        # The dashboard is I/O-light: it reads a few small files every ten
        # seconds. It must never compete with an arm for the GPU or the disk.
        "Nice": 10,
    }
    return plistlib.dumps(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["install", "uninstall", "status", "print"])
    a = ap.parse_args()

    if a.action == "print":
        sys.stdout.buffer.write(build())
        return 0

    if a.action == "install":
        PLIST.parent.mkdir(parents=True, exist_ok=True)
        PLIST.write_bytes(build())
        domain = f"gui/{os.getuid()}"
        subprocess.run(["launchctl", "bootout", f"{domain}/{LABEL}"], capture_output=True)
        r = subprocess.run(["launchctl", "bootstrap", domain, str(PLIST)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            r2 = subprocess.run(["launchctl", "load", "-w", str(PLIST)],
                                capture_output=True, text=True)
            if r2.returncode != 0:
                print(f"install failed: {r.stderr.strip()} / {r2.stderr.strip()}",
                      file=sys.stderr)
                return 1
        # Prove it answers, rather than reporting success because launchd said so.
        import time
        import urllib.request
        for _ in range(20):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/healthz",
                                            timeout=1) as resp:
                    if resp.status == 200:
                        print(f"installed {PLIST}")
                        print(f"  http://127.0.0.1:{PORT}  (healthz OK, "
                              f"loopback only)")
                        print(f"  logs: {ROOT/'logs'}/dashboard.*.log")
                        return 0
            except Exception:
                time.sleep(0.5)
        print("installed, but /healthz did not answer — see logs/dashboard.err.log",
              file=sys.stderr)
        return 1

    if a.action == "uninstall":
        subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"],
                       capture_output=True)
        subprocess.run(["launchctl", "unload", "-w", str(PLIST)], capture_output=True)
        if PLIST.exists():
            PLIST.unlink()
        print(f"removed {LABEL}")
        return 0

    r = subprocess.run(["launchctl", "list"], capture_output=True, text=True)
    found = [x for x in r.stdout.splitlines() if LABEL in x]
    print("launchd:", found[0].strip() if found else "not loaded")
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/status",
                                    timeout=2) as resp:
            import json
            d = json.loads(resp.read())
            print(f"serving:  yes  ({len(d)} fields, {d['n_arms_done']}/{d['n_arms_total']} arms)")
    except Exception as exc:
        print(f"serving:  NO — {type(exc).__name__}: {exc}")
    err = ROOT / "logs" / "dashboard.err.log"
    if err.is_file() and err.stat().st_size:
        print("\n--- last errors ---")
        for x in err.read_text(errors="ignore").splitlines()[-5:]:
            print("  " + x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
