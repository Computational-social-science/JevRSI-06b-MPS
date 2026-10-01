#!/usr/bin/env python3
"""One screen: what the loop has done, what it is doing, what it will do.

EDITABLE — ours. Written for the morning after: no arguments, no log-diving, no
remembering which file holds what. It reads only the logs, so it cannot report
anything that is not in them.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))


def human(s: float | None) -> str:
    if s is None:
        return "—"
    s = float(s)
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.0f}m"
    return f"{s/3600:.1f}h"


def tail_lines(p: Path, n: int) -> list[str]:
    if not p.is_file():
        return []
    return [x for x in p.read_text(errors="ignore").splitlines() if x.strip()][-n:]


def main() -> int:
    from agenda import AGENDA, propose
    from loop import RECORDS, STATE, read_prereg, read_records

    arms = read_records()
    prereg = {p["arm"]: p for p in read_prereg()}
    champ = None
    cp = STATE / "champion.json"
    if cp.is_file():
        champ = json.loads(cp.read_text())

    print("=" * 72)
    print(f"RSI-Jev on Qwen3-0.6B — status {time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 72)

    # --- liveness
    running = subprocess.run(["pgrep", "-f", "run_one.py"],
                             capture_output=True).returncode == 0
    loop = subprocess.run(["launchctl", "list"], capture_output=True, text=True).stdout
    in_launchd = "com.research.rsijev" in loop
    sleep_ok = "PreventSystemSleep" in subprocess.run(
        ["pmset", "-g", "assertions"], capture_output=True, text=True).stdout
    wd = "RUNNING" if subprocess.run(["pgrep", "-f", "watchdog.py"],
                                     capture_output=True).returncode == 0 else "DEAD"
    print(f"\nloop       : {'in launchd' if in_launchd else 'NOT INSTALLED'}"
          f"   arm: {'running' if running else 'idle'}")
    dash = "SERVING" if subprocess.run(
        ["pgrep", "-f", "pipeline/dashboard.py"],
        capture_output=True).returncode == 0 else "DEAD"
    print(f"sleep      : {'prevented' if sleep_ok else 'NOT PREVENTED'}"
          f"   watchdog: {wd}   dashboard: {dash} (:8788)")

    # --- power
    pw_p = STATE / "power.json"
    if pw_p.is_file():
        pw = json.loads(pw_p.read_text())
        src = ("measured" if pw["n_null_arms"] >= 2 else "FALLBACK")
        print(f"power      : {pw['n_null_arms']} null arms, sd {pw['paired_sd']:.4f} "
              f"({src})   bar +{pw['recommended_bar']:.4f}")
    else:
        print("power      : not yet measured (first null arm still running)")

    # --- results
    print(f"\narms       : {len(arms)} of {len(AGENDA)} complete")
    if arms:
        print(f"\n{'arm':26s} {'Δ vs control':>13s} {'expected':>10s}  verdict")
        print("-" * 72)
        for a in arms:
            c, k = a.get("pooled_top1_candidate"), a.get("pooled_top1_control")
            d = f"{c - k:+.4f}" if c is not None and k is not None else "—"
            e = prereg.get(a.get("arm"), {}).get("expected_effect")
            e_s = f"{e:+.4f}" if isinstance(e, (int, float)) else "—"
            und = " *" if (pw_p.is_file() and isinstance(e, (int, float)) and e
                           and a.get("verdict") != "kept"
                           and abs(e) < json.loads(pw_p.read_text())["recommended_bar"]) else ""
            print(f"{a.get('arm',''):26s} {d:>13s} {e_s:>10s}  {a.get('verdict')}{und}")
        print("\n  * expected effect is below what this design can detect, so a null")
        print("    means 'no effect we could see', not 'no effect'.")

    if champ:
        print(f"\nchampion   : {champ.get('arm')}  top1 {champ.get('pooled_top1')}"
              f"  (control {champ.get('control_top1')})")

    # --- what is next
    nxt = propose(arms)
    print(f"\nnext arm   : {nxt.name}  ({nxt.axis})")
    print(f"             {nxt.change[:90]}")

    # --- the most recent activity
    lines = tail_lines(ROOT / "logs" / "daemon.log", 6)
    if lines:
        print("\nrecent:")
        for x in lines:
            print("  " + x[22:] if x.startswith("[") else "  " + x)
    hb = tail_lines(ROOT / "logs" / "watchdog.log", 1)
    if hb:
        print("  " + hb[0][22:])

    # --- artifacts
    print(f"\nrecord     : {ROOT/'versions'/'current.md'}")
    print(f"trajectory : {ROOT/'state'/'trajectory.html'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
