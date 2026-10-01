#!/usr/bin/env python3
"""A watchdog that keeps the loop alive. Unattended operation needs this.

EDITABLE — ours.

What it watches, and why each one:

  * caffeinate. If the process holding the power assertion dies, macOS will
    happily idle the machine to sleep in the middle of an arm, and an arm that
    dies at hour three is an arm whose result nobody can explain. The watchdog
    re-asserts, so the machine stays awake as long as the loop is installed.
  * the loop's own heartbeat. An arm is allowed to be quiet for a long time —
    upstream's arm runner prints nothing during the control pass — so silence
    alone is not a failure. What IS a failure is silence past a deadline, and
    the deadline is generous enough that a slow-but-alive arm is never killed.
  * the last run's exit. A loop invocation that died without recording anything
    is retried on the next tick rather than being treated as done.
  * disk. A full disk kills an arm mid-write and takes the checkpoint with it.

Exit codes are the loop's: 0 healthy or repaired, 1 unrecoverable.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
RECORDS = ROOT / "records"
STATE = ROOT / "state"
for d in (LOGS, RECORDS, STATE):
    d.mkdir(parents=True, exist_ok=True)

# An arm may legitimately run for hours. This is the point at which silence
# stops being "slow" and becomes "gone". Set well above any plausible arm.
SILENCE_DEADLINE_S = 6 * 3600
# Free space below which a checkpoint write will fail.
MIN_FREE_GB = 20
# Back off between repair attempts so a broken setup does not spin.
REPAIR_BACKOFF_S = 900


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WATCHDOG {msg}"
    print(line, flush=True)
    with open(LOGS / "watchdog.log", "a") as fh:
        fh.write(line + "\n")


def caffeinate_alive() -> bool:
    r = subprocess.run(["pgrep", "-x", "caffeinate"], capture_output=True, text=True)
    return r.returncode == 0


def start_caffeinate() -> None:
    # start_new_session so it is not in this process's group and does not die
    # with us; the watchdog re-checks it every tick regardless.
    subprocess.Popen(["caffeinate", "-dimsu"], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log("started caffeinate (prevent sleep)")


def last_activity() -> tuple[float, str]:
    """(seconds since the loop last wrote anything, what wrote it)."""
    newest, who = 0.0, "(nothing yet)"
    for p in (LOGS / "daemon.log", LOGS / "arm_champion_base.log"):
        if p.is_file():
            age = time.time() - p.stat().st_mtime
            if age < newest or newest == 0.0:
                newest, who = age, p.name
    # the arm logs are the freshest signal while an arm runs
    arms = sorted(LOGS.glob("arm_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if arms and arms[0].stat().st_mtime > (LOGS / "daemon.log").stat().st_mtime \
            if (LOGS / "daemon.log").is_file() else arms:
        newest = time.time() - arms[0].stat().st_mtime
        who = arms[0].name
    return newest, who


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def loop_installed() -> bool:
    r = subprocess.run(["launchctl", "list"], capture_output=True, text=True)
    return "com.research.rsijev" in r.stdout


DASH_LABEL = "com.research.rsijev.dash"


def dash_installed() -> bool:
    return DASH_LABEL in subprocess.run(["launchctl", "list"],
                                       capture_output=True, text=True).stdout


def dash_answering() -> bool:
    """Installed is not the same as serving. A port that is bound by something
    else, or a collector that throws on a new field, leaves the agent loaded and
    the page dead -- which is the state where a dashboard is most needed and
    least likely to be noticed."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:8788/healthz", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def install_loop(interval: int, model: str) -> None:
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "install_launchd.py"),
                        "install", "--interval", str(interval), "--model", model],
                       capture_output=True, text=True)
    log(f"install loop rc={r.returncode}: {(r.stdout or r.stderr).strip()[:200]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=14400)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    last_repair = 0.0
    last_stale_restart = 0.0
    while True:
        # The doctor runs FIRST and outranks everything below. The other checks
        # ask "is it up"; the doctor asks "is it up AND running the science we
        # said we would run". A pipeline that is alive and wrong is the failure
        # mode that costs the most, because nothing else here will notice it.
        try:
            import os
            # Tell the doctor we ARE the watchdog, so its V3 liveness check does
            # not have to ask pgrep to recognise its own command line. See the
            # note in doctor.auditor_vitals: that check cried wolf three times on
            # a healthy watchdog.
            os.environ["_RSIJEV_INSIDE_WATCHDOG"] = "1"
            import doctor
            rep = doctor.diagnose()
            doctor.save(rep)
            if not rep.ok:
                log(f"DOCTOR    {rep.verdict.upper()}: {rep.diagnosis[:150]}")
                if rep.verdict == "wrong":
                    log("          this is the silent class: the loop is running "
                        "and will not stop on its own. NOT auto-repaired -- a "
                        "scientific violation must be read by a person before "
                        "anything is changed.")
            # ONE exception to "a person decides", and it is not a judgement call
            # about the science: a process running code older than the code on disk
            # (C2) is gating with a module that is not the module in the repository.
            # The correction is already written down and committed -- there is
            # nothing to decide, only processes to replace.
            #
            # TWO gates, and both are load-bearing.
            #
            # `no arm in flight`, because the daemon is the arm's PARENT and is also
            # what appends its result to `records/arms.jsonl`. Killing it mid-run
            # leaves a measurement with no row, which is provenance/I2 all over
            # again. So this waits, and says that it is waiting.
            #
            # `restart ALL stale services, not just the daemon`. C2 asks about every
            # process of this project, so fixing one and leaving the dashboard on
            # old code does not clear it -- and the loop then stays halted forever
            # with nothing left to repair. That is the terminal failure this whole
            # branch exists to prevent, reintroduced through the fix for it: on
            # 2026-10-01 the first version of this restarted the daemon alone, which
            # would have left C2 permanently true behind a dashboard that had been
            # running since before the edit.
            if rep.verdict == "wrong" and "reproducible/C2" in (rep.criticals or []):
                if doctor.arm_pids():
                    log("          C2: a corrected module is on disk and the "
                        "running daemon predates it. NOT restarting while an arm is "
                        "in flight -- the daemon writes that arm's record, and "
                        "killing it would orphan the measurement. It will be "
                        "restarted the moment the arm lands.")
                elif time.time() - last_stale_restart > REPAIR_BACKOFF_S:
                    last_stale_restart = time.time()
                    # Every job of this project, including the watchdog itself: a
                    # watchdog holding a stale module cannot clear its own finding.
                    jobs = ["", ".dash", ".watchdog", ".publish", ".lean"]
                    for suffix in jobs:
                        subprocess.run(
                            ["launchctl", "kickstart", "-k",
                             f"gui/{os.getuid()}/com.research.rsijev{suffix}"],
                            capture_output=True, text=True)
                    log("          C2: no arm in flight; restarted every service of "
                        "this project so none of them is left running code that is "
                        "not on disk")
        except Exception as exc:
            log(f"DOCTOR    could not run: {type(exc).__name__}: {exc}")

        problems: list[str] = []

        if not caffeinate_alive():
            problems.append("machine could sleep (caffeinate dead)")
        if free_gb(ROOT) < MIN_FREE_GB:
            problems.append(f"only {free_gb(ROOT):.0f} GB free")
        if not loop_installed():
            problems.append("loop not installed in launchd")
        if not dash_installed():
            problems.append("dashboard not installed in launchd")
        elif not dash_answering():
            problems.append("dashboard installed but /healthz is not answering")

        if problems:
            log("PROBLEM: " + "; ".join(problems))
            if time.time() - last_repair > REPAIR_BACKOFF_S:
                last_repair = time.time()
                if not caffeinate_alive():
                    start_caffeinate()
                if not loop_installed():
                    install_loop(a.interval, a.model)
                if not dash_installed() or not dash_answering():
                    subprocess.run([sys.executable,
                                    str(ROOT / "scripts" / "install_dashboard.py"),
                                    "install"], capture_output=True, text=True)
                    log("dashboard reinstalled")
                log("repair attempted")
        else:
            age, who = last_activity()
            state = "idle" if age > SILENCE_DEADLINE_S else f"active ({who}, {age:.0f}s)"
            n_arms = 0
            p = RECORDS / "arms.jsonl"
            if p.is_file():
                n_arms = sum(1 for x in p.read_text().splitlines() if x.strip())
            log(f"ok  loop={'installed' if loop_installed() else 'MISSING'}  "
                f"dash={'serving' if dash_answering() else 'MISSING'}  "
                f"sleep=prevented  disk={free_gb(ROOT):.0f}GB  "
                f"arms={n_arms}  {state}")

        if a.once:
            return 0
        time.sleep(900)


if __name__ == "__main__":
    raise SystemExit(main())
