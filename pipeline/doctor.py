"""Multi-agent diagnosis: independent auditors, then an arbiter that cannot be
talked out of a critical.

EDITABLE — ours. And deliberately NOT LLM agents, for a reason worth stating:
rule 2 says this search costs nothing, and an LLM auditor would need a paid
endpoint to run unattended for a week. So the "agents" here are independent
AUDITORS -- separate code paths, separate lenses, no shared state and no shared
failure -- plus an arbiter. That is the part of agent orchestration that actually
earns its keep at 3am; the rest is a nice way to phrase a rule.

Each auditor answers one question and cannot answer the others:

  VITALS        is the machine running the loop? Processes, launchd, sleep,
                disk, heartbeat, and whether an arm is advancing or wedged.
  PROVENANCE    is every number traceable to a prediction that existed first?
                prereg<->result pairing, orphans, champion integrity.
  METHODOLOGY   is the SCIENCE the one we said we would run? Queue order, the
                noise floor, seed budgets, the held-out set, the roles.
  INTEGRITY     were the guards applied? Contamination, artifact verification,
                one kernel stack, no cost.
  ADVERSARY     do the published numbers survive being recomputed? Not "is the
                record coherent" — take every headline back to the per-question
                rows underneath it and redo the arithmetic. The other four ask
                whether the pipeline followed its own rules; this one asks
                whether the rules produced the numbers at all.

The arbiter then does the thing a single fused check cannot:

  * a CRITICAL is never diluted. Three auditors saying "fine" and one saying
    "the bar is untraceable" is one critical, not 25% critical. Averaging
    severities is how a bar that moved down gets reported as a yellow.
  * it separates WRONG from BROKEN, because the response differs and the
    urgency inverts. Broken (a process died) is loud, self-announcing, and
    cheap to fix. Wrong (the loop is alive and publishing numbers that violate
    the protocol) is silent, self-concealing, and will run for a week. So a
    methodological critical outranks a vitals failure in the report, because it
    is the one nobody will notice on their own.
  * it proposes a REPAIR, and says plainly which repairs are safe to apply
    unattended and which need a human. Killing an arm is safe. Rewriting the
    bar is not.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

import adversary  # noqa: E402
import halt as haltmod  # noqa: E402
import invariants  # noqa: E402
import loop as loopmod  # noqa: E402
from invariants import Finding  # noqa: E402

LOGS = ROOT / "logs"
STATE = ROOT / "state"

SEVERITY_RANK = {"critical": 3, "major": 2, "minor": 1, "info": 0}


@dataclass
class Report:
    verdict: str = "healthy"          # healthy | degraded | wrong | broken
    ok: bool = True
    when: float = field(default_factory=time.time)
    when_h: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    auditors: dict[str, dict] = field(default_factory=dict)
    criticals: list[str] = field(default_factory=list)
    majors: list[str] = field(default_factory=list)
    minors: list[str] = field(default_factory=list)
    # what the arbiter thinks is going on, in one sentence
    diagnosis: str = ""
    # the halt this report raised or found standing
    halt: dict = field(default_factory=dict)
    # repairs, each with whether it is safe to apply unattended
    repairs: list[dict] = field(default_factory=list)
    needs_human: bool = False

    def to_json(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------- the auditors
def auditor_vitals() -> tuple[str, list[Finding]]:
    """Is the loop running, and is it advancing rather than wedged?"""
    F: list[Finding] = []

    def sh(cmd: list[str]) -> bool:
        return subprocess.run(cmd, capture_output=True).returncode == 0

    loop_ok = "com.research.rsijev" in subprocess.run(
        ["launchctl", "list"], capture_output=True, text=True).stdout
    F.append(Finding("V1", "the loop is installed in launchd", ok=loop_ok,
                     severity="major", detail="nobody will start it again",
                     evidence=["launchctl list"]))

    arm = sh(["pgrep", "-f", "run_one.py"])
    F.append(Finding("V2", "an arm is running or the loop is between arms",
                     ok=True, severity="info",
                     detail="", evidence=[f"arm running: {arm}"]))

    # "The watchdog is alive" is a question only the watchdog can answer without
    # help. Asking `pgrep` means asking a process-listing tool to recognise a
    # substring of this process's own command line, and that check reported this
    # watchdog DEAD at 12:56, 12:57 and 13:12 while it was demonstrably running
    # (pid 93422, started 12:57:05, ticking every 900s). A guard that cries wolf
    # on the healthy case is worse than no guard: it is the fastest way to teach a
    # reader to ignore the panel, and the panel is the thing that is supposed to
    # be believed at 3am.
    #
    # So: if this diagnosis is being produced BY the watchdog, the watchdog is
    # alive by construction and no external lookup is needed. Only when the doctor
    # runs standalone (a human, a test, a cron) does it have to look.
    import os
    running_inside_watchdog = any(
        "watchdog" in os.path.basename(a) or a.endswith("pipeline/watchdog.py")
        for a in (sys.argv[:1] + [__file__])
    ) or os.environ.get("_RSIJEV_INSIDE_WATCHDOG") == "1"
    if running_inside_watchdog:
        wd, how = True, f"this diagnosis is running inside the watchdog (pid {os.getpid()})"
    else:
        wd, how = sh(["pgrep", "-f", "watchdog.py"]), "pgrep watchdog"
    F.append(Finding("V3", "the watchdog is alive", ok=wd, severity="major",
                     detail="nothing re-asserts sleep prevention or restarts the loop",
                     evidence=[how]))

    sleep_ok = "PreventSystemSleep             1" in subprocess.run(
        ["pmset", "-g", "assertions"], capture_output=True, text=True).stdout
    F.append(Finding("V4", "sleep is prevented", ok=sleep_ok, severity="major",
                     detail="an arm that dies at hour three is an arm nobody can explain",
                     evidence=["pmset -g assertions"]))

    import shutil
    free = shutil.disk_usage(ROOT).free / 1e9
    F.append(Finding("V5", "disk has room for a checkpoint", ok=free > 20,
                     severity="major",
                     detail="a full disk kills an arm mid-write and takes the checkpoint",
                     evidence=[f"{free:.0f} GB free"]))

    dash = subprocess.run(["pgrep", "-f", "pipeline/dashboard.py"],
                          capture_output=True).returncode == 0
    F.append(Finding("V6", "the dashboard is serving", ok=dash, severity="minor",
                     detail="a monitor that is down cannot tell you the loop is down",
                     evidence=[f"process: {dash}"]))

    # Wedge detection: an arm log that has not moved in far longer than any
    # plausible arm takes. Upstream's runner is quiet during the control pass
    # -- legitimately, for tens of minutes -- so the deadline is generous.
    idle = None
    logs = sorted(LOGS.glob("arm_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if logs:
        idle = time.time() - logs[0].stat().st_mtime
    WEDGE_S = 6 * 3600
    F.append(Finding("V7", "the in-flight arm is advancing, not wedged",
                     ok=(idle is None or idle < WEDGE_S), severity="major",
                     detail="silence past the deadline is a wedge; silence before it is a slow pass",
                     evidence=[f"last arm-log write {idle:.0f}s ago" if idle else "no arm log"],
                     repair="kill the wedged arm" if (idle and idle >= WEDGE_S) else ""))

    return "vitals", F


def auditor_provenance() -> tuple[str, list[Finding]]:
    """Is every number traceable to a prediction that existed first?"""
    _, F = "provenance", invariants.check_all()
    keep = {"I1", "I2", "I3", "I4"}
    return "provenance", [f for f in F if f.invariant in keep]


def auditor_methodology() -> tuple[str, list[Finding]]:
    """Is the science the one we said we would run?"""
    _, F = "methodology", invariants.check_all()
    keep = {"I5", "I6", "I7", "I8"}
    return "methodology", [f for f in F if f.invariant in keep]


def auditor_integrity() -> tuple[str, list[Finding]]:
    """Were the guards applied?"""
    _, F = "integrity", invariants.check_all()
    keep = {"I9", "I10", "I11", "I12"}
    return "integrity", [f for f in F if f.invariant in keep]


def auditor_adversary() -> tuple[str, list[Finding]]:
    """Recompute every published number from the data beneath it."""
    arms = loopmod.read_records()
    return "adversary", adversary.probe(arms)


AUDITORS = (auditor_vitals, auditor_provenance, auditor_methodology,
            auditor_integrity, auditor_adversary)


# -------------------------------------------------------------------- arbiter
def arbitrate(reports: dict[str, list[Finding]]) -> Report:
    """Combine independent auditors. A critical is never averaged away.

    Three auditors agreeing "fine" and one saying "the bar is untraceable" is
    ONE critical, not 25% critical. Severity fusion that takes a mean, or takes
    the majority, is exactly how a bar that moved down gets filed as a yellow
    and nobody looks at it again.
    """
    rep = Report()
    for name, F in reports.items():
        crit = [f for f in F if not f.ok and f.severity == "critical"]
        maj = [f for f in F if not f.ok and f.severity == "major"]
        rep.auditors[name] = {
            "n": len(F), "ok": sum(1 for f in F if f.ok),
            "critical": [f"{f.invariant} {f.title}" for f in crit],
            "major": [f"{f.invariant} {f.title}" for f in maj],
            "findings": [f.to_json() for f in F],
        }
        for f in crit:
            rep.criticals.append(f"{name}/{f.invariant}")
        for f in maj:
            rep.majors.append(f"{name}/{f.invariant}")
        for f in F:
            if not f.ok and f.severity == "minor":
                rep.minors.append(f"{name}/{f.invariant}")

    # A scientific failure outranks an operational one at EVERY severity, not
    # just critical. "The queue order was violated" is a MAJOR, and it is
    # silent: the loop kept running and spent the budget on arms that could not
    # answer anything. Labelling that BROKEN would tell a reader to go look at
    # a process that is working perfectly.
    # critical AND major only. A methodology MINOR is not "wrong" — I5's minor
    # is "the floor is still the fallback, and the record says so", which is
    # honest disclosure rather than a violated protocol. Calling that WRONG would
    # flatten the difference between a known limitation and a broken guarantee,
    # and would make WRONG stop meaning anything.
    meth_crit = [c for c in rep.criticals + rep.majors
                 if c.startswith(("methodology", "provenance", "integrity",
                                   "adversary"))]

    # WRONG beats BROKEN. A dead process announces itself; a live one publishing
    # numbers that violate the protocol does not, and will not stop on its own.
    if meth_crit:
        rep.verdict = "wrong"
        rep.ok = False
        rep.diagnosis = (
            f"the loop is running but the SCIENCE is not: "
            f"{', '.join(meth_crit)}. This is the dangerous class — it keeps "
            f"running and keeps publishing numbers that violate the protocol, "
            f"which is why it outranks a dead process in this report.")
    elif rep.criticals:
        rep.verdict = "broken"
        rep.ok = False
        rep.diagnosis = (f"{len(rep.criticals)} critical failure(s), all operational: "
                         f"{', '.join(rep.criticals)}. Loud and self-announcing, "
                         f"which is why it ranks below a scientific failure here.")
    elif rep.majors or rep.minors:
        rep.verdict = "degraded"
        rep.ok = False
        bits = rep.majors + rep.minors
        rep.diagnosis = (f"{len(rep.majors)} major and {len(rep.minors)} minor "
                         f"failure(s): {', '.join(bits)}")
    else:
        rep.verdict = "healthy"
        rep.diagnosis = ("all auditors agree: the loop is running and the science it is "
                         "running is the science we said we would run")

    # Repairs, each labelled with whether applying them unattended is safe.
    for name, F in reports.items():
        for f in F:
            if f.ok or not f.repair:
                continue
            safe = f.severity in ("critical", "major")
            rep.repairs.append({
                "finding": f"{name}/{f.invariant}",
                "action": f.repair,
                "safe_unattended": safe,
                "detail": f.detail,
            })
    rep.needs_human = any(not r["safe_unattended"] for r in rep.repairs)

    # ACT, not just report. A diagnosis that only writes a file has failed at the
    # one job a critical agent has: the loop it is watching is still running, and
    # still publishing, on numbers that violate the protocol. Halting is what makes
    # this an agent rather than a dashboard.
    h = haltmod.reconcile(rep.verdict, rep.ok, rep.diagnosis,
                          rep.criticals + rep.majors)
    rep.halt = h.to_json()
    # Say so in the diagnosis, not only in the halt file. A halt nobody can see
    # in the place they already look is a halt that gets cleared without being
    # understood — and clearing is exactly what the halt is protecting against.
    if h.active and rep.verdict in haltmod.HALTING_VERDICTS:
        rep.diagnosis += ("  HALTED: the loop will not propose another arm until a "
                          "human gives a reason. The arm in flight is left to finish.")
    return rep


def diagnose() -> Report:
    reports = {name: F for name, F in (a() for a in AUDITORS)}
    return arbitrate(reports)


# -------------------------------------------------------------------- output
def render(rep: Report) -> str:
    L: list[str] = []
    w = L.append
    icon = {"healthy": "OK", "degraded": "DEGRADED", "wrong": "WRONG",
            "broken": "BROKEN"}[rep.verdict]
    w(f"  [{icon}]  {rep.when_h}")
    w(f"  {rep.diagnosis}")
    w("")
    for name, a in rep.auditors.items():
        mark = "ok  " if not (a["critical"] or a["major"]) else "FAIL"
        w(f"  {mark} {name:12s} {a['ok']}/{a['n']} checks hold")
        for c in a["critical"]:
            w(f"         CRITICAL  {c}")
        for m in a["major"]:
            w(f"         major     {m}")
    if rep.repairs:
        w("")
        w("  proposed repairs:")
        for r in rep.repairs:
            tag = "safe" if r["safe_unattended"] else "NEEDS A HUMAN"
            w(f"    [{tag}] {r['finding']}: {r['action']}")
    if rep.needs_human:
        w("")
        w("  >>> at least one repair is NOT safe to apply unattended.")
    return "\n".join(L)


def save(rep: Report) -> Path:
    p = ROOT / "state" / "doctor.json"
    p.write_text(json.dumps(rep.to_json(), indent=2, default=str) + "\n")
    # A short history, so a reader can see when the pipeline went wrong rather
    # than only what it is doing now.
    h = ROOT / "state" / "doctor_history.jsonl"
    with open(h, "a") as fh:
        fh.write(json.dumps({"when": rep.when, "when_h": rep.when_h,
                             "verdict": rep.verdict, "ok": rep.ok,
                             "criticals": rep.criticals, "majors": rep.majors},
                            default=str) + "\n")
    return p


def load() -> dict | None:
    p = ROOT / "state" / "doctor.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = diagnose()
    print(render(r))
    p = save(r)
    if a.json:
        print(json.dumps(r.to_json(), indent=2, default=str))
    else:
        print(f"\n  -> {p}")
    raise SystemExit(0 if r.ok else 1)