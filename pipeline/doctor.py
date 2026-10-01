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
def arm_pids() -> list[str]:
    """pids of arms actually running.

    The pattern includes the flags a real invocation carries, because
    `pgrep -f run_one.py` also matches any shell whose command line happens to
    mention the file -- including the ones grepping for it while diagnosing this
    very question. Same trap, same fix as the watchdog check below.

    A named function so a test can substitute a process list: a check that can
    only be exercised by really starting two arms is a check nobody runs.
    """
    r = subprocess.run(["pgrep", "-f", "pipeline/run_one.py --arm"],
                       capture_output=True, text=True)
    return [p for p in r.stdout.split() if p.strip()]


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

    pids = arm_pids()
    F.append(Finding("V2", "at most one arm runs at a time",
                     ok=len(pids) <= 1, severity="critical",
                     detail=("two arms writing the same records/<arm>/ and "
                             "ckpt/<arm>/ at once is the duplicate-measurement "
                             "failure this project has already paid for once -- two "
                             "arm names, one run, left in records/ where a reader "
                             "takes them for points of a spread. This check was "
                             "hardcoded ok=True and therefore blind to exactly that: "
                             "two null2 processes were live for 20 minutes on "
                             "2026-10-01 and this panel could not see them."),
                     evidence=[f"arm pids: {pids or 'none'}"]))

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
    keep = {"I9", "I10", "I11", "I12", "I13"}
    return "integrity", [f for f in F if f.invariant in keep]


def auditor_adversary() -> tuple[str, list[Finding]]:
    """Recompute every published number from the data beneath it."""
    arms = loopmod.read_records()
    return "adversary", adversary.probe(arms)


def _pids_named(*frags: str) -> list[tuple[int, float]]:
    """(pid, absolute_start_time) for processes whose command matches a fragment.

    macOS `ps` has no `etimes` -- that column is Linux-only -- so the first
    version of this helper raised on this machine, the lens did nothing, and the
    doctor reported healthy while the daemon gated with a four-times-weaker bar.
    A guard that cannot run on the platform it runs on is worse than no guard,
    because it is indistinguishable from one.
    """
    out: list[tuple[int, float]] = []
    try:
        r = subprocess.run(["ps", "-eo", "pid=,etime=,command="],
                           capture_output=True, text=True, timeout=20)
    except Exception:
        return out
    now = time.time()
    for line in r.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or "ps -eo" in line:
            continue
        pid, etime, cmd = parts
        if not any(f in cmd for f in frags):
            continue
        days, etime = 0, etime
        if "-" in etime:
            d, _, etime = etime.partition("-")
            days = int(d or 0)
        bits = [float(x) for x in etime.split(":")]
        if len(bits) == 3:
            secs = bits[0] * 3600 + bits[1] * 60 + bits[2]
        else:
            secs = bits[-1] + (bits[0] * 60 if len(bits) == 2 else 0)
        try:
            out.append((int(pid), now - (days * 86400 + secs)))
        except ValueError:
            continue
    return out


def self_processes() -> list[tuple[int, float, str]]:
    """(pid, absolute_start_time, script) for every live process of THIS project.

    The dashboard, the watchdog, the publisher and the loop are all long-lived and
    all read `state/`. A module corrected on disk reaches none of them until they
    are restarted, so a check that only asks about the loop is blind to the other
    three -- and the dashboard served HTTP 200 with `/api/status` returning 500
    for hours that way, which looks exactly like a working system to anyone
    checking that the page loads.

    The script name is returned so a finding can say WHICH service is stale
    instead of quoting a pid, because the fix is a restart and the restart needs
    a name.
    """
    out: list[tuple[int, float, str]] = []
    try:
        r = subprocess.run(["ps", "-eo", "pid=,etime=,command="],
                           capture_output=True, text=True, timeout=20)
    except Exception:
        return out
    now = time.time()
    for line in r.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or "ps -eo" in line:
            continue
        pid, etime, cmd = parts
        if f"{ROOT}/pipeline/" not in cmd:
            continue
        days, etime = 0, etime
        if "-" in etime:
            d, _, etime = etime.partition("-")
            days = int(d or 0)
        bits = [float(x) for x in etime.split(":")]
        secs = (bits[0] * 3600 + bits[1] * 60 + bits[2]) if len(bits) == 3 \
            else (bits[0] * 60 + bits[1]) if len(bits) == 2 else bits[0]
        script = ""
        for tok in cmd.split():
            if tok.endswith(".py") and "/pipeline/" in tok:
                script = Path(tok).name
        out.append((int(pid), now - (days * 86400 + secs), script or "?"))
    return out


def auditor_reproducible() -> tuple[str, list[Finding]]:
    """Could a stranger clone this and rerun it?

    Run `tests/test_conventions.py` as a SUBPROCESS and report its result, rather
    than reimplementing any of it. A convention checker that exists only in the
    test suite is a wish rather than a rule: it runs when someone remembers to
    run it, and never at 3am. Reimplementing the checks here would create a
    second version to drift, which is the failure this project has now collected
    five of.
    """
    import os
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "test_conventions.py")],
        cwd=ROOT, capture_output=True, text=True, timeout=300, env=env)
    fails = [l.strip() for l in r.stdout.splitlines() if l.strip().startswith("FAIL")]
    ev = (fails[:4] or ["all conventions hold"])
    # Does the RUNNING loop use the science currently on disk? A long-lived process
    # holds its modules in memory, so correcting power.py or loop.py does NOT reach
    # it. This happened for real: the bar was corrected on disk to 0.188 while the
    # daemon went on gating at 0.045, and nothing said so — the pipeline reported
    # healthy throughout. For a 24/7 loop that is the most dangerous kind of quiet:
    # the file on disk and the science being applied are two different things, and
    # only one of them is running.
    #
    # It is not only the loop. The dashboard, the watchdog and the publisher are
    # long-lived too, and they read the same state. On 2026-10-01 the dashboard
    # kept serving its page while `/api/status` returned HTTP 500 --
    # `PowerResult.__init__() got an unexpected keyword argument 'do_nothing_mean'`
    # -- because it was started three hours before `power.json` grew a field. The
    # same failure had already killed the watchdog silently at 14:37, 14:52, 15:07
    # and 17:56 ("DOCTOR could not run: TypeError"), four times, with the watchdog
    # logging each one and carrying on. A page that loads and a job that keeps
    # running are not evidence that they are working.
    #
    # So the check asks about EVERY process of this project, not the two the loop
    # happens to own. Erring toward reporting is right: a false positive is a line
    # of text and a restart, and it clears itself the moment the service is
    # restarted.
    stale = []
    for pid, start, script in self_processes():
        for mod in ("pipeline/power.py", "pipeline/loop.py"):
            f_ = ROOT / mod
            if f_.exists() and start and f_.stat().st_mtime > start:
                stale.append(f"{mod} edited after {script} (pid {pid}) started")
    # The evidence is the list of stale processes. Without it the finding names a
    # condition ("the science on disk is not the science running") and not a thing
    # to restart, and the repair — which is a list of job labels — has nothing to
    # act on.
    live = [f"{script} (pid {pid}, {(time.time() - start) / 60:.0f} min)"
            for pid, start, script in self_processes()]
    findings = [Finding(
        "C2", "the RUNNING loop uses the science currently on disk",
        ok=not stale, severity="critical" if stale else "info",
        detail=("the loop is gating with modules older than the ones on disk: "
                + "; ".join(stale[:3]) if stale
                else "every running process postdates power.py and loop.py"),
        evidence=stale[:6] or live[:6],
        repair=("restart the services named above so they run the code on disk; "
                "the daemon's restart waits for the arm in flight to land, because "
                "the daemon is what records that arm's result" if stale else ""))]
    findings.append(Finding(
        "C1", "the repository is reproducible from a clean clone",
        ok=r.returncode == 0,
        severity="major" if r.returncode else "info",
        detail=("a hardcoded path, an unpinned dependency, a missing licence or "
                "non-English documentation means the numbers cannot be re-derived"
                if r.returncode else "conventions hold"),
        evidence=ev,
        repair="fix the convention violations; see tests/test_conventions.py"))
    return "reproducible", findings


AUDITORS = (auditor_vitals, auditor_provenance, auditor_methodology,
            auditor_integrity, auditor_adversary, auditor_reproducible)


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
                                   "adversary", "reproducible"))]

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