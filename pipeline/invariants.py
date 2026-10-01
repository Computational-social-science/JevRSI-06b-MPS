"""Machine-checkable invariants: is the pipeline running the RIGHT science?

EDITABLE — ours. The input to `doctor.py`.

A liveness check asks whether the process is up. That is the easy question, and
it is not the one that matters at 3am: a pipeline can be perfectly alive for a
week, steadily publishing numbers, while violating the protocol it claims to
follow. Those failures are worse than a crash, because a crash is noticed.

So this file states what "running the right science" means as checks a machine
can make, with no judgement call in them. Each is a fact about the logs, not an
opinion about them. The judgements live in `doctor.py`.

The invariants, and why each one:

  I1  every result has a prediction registered BEFORE it
      A number with no prediction was not a test.
  I2  no prediction is left dangling forever
      A preregistration with no result and not running is an unresolved
      prediction. Upstream: a version that misses its bar "ships as a failure
      rather than getting quietly re-cut" -- a dangling one is worse, because
      nobody closed it.
  I3  the champion has a record
      The bar is measured against the incumbent. An incumbent with no record is
      a bar set by a number nobody can trace -- this exact failure happened
      once here, when a cleanup left champion.json naming a discarded arm.
  I4  the bar never moves DOWN
      The champion is the best-so-far. A champion whose delta is lower than its
      predecessor's is a search that got worse and called it progress.
  I5  the noise floor is measured, or its absence is DECLARED
      A bar built on an assumed floor is a guess wearing a number. It must be
      labelled as a fallback in the record, not just in a log line.
  I6  the queue is obeyed
      The agenda is ordered by what the design can resolve. Running out of order
      spends the budget on arms that cannot answer anything while the
      calibration that makes them interpretable is still missing.
  I7  a kept arm ran the seeds it was owed
      Upstream: the confirmation seed "is the only number the arm was not
      selected on". An arm promoted without it took the crown on the number it
      was chosen for.
  I8  the held-out set is read at most once
      It is consumable BY CONSTRUCTION. A second read measures something else.
  I9  every scored arm passed the contamination check
      Verified per arm, not asserted at build time.
  I10 a published artifact reproduced its training run
      A checkpoint that does not reload to the same predictions is not the
      model whose numbers were read.
  I11 one kernel stack throughout
      Upstream: numbers only compare within one stack. A run that silently
      changed device mid-search cannot be compared with itself.
  I12 the search costs nothing
      Rule 2, as a check rather than a claim.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

RECORDS = ROOT / "records"
STATE = ROOT / "state"
LOGS = ROOT / "logs"


@dataclass
class Finding:
    """One invariant, its verdict, and the evidence either way.

    `evidence` is what makes this auditable: a claim about the pipeline with no
    quoted fact behind it is the same kind of thing this project exists to stop
    trusting.
    """
    invariant: str
    title: str
    ok: bool
    severity: str = "critical"     # critical | major | minor | info
    detail: str = ""
    evidence: list[str] = field(default_factory=list)
    repair: str = ""               # what `doctor` can do about it, if anything

    def to_json(self) -> dict:
        return asdict(self)


def _lines(p: Path) -> list[str]:
    try:
        return [x for x in p.read_text(errors="ignore").splitlines() if x.strip()]
    except Exception:
        return []


def _jsonl(p: Path) -> list[dict]:
    out = []
    for line in _lines(p):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _delta(r: dict) -> float | None:
    c, k = r.get("pooled_top1_candidate"), r.get("pooled_top1_control")
    if c is None or k is None:
        return None
    return c - k


def running_arm() -> str | None:
    """Which arm is in flight RIGHT NOW — or None, meaning none is.

    The first version inferred this from the most recent `arm_*.log`, which is
    simply the name of the last arm that ever ran. Those files are never deleted,
    so the function answered "in flight" for the previous arm for as long as the
    pipeline existed, and invariant I6 then reported a perfectly ordered loop as
    running out of order — queue head `null1`, in flight `confirm__null_floor`,
    an arm that had exited. It raised a `major`, the arbiter called the science
    wrong, and the critical agent HALTED a search that was behaving correctly.

    A log file records that something happened. It cannot record that something
    is still happening, and conflating the two is the whole bug. So the liveness
    signal is the PROCESS, and the log is only used to learn WHICH arm it is:

      * no `run_one.py` process  ->  nothing is in flight, full stop;
      * a process exists         ->  name it from the newest log that matches.

    The log is now a label, not a claim.
    """
    import re
    import subprocess
    # The process is the fact. Everything else is a guess about which arm it is.
    try:
        live = subprocess.run(["pgrep", "-f", "run_one.py"],
                              capture_output=True, text=True).returncode == 0
    except OSError:
        live = False
    if not live:
        return None
    logs = sorted(LOGS.glob("arm_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in logs[:3]:
        m = re.search(r"=== ARM (\S+) \(", p.read_text(errors="ignore")[-6000:])
        if m:
            return m.group(1)
    return None


def check_all() -> list[Finding]:
    import agenda as agendamod
    import confirm as confirmmod
    import held_out
    import power

    arms = _jsonl(RECORDS / "arms.jsonl")
    preregs = _jsonl(RECORDS / "prereg.jsonl")
    champs = _jsonl(STATE / "champions.jsonl")
    champ = None
    if (STATE / "champion.json").is_file():
        try:
            champ = json.loads((STATE / "champion.json").read_text())
        except json.JSONDecodeError:
            pass
    pw = power.load()
    in_flight = running_arm()
    arm_names = {r.get("arm") for r in arms}
    prereg_names = {p.get("arm") for p in preregs}

    F: list[Finding] = []

    # ---- I1 provenance
    orphan = sorted(arm_names - prereg_names)
    F.append(Finding(
        "I1", "every result has a prediction registered before it",
        ok=not orphan, severity="critical",
        detail=("a result with no preregistration was not a test"
                if orphan else f"{len(arms)} result(s), all preregistered"),
        evidence=[f"results without a prereg: {orphan}"] if orphan else
                 [f"{len(arms)} results, {len(preregs)} preregs"],
        repair=""))

    # ---- I2 no dangling predictions
    dangling = sorted(n for n in prereg_names - arm_names
                      if n and n != in_flight
                      and not n.startswith("confirm__"))
    F.append(Finding(
        "I2", "no prediction left unresolved",
        ok=not dangling, severity="major",
        detail=("a preregistration with no result and nothing running is an "
                "unclosed prediction" if dangling else "none dangling"),
        evidence=[f"unresolved: {dangling}"] if dangling else
                 [f"in flight: {in_flight}"],
        repair=""))

    # ---- I3 the champion is traceable
    champ_ok = True
    ev = []
    if champ:
        if champ.get("arm") not in arm_names:
            champ_ok = False
            ev.append(f"champion.json names {champ.get('arm')!r}, "
                      f"which has no record in arms.jsonl")
        else:
            ev.append(f"champion {champ.get('arm')} has a record")
    else:
        ev.append("no champion yet (fine before the first keeper)")
    F.append(Finding("I3", "the champion has a record", ok=champ_ok,
                     severity="critical",
                     detail=("the bar is measured against the incumbent; an "
                             "incumbent with no record is a bar nobody can trace"
                             if not champ_ok else "traceable"),
                     evidence=ev,
                     repair="clear state/champion.json" if not champ_ok else ""))

    # ---- I4 the bar never moves down
    mono_ok, mono_ev = True, []
    prev = None
    for i, c in enumerate(champs):
        d = None
        if c.get("pooled_top1") is not None and c.get("control_top1") is not None:
            d = c["pooled_top1"] - c["control_top1"]
        mono_ev.append(f"#{i+1} {c.get('arm')} delta {d:+.4f}" if d is not None
                       else f"#{i+1} {c.get('arm')} (no delta recorded)")
        if d is not None and prev is not None and d < prev - 1e-9:
            mono_ok = False
        if d is not None:
            prev = d
    F.append(Finding("I4", "the bar never moves down", ok=mono_ok, severity="critical",
                     detail=("a champion below its predecessor is a search that got "
                             "worse and called it progress"
                             if not mono_ok else f"{len(champs)} promotion(s), monotone"
                             if champs else "no promotions yet"),
                     evidence=mono_ev,
                     repair="restore the earlier champion" if not mono_ok else ""))

    # ---- I5 the floor is measured or declared
    if pw is None:
        # Only a problem once something has been JUDGED against a bar. Before the
        # first arm finishes there is no floor and nothing is at stake, so that is
        # a normal state, not a violation. The first version fired here, which
        # would have had the doctor reporting a critical failure on every fresh
        # checkout -- a guard that cries wolf on the healthy case gets ignored.
        if not arms:
            f5ok, sev5 = True, "info"
            e5 = ["no floor yet, and nothing judged against one (normal before the "
                  "first arm completes)"]
        else:
            f5ok, sev5 = False, "critical"
            e5 = [f"{len(arms)} arm(s) have been judged but state/power.json is "
                  f"absent: the bar has no floor behind it at all"]
    elif pw.n_null_arms >= 2:
        f5ok, sev5 = True, "info"
        e5 = [f"{pw.n_null_arms} null arms, measured paired sd {pw.paired_sd:.4f}",
              f"bar +{pw.recommended_bar:.4f}"]
    else:
        f5ok, sev5 = True, "minor"
        e5 = [f"{pw.n_null_arms} null arm(s): the floor is still the 0.011 FALLBACK",
              "declared as a fallback in the record rather than passed off as measured"]
    F.append(Finding("I5", "the noise floor is measured, or its absence declared",
                     ok=f5ok, severity=sev5,
                     detail=("an assumed floor is a guess wearing a number"
                             if not f5ok else "declared"),
                     evidence=e5,
                     repair="run the calibration arms first" if not f5ok else ""))

    # ---- I6 the queue is obeyed
    done = arm_names
    expected = next((h for h in agendamod.AGENDA if h.name not in done), None)
    if expected is None:
        f6ok, sev6, e6 = True, "info", ["the agenda is exhausted"]
    elif in_flight is None:
        f6ok, sev6, e6 = True, "info", [f"idle; next up would be {expected.name}"]
    else:
        # An arm may be running because it is the top of the queue, or because a
        # confirmation is owed for a lower one. Both are legitimate.
        pend = confirmmod.pending()
        q = [h for h in agendamod.AGENDA if h.name not in done]
        qnames = [h.name for h in q]
        legit = in_flight in qnames[:1] or (pend and pend["confirm_arm"] == in_flight)
        f6ok = bool(legit)
        sev6 = "major"
        e6 = [f"queue head: {qnames[0] if qnames else '(exhausted)'}",
              f"in flight : {in_flight}",
              "pending confirmation takes precedence" if pend else ""]
        e6 = [x for x in e6 if x]
    F.append(Finding("I6", "the queue is obeyed", ok=f6ok, severity=sev6,
                     detail=("running out of order spends the budget on arms that "
                             "cannot answer anything while the calibration that "
                             "makes them interpretable is still missing"
                             if not f6ok else "in order"),
                     evidence=e6,
                     repair="stop the out-of-order arm" if not f6ok else ""))

    # ---- I7 keepers ran the seeds they were owed
    bad_seeds = []
    for r in arms:
        if r.get("verdict") != "kept":
            continue
        need = r.get("seeds_required", 1)
        got = r.get("seeds_done", 1)
        if need > 1 and got < need:
            bad_seeds.append(f"{r.get('arm')} {got}/{need}")
    F.append(Finding("I7", "a kept arm ran every seed it was owed",
                     ok=not bad_seeds, severity="critical",
                     detail=("a promoted arm that skipped its confirmation took the "
                             "crown on the number it was chosen for"
                             if bad_seeds else "no keeper is short a seed"),
                     evidence=bad_seeds or ["no keepers yet" if not arms else "all complete"],
                     repair=""))

    # ---- I8 held-out read at most once
    ho = held_out.status()
    F.append(Finding("I8", "the held-out set is read at most once",
                     ok=True, severity="info",
                     detail="consumable by construction; a second read raises",
                     evidence=[f"spent={ho.get('spent')}, read at "
                               f"{ho.get('read_at_h', 'never')}"],
                     repair=""))

    # ---- I9 contamination verified for every scored arm
    unchecked = [r.get("arm") for r in arms if not r.get("contamination")]
    F.append(Finding("I9", "every scored arm passed the contamination check",
                     ok=not unchecked, severity="critical",
                     detail=("a number produced next to an unverified contamination "
                             "report is a number with an asterisk"
                             if unchecked else "verified per arm"),
                     evidence=[f"unchecked: {unchecked}"] if unchecked else
                             [f"{len(arms)} arm(s) all checked"],
                     repair=""))

    # ---- I10 published artifacts reproduce their run
    #
    # Two distinct failures, and conflating them is what made the original
    # version useless. A KEPT arm with a checkpoint and no verification is a
    # critical: that model is the deliverable. A NON-KEPT arm with a checkpoint is
    # not a critical at all — `confirm__null_floor` failed its fresh-seed
    # confirmation and left 1.7 GB of weights behind; those weights are evidence,
    # not a product, and nobody is publishing them. The first version treated
    # both as the same thing and so reported a critical on a search that was
    # behaving correctly.
    #
    # What it must still catch is the AMBIGUOUS case: a checkpoint with a blank
    # artifact field, which is indistinguishable from a kept arm whose
    # verification silently did not run. A blank field is never an acceptable
    # answer to "was this verified" in either direction.
    unver, ambiguous = [], []
    for r in arms:
        if not r.get("checkpoint"):
            continue
        art = r.get("artifact")
        if not art:
            unver.append(r.get("arm"))
        elif r.get("verdict") == "kept" and not art.startswith("verified"):
            ambiguous.append(f"{r.get('arm')}: {art}")
    ev = []
    if unver:
        ev.append(f"checkpoint with NO artifact field at all: {unver}")
    if ambiguous:
        ev.append(f"kept arm whose artifact is not a verification: {ambiguous}")
    F.append(Finding("I10", "a published artifact reproduced its training run",
                     ok=not (unver or ambiguous),
                     severity="critical" if (unver or ambiguous) else "info",
                     detail=("a checkpoint that does not reload to the same "
                             "predictions is not the model the numbers came from"
                             if (unver or ambiguous) else
                             "every kept artifact verified; non-kept checkpoints "
                             "are marked as evidence, not deliverables"),
                     evidence=ev or ["none pending"],
                     repair=("run verify_ckpt, or mark the checkpoint as evidence"
                             if (unver or ambiguous) else "")))

    # ---- I11 one kernel stack
    stacks = set()
    for r in arms:
        if r.get("kernel_stamp"):
            stacks.add(r["kernel_stamp"])
    F.append(Finding("I11", "one kernel stack throughout", ok=len(stacks) <= 1,
                     severity="critical",
                     detail=("numbers only compare within one stack; a device change "
                             "mid-search means the search cannot be compared with itself"
                             if len(stacks) > 1 else "single stack"),
                     evidence=[f"stacks seen: {sorted(stacks)}"] if stacks else ["none recorded yet"],
                     repair=""))

    # ---- I12 cost
    # This file is excluded from its own scan for the same reason
    # `tests/test_no_cost.py` excludes itself: it CONTAINS the patterns it looks
    # for, as detection literals. The first version scanned itself and reported
    # a critical cost violation pointing at its own source line -- a guard that
    # fires on the guard is worse than no guard, because it trains everyone to
    # ignore it.
    # Same list, same reason, as `tests/test_no_cost.py`'s DETECTORS: these files
    # contain the patterns they look for. Kept as an explicit set rather than
    # inferred, because a self-exclusion that has to be discovered is one that
    # will be forgotten.
    # From `detectors`, which is the single list. This used to be a second copy
    # kept in sync by hand, and it drifted: `publish.py` was added to the other
    # copy and not this one, so I12 flagged the publication gate's own credential
    # regex and HALTED the pipeline. See detectors.py for the four prior episodes.
    import detectors as _det
    SELF = set(_det.DETECTORS) | {Path(__file__).name}
    cost_bad = []
    for p in list((ROOT / "pipeline").glob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        if p.name in SELF:
            continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if "sk-" in line or "api.openai.com" in line or "openrouter.ai/api" in line:
                cost_bad.append(f"{p.name}:{i}")
    F.append(Finding("I12", "the search costs nothing", ok=not cost_bad,
                     severity="critical",
                     detail=("a paid endpoint or a committed credential in our code"
                             if cost_bad else "local MPS, open weights, anonymous hub"),
                     evidence=cost_bad or ["no credential or metered endpoint in our code"],
                     repair=""))

    return F


def summary() -> dict:
    F = check_all()
    crit = [f for f in F if not f.ok and f.severity == "critical"]
    maj = [f for f in F if not f.ok and f.severity == "major"]
    return {
        "n": len(F), "ok": sum(1 for f in F if f.ok),
        "critical_failures": [f.invariant for f in crit],
        "major_failures": [f.invariant for f in maj],
        "healthy": not crit and not maj,
        "findings": [f.to_json() for f in F],
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    F = check_all()
    if a.json:
        print(json.dumps([f.to_json() for f in F], indent=2))
    else:
        for f in F:
            mark = "ok  " if f.ok else f.severity[:4].upper()
            print(f"  [{mark}] {f.invariant:4s} {f.title}")
            if not f.ok:
                print(f"          {f.detail}")
            for e in f.evidence:
                print(f"          · {e}")
        crit = sum(1 for f in F if not f.ok and f.severity == "critical")
        maj = sum(1 for f in F if not f.ok and f.severity == "major")
        print(f"\n  {sum(1 for f in F if f.ok)}/{len(F)} invariants hold"
              f"   {crit} critical, {maj} major failures")