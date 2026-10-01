"""Lineage: which offspring survived, which were discarded, and which never ran.

## What is borrowed, and what is not

One IDEA is borrowed, from karpathy's `autoresearch`: that an experiment log needs
**three** outcomes rather than two, because "it ran and it lost" and "it never
ran" are different facts and collapsing them is how a search record starts lying.
That distinction is his, it is a good one, and it is credited here.

**Nothing else is taken.** No code, no file format, no data schema, no metric, no
hardware assumption, no training recipe, no build or run tooling. The
implementation here is this project's own, and it has to be: autoresearch edits a
source file and rewinds a git branch, whereas this pipeline trains from a frozen
recipe, gates on a measured noise floor, and must keep an audit trail that survives
its own author. Those are different problems with different shapes, and importing
theirs would import assumptions this project cannot make.

So the three states here are `survived` / `discarded` / `errored` over arms in
`records/arms.jsonl` — not a TSV, not their columns, not their vocabulary. An
earlier draft of this docstring quoted their example rows verbatim; that was
removed, because quoting someone's data format while claiming to borrow only their
idea is how a "we wrote this ourselves" claim quietly becomes untrue.

**A discard is a result.** The arm ran, produced a valid number, and that number
said "not better than its parent". It is evidence and it belongs in the trajectory
exactly as much as a keep does. Only a crash is the absence of evidence.

This project had been conflating the two, and the cost was concrete rather than
cosmetic:

  * `confirm__null_floor` failed its fresh-seed confirmation — +0.0435 on a seed it
    was never selected on, against +0.0665 on the one it was — and left **1.7 GB
    of weights** on disk. A discarded offspring was occupying space as though it
    were still a candidate parent, and invariant I10 read its blank `artifact`
    field as an unverified publication and halted the search.
  * At one arm per ~100 minutes that is ~1.7 GB per 100 minutes, unbounded, for
    models that can never be published.

So: **weights are lineage, not evidence.** A surviving offspring's weights are
the deliverable; a discarded one's are the measurement's already-written record,
and keeping the bytes buys nothing. `collect()` deletes them and leaves the
number.

The rule this module enforces, stated so it can be tested:

    an arm retains its weights IFF it is the current survivor

Everything else is discarded or errored, and both have their bytes removed.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

CKPT = ROOT / "ckpt"
RECORDS = ROOT / "records" / "arms.jsonl"

SURVIVED = "survived"
DISCARDED = "discarded"
ERRORED = "errored"
# A keeper that has cleared the bar on the seed it was selected on and is waiting
# for its fresh-seed confirmation. It is NOT discarded, and the distinction is the
# whole content of the confirmation discipline: the arm is the current candidate
# parent, and its weights are the deliverable-in-waiting.
#
# This state did not exist until 2026-10-01, and its absence had a consequence
# that is easy to miss. `kept_pending_confirm` fell through to DISCARDED, so
# `collect()` deleted its weights — and because `champion.json` is only written
# when an arm is fully `kept`, and no arm was ever fully kept, `champ` was None
# and the guard "never touch the champion" protected nothing. Four arms of compute
# later, `ckpt/` was empty: the loop had produced numbers whose weights were gone,
# so there was nothing to re-score from disk, nothing to verify per-question, and
# nothing that could ever be published or measured on the held-out set.
#
# The rule this module enforces is "weights are lineage, not evidence", and it was
# being applied one step too early: a pending candidate's weights are lineage, and
# lineage is what you keep.
PENDING = "pending_confirmation"

# Verdicts that mean "no valid measurement was produced". Everything else that
# is not a keeper is a discard: it ran, it measured, the answer was no.
NO_MEASUREMENT = {"error"}


@dataclass
class Arm:
    arm: str
    survival: str
    verdict: str
    delta: float | None = None
    role: str = ""
    keep: bool = False
    discard: bool = False
    crash: bool = False
    # Cleared the bar, awaiting its fresh-seed confirmation: the candidate parent,
    # and the only arm whose weights are a deliverable-in-waiting.
    pending: bool = False
    has_ckpt: bool = False
    ckpt_bytes: int = 0
    reason: str = ""


def classify(rec: dict) -> str:
    """The three-valued survival of one record.

    `survived` is deliberately STRICTER than `verdict == "kept"`. An arm that
    cleared the bar on the seed it was selected on but did not reproduce on a
    fresh seed did not survive — that is the entire content of the confirmation
    discipline, and a lineage that counted it as a survivor would be recording
    seed luck as an inheritance.
    """
    v = rec.get("verdict") or ""
    if v in NO_MEASUREMENT or rec.get("error"):
        return ERRORED
    if v == "kept_pending_confirm":
        return PENDING
    if v == "kept" and rec.get("artifact", "").startswith("verified"):
        return SURVIVED
    return DISCARDED


def read_arms() -> list[dict]:
    if not RECORDS.is_file():
        return []
    out = []
    for line in RECORDS.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    # Fold supersessions: the last row per arm wins. A withdrawn verdict has to be
    # withdrawn on the DISK, not only in the log that recorded it, or the lineage
    # keeps showing a keeper the record has already retired -- and an arm that
    # appears twice is an arm counted twice.
    import agenda
    return list(agenda.latest_by_arm(out).values())


def _ckpt_bytes(arm: str) -> tuple[bool, int]:
    d = CKPT / arm
    if not d.is_dir():
        return False, 0
    n = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    return True, n


def lineage() -> list[Arm]:
    """Every recorded arm, classified, with what it currently costs on disk."""
    recs = read_arms()
    champ = None
    try:
        champ = json.loads((ROOT / "state" / "champion.json").read_text()).get("arm")
    except (OSError, json.JSONDecodeError):
        champ = None
    out: list[Arm] = []
    for r in recs:
        name = str(r.get("arm", "?"))
        surv = classify(r)
        has, nbytes = _ckpt_bytes(name)
        c = r.get("pooled_top1_candidate")
        k = r.get("pooled_top1_control")
        out.append(Arm(
            arm=name, survival=surv, verdict=str(r.get("verdict", "")),
            delta=(c - k) if (c is not None and k is not None) else None,
            role=str(r.get("role", "")),
            keep=(surv == SURVIVED), discard=(surv == DISCARDED),
            crash=(surv == ERRORED),
            # A pending candidate is neither a keeper nor a discard. Folding it into
            # either is what made the record claim a lineage that the disk did not
            # have: `n_keep + n_discard + n_crash` must equal the number of arms.
            pending=(surv == PENDING),
            has_ckpt=has, ckpt_bytes=nbytes,
            reason=str(r.get("reason", ""))[:200],
        ))
    return out


def collect(apply: bool = False) -> dict:
    """Delete the weights of every offspring that did not survive.

    Three guards, each of which exists because the alternative is unrecoverable:

      * the current champion is never touched, whatever its verdict says — a
        champion is a survivor by definition and deleting it would destroy the
        only model that can be published;
      * an arm with a live `run_one.py` is never touched, because its directory
        is being written to right now;
      * a record is never deleted, only the bytes under `ckpt/`. The measurement
        is the evidence; the weights are a cache of it.

    `apply=False` is the default so a human can read what would go before it goes.
    """
    arms = lineage()
    champ = None
    try:
        champ = json.loads((ROOT / "state" / "champion.json").read_text()).get("arm")
    except (OSError, json.JSONDecodeError):
        pass

    in_flight = None
    try:
        import subprocess
        live = subprocess.run(["pgrep", "-f", "run_one.py"],
                              capture_output=True).returncode == 0
        if live:
            import re
            logs = sorted((ROOT / "logs").glob("arm_*.log"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
            for p in logs[:2]:
                m = re.search(r"=== ARM (\S+) \(", p.read_text(errors="ignore")[-6000:])
                if m:
                    in_flight = m.group(1)
                    break
    except OSError:
        pass

    freed = 0
    removed: list[str] = []
    kept: list[str] = []
    for a in arms:
        if not a.has_ckpt:
            continue
        if a.arm == champ:
            kept.append(a.arm)
            continue
        # A candidate awaiting its fresh-seed confirmation is the current parent in
        # all but name, and its weights are the only artifact a keeper would ever
        # have. Collecting it here is how `ckpt/` ended up empty: the arm had
        # cleared the bar, the confirmation had not yet run, and the bytes went.
        if a.survival == PENDING:
            kept.append(a.arm)
            continue
        if in_flight and a.arm == in_flight:
            kept.append(a.arm)
            continue
        freed += a.ckpt_bytes
        removed.append(f"{a.arm} ({a.ckpt_bytes/1e9:.2f} GB, {a.survival})")
        if apply:
            shutil.rmtree(CKPT / a.arm, ignore_errors=True)
    return {"freed_bytes": freed, "freed_gb": freed / 1e9,
            "removed": removed, "kept": kept, "in_flight": in_flight,
            "champion": champ, "applied": apply}


def series() -> dict:
    """The keeps/discards series, in the order things actually happened.

    Confirmation runs are folded into the offspring they are testing. A
    confirmation is not a competitor — it is a second measurement OF the same
    candidate, on a seed it was never selected on — so counting it as its own
    entry in the trajectory would draw a keep that never happened and a discard
    that is really a verdict on the parent.
    """
    arms = lineage()
    base = [a for a in arms if not a.arm.startswith("confirm__")]
    by_arm = {a.arm: a for a in arms}
    out = []
    for a in base:
        confs = [c for c in arms if c.arm.startswith("confirm__")
                 and _confirms(c.arm, a.arm)]
        survived = a.keep or any(c.keep for c in confs)
        out.append({
            "arm": a.arm, "role": a.role, "delta": a.delta,
            "keep": survived,
            # A pending candidate is neither. Reporting it as a discard drew a
            # trajectory in which the loop's one keeper had been thrown away, which
            # is the opposite of what had happened: it had cleared the bar and was
            # waiting for the seed it was not selected on.
            "pending": (not survived) and a.pending,
            "discard": (not survived) and not a.crash and not a.pending,
            "crash": a.crash,
            "confirmations": [{"arm": c.arm, "delta": c.delta, "keep": c.keep}
                              for c in confs],
        })
    return {"series": out,
            "n_keep": sum(1 for r in out if r["keep"]),
            "n_pending": sum(1 for r in out if r["pending"]),
            "n_discard": sum(1 for r in out if r["discard"]),
            "n_crash": sum(1 for r in out if r["crash"])}


def _confirms(confirm_arm: str, base: str) -> bool:
    return confirm_arm.split("__", 1)[-1].split("_r")[0] == base


def render() -> str:
    arms = lineage()
    s = series()
    L = ["  lineage — weights are lineage, not evidence", ""]
    L.append(f"  {'arm':30s} {'survival':10s} {'verdict':22s} {'delta':>9s}  ckpt")
    for a in arms:
        mark = f"{a.ckpt_bytes/1e9:.2f} GB" if a.has_ckpt else "—"
        d = f"{a.delta:+.4f}" if a.delta is not None else "—"
        L.append(f"  {a.arm:30s} {a.survival:10s} {a.verdict:22s} {d:>9s}  {mark}")
    L.append("")
    L.append(f"  keep {s['n_keep']}   pending-confirm {s['n_pending']}   "
             f"discard {s['n_discard']}   crash {s['n_crash']}")
    # The arithmetic is the check. keep + pending + discard + crash must be the
    # number of arms, because a state that is not one of the four is a state the
    # record can describe and the disk cannot.
    total = s["n_keep"] + s["n_pending"] + s["n_discard"] + s["n_crash"]
    L.append(f"  {total} accounted for of {len(arms)} recorded"
             + ("" if total == len(arms) else "   <-- MISMATCH"))
    c = collect(apply=False)
    if c["removed"]:
        L.append(f"  collectable now: {c['freed_gb']:.2f} GB — {', '.join(c['removed'])}")
    else:
        L.append("  collectable now: nothing")
    if c["kept"]:
        L.append(f"  kept by rule: {', '.join(c['kept'])}")
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually delete. Without it this only reports.")
    a = ap.parse_args()
    if a.apply:
        c = collect(apply=True)
        print(f"  freed {c['freed_gb']:.2f} GB across {len(c['removed'])} arm(s)")
        for r in c["removed"]:
            print(f"    removed {r}")
        if c["kept"]:
            print(f"    kept   {', '.join(c['kept'])}")
    else:
        print(render())
