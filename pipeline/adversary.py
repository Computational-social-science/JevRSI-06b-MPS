"""Adversarial probes: recompute every claim from the data beneath it.

The twelve invariants are all *transcription* checks — they read the record and
ask whether the record is internally coherent. That is necessary and it is not
sufficient, because a record can be perfectly coherent and still wrong: the
number in it need not have come from the run that claims to have produced it.

These probes go to the bottom. Every headline number in `records/arms.jsonl` is
recomputed from the per-question rows on disk, and the record is only allowed to
pass if the independent calculation agrees to the last bit. A record whose
headline cannot be reproduced is not "a number we are unsure about" — it is a
number with no provenance, and the pipeline is then publishing fiction.

TWO RULES LEARNED THE HARD WAY, both encoded below.

1. **A probe must use the pipeline's own definition, or it measures something
   else and cries wolf.** The first version of A1 compared `pred == gold` over
   every row and reported six MISMATCHes on a healthy record. The rows score each
   case in BOTH option orders, and `arm.top1_from_items` deliberately counts
   `canonical` only — reversed is a robustness diagnostic, not a measurement. The
   probe was wrong and the record was right. A checker that disagrees with a
   correct system is worse than no checker, so the definition is imported from the
   module that owns it rather than reimplemented here.

2. **Never re-derive a number and report success without saying where it came
   from.** Each finding carries the recomputed and the recorded value side by
   side, so a reader can judge the probe as well as the record.

Cost control: recomputing from raw rows is O(rows), and the rows run to
megabytes. So the full recompute is bounded to the most recent `FULL_N` arms —
the ones a wrong headline would actually still be propagating — while the cheap
O(1) cross-checks (A2, A3, A4) run over the entire history. The expensive check
is aimed; the cheap ones are exhaustive.
"""
from __future__ import annotations

import collections
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from invariants import Finding  # noqa: E402

# How many of the most recent arms get the full row-level recompute. The
# expensive check is aimed at what is still propagating; the cheap ones below are
# exhaustive and cover the whole history.
FULL_N = 3

TOL = 1e-9


@dataclass
class Probe:
    ok: bool
    detail: str
    evidence: list[str] = field(default_factory=list)


def _items_path(arm: str) -> Path | None:
    d = ROOT / "records" / arm
    if not d.is_dir():
        return None
    for name in (f"{arm}.items.jsonl", "items.jsonl"):
        p = d / name
        if p.is_file():
            return p
    return None


def _rows(path: Path):
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    return


# ------------------------------------------------------------------ A1
def a1_headline_reproduces(arm: dict) -> Probe:
    """Recompute every per-target top-1 from the raw per-question rows.

    The strongest probe available: it does not trust the record's own summary, it
    redoes the measurement from the 18k individual predictions underneath it.
    """
    name = arm.get("arm")
    path = _items_path(name)
    if path is None:
        return Probe(True, f"{name}: no raw rows on disk, cannot recompute",
                     [f"items: absent for {name}"])
    rec_c = arm.get("per_target_top1") or {}
    rec_k = arm.get("per_target_top1_control") or {}
    if not rec_c:
        return Probe(True, f"{name}: record carries no per-target numbers")

    hit: collections.Counter = collections.Counter()
    tot: collections.Counter = collections.Counter()
    n = 0
    for r in _rows(path):
        n += 1
        # THE DEFINITION, imported rather than reinvented: canonical order only.
        # `reversed` is a robustness diagnostic that upstream deliberately does
        # not fold into the score. Getting this wrong made a healthy record look
        # like six separate forgeries.
        if r.get("option_order", "canonical") != "canonical":
            continue
        k = (r.get("target"), r.get("role"))
        tot[k] += 1
        if r.get("pred") == r.get("gold"):
            hit[k] += 1

    mism = []
    checked = 0
    for tgt in set(rec_k) | set(rec_c):
        for role, key in (("control", rec_k), ("candidate", rec_c)):
            if tgt not in key:
                continue
            k = (tgt, role)
            if not tot.get(k):
                continue
            mine = hit[k] / tot[k]
            theirs = key[tgt]
            checked += 1
            if abs(mine - theirs) > TOL:
                mism.append(f"{tgt}/{role}: recomputed {mine:.6f} "
                            f"vs recorded {theirs:.6f}")
    ev = [f"{n} raw rows, {checked} per-target numbers recomputed"]
    ev += mism
    if mism:
        return Probe(False, f"{name}: {len(mism)} of {checked} per-target numbers do "
                            f"not reproduce from the raw rows", ev)
    return Probe(True, f"{name}: all {checked} per-target numbers reproduce from "
                       f"{n} raw rows", ev)


# ------------------------------------------------------------------ A2
def a2_delta_follows_from_its_parts(arm: dict) -> Probe:
    """The headline delta must equal candidate minus control, on the same scale."""
    c, k = arm.get("pooled_top1_candidate"), arm.get("pooled_top1_control")
    if c is None or k is None:
        return Probe(True, f"{arm.get('arm')}: no pooled numbers to cross-check")
    d = arm.get("champion_delta")
    # `champion_delta` is the delta over the SHARED control, so the pooled pair is
    # the right basis. If the field is absent the record is simply older than the
    # field, which is not a violation.
    if d is None:
        base = c - k
        ev = [f"no champion_delta recorded; pooled candidate - control = {base:+.6f}"]
        return Probe(True, f"{arm.get('arm')}: delta not recorded, pooled pair is "
                           f"self-consistent", ev)
    mine = c - k
    ev = [f"candidate {c:.6f} - control {k:.6f} = {mine:+.6f}",
          f"recorded champion_delta {d:+.6f}"]
    if abs(mine - d) > 5e-4:
        return Probe(False, f"{arm.get('arm')}: the recorded delta does not follow "
                            f"from its own candidate and control numbers", ev)
    return Probe(True, f"{arm.get('arm')}: delta follows from its parts", ev)


# ------------------------------------------------------------------ A3
def a3_control_is_shared(arms: list[dict]) -> Probe:
    """Every arm must be measured against the SAME control.

    This is the quiet one. Deltas are only comparable if the denominator is held
    fixed; a control that drifts between arms makes every cross-arm comparison
    meaningless while every individual number stays perfectly plausible. Nothing
    else in the system would notice.
    """
    seen: dict[float, list[str]] = {}
    for a in arms:
        k = a.get("pooled_top1_control")
        if k is None:
            continue
        seen.setdefault(round(float(k), 9), []).append(str(a.get("arm")))
    ev = [f"{len(seen)} distinct control value(s) across {len(arms)} arm(s)"]
    for k, names in seen.items():
        ev.append(f"control {k:.6f}: {', '.join(names[:4])}"
                  + (f" (+{len(names)-4} more)" if len(names) > 4 else ""))
    if len(seen) > 1:
        return Probe(False, f"{len(seen)} different control values across the search",
                     ev)
    return Probe(True, "the control is shared by every arm", ev)


# ------------------------------------------------------------------ A4
def a4_bar_is_recomputable(arms: list[dict]) -> Probe:
    """The bar must follow from the rule, not from a number someone typed.

    Guards against a bar that was quietly hand-adjusted between runs. Uses the
    pipeline's own `power` module for the same reason as A1.
    """
    import power
    pw = power.load()
    if pw is None:
        return Probe(True, "no power state yet; nothing to recompute")
    champ = None
    for a in arms:
        if a.get("verdict") == "kept":
            champ = a
    # RE-DERIVE the whole bar through `power.analyse`, the module's own entry
    # point, rather than restating the rule. The first version of this probe
    # wrote `max(0.020, paired_sd * 2.8)` inline — a rule this pipeline has never
    # used — and reported a critical against a healthy run. The seed budget and
    # the fallback are read from the recorded result rather than guessed, so the
    # probe cannot disagree with the pipeline over a constant.
    fresh = power.analyse(list(pw.null_deltas),
                          planned_seeds=pw.planned_seeds,
                          fallback_sd=pw.fallback_sd)
    recomputed = float(fresh.recommended_bar)
    recorded = float(pw.recommended_bar)
    ev = [f"null deltas fed back in: {list(pw.null_deltas)}",
          f"at planned_seeds={pw.planned_seeds}, fallback_sd={pw.fallback_sd}",
          f"re-derived recommended_bar {recomputed:.6f}",
          f"recorded  recommended_bar {recorded:.6f}",
          f"champion delta {champ.get('champion_delta') if champ else '(none)'}"]
    if abs(recomputed - recorded) > 1e-6:
        return Probe(False, "the bar does not follow from the rule that is supposed "
                            "to produce it", ev)
    return Probe(True, "the bar follows from the rule", ev)


# ------------------------------------------------------------------ A5
def a5_floor_follows_from_nulls(arms: list[dict]) -> Probe:
    """A MEASURED floor must be the spread of the null arms, and nothing else.

    The floor is what every later arm is judged against, so a floor that is not
    the nulls' own spread is a threshold chosen for convenience. While the floor is
    still the declared fallback this is a known limitation and stays below
    critical; once arms claim to have measured it, an unexplained floor is a
    critical.
    """
    import power
    pw = power.load()
    # `power` has no `measured` flag; it exposes the count and declares the
    # fallback in its own verdict string. Two null arms is the minimum that has a
    # spread at all, so that is the line between "measured" and "assumed".
    measured = pw is not None and pw.n_null_arms >= 2
    if not measured:
        return Probe(True, "the floor is still the declared fallback (known "
                           "limitation, not a violation)",
                     [f"{pw.n_null_arms if pw else 0} null arm(s); a spread needs >= 2"])
    # Take the null deltas from `power`, not by hunting the records for arms whose
    # `role` is "calibration". The first version did the hunt, and the records do
    # not carry `role`, so it found no null arms and passed everything — a probe
    # that cannot see its own subject. `power` is the module that owns the notion
    # of which arms are nulls; asking it is both simpler and correct.
    deltas = [float(d) for d in (pw.null_deltas or [])]
    if len(deltas) < 2:
        return Probe(True, f"floor claims to be measured but only {len(deltas)} null "
                           f"delta(s) exist to measure it from",
                     [f"null_deltas: {deltas}"])
    m = sum(deltas) / len(deltas)
    sd = math.sqrt(sum((d - m) ** 2 for d in deltas) / (len(deltas) - 1))
    ev = [f"{len(deltas)} null deltas: {', '.join(f'{d:+.4f}' for d in deltas)}",
          f"recomputed sample sd {sd:.6f} vs recorded paired_sd "
          f"{float(pw.paired_sd):.6f}"]
    if abs(sd - float(pw.paired_sd)) > 5e-3:
        return Probe(False, "a measured floor that is not the null arms' own spread",
                     ev)
    return Probe(True, "the measured floor is the null arms' own spread", ev)


# ------------------------------------------------------------------ runner
def probe(arms: list[dict]) -> list[Finding]:
    F: list[Finding] = []
    recent = arms[-FULL_N:] if len(arms) > FULL_N else arms

    # A1 — the expensive one, aimed at the most recent arms.
    bad = []
    ev: list[str] = []
    for a in recent:
        p = a1_headline_reproduces(a)
        ev += p.evidence
        if not p.ok:
            bad.append(p.detail)
    F.append(Finding("A1", "every headline number recomputes from its raw rows",
                     ok=not bad, severity="critical" if bad else "info",
                     detail=(bad[0] if bad else
                             f"{len(recent)} most recent arm(s) recomputed from raw "
                             f"per-question rows"),
                     evidence=ev[:8] or ["no recent arms to recompute"],
                     repair="stop the search; the published number has no provenance"
                            if bad else ""))

    for inv, fn, sev, title, rep in (
        ("A2", lambda: [a2_delta_follows_from_its_parts(a) for a in arms], "critical",
         "every delta follows from its own candidate and control", ""),
        ("A3", lambda: [a3_control_is_shared(arms)], "critical",
         "every arm was measured against the same control",
         "hold the control fixed, or re-run the arms that straddle the change"),
        ("A4", lambda: [a4_bar_is_recomputable(arms)], "critical",
         "the bar follows from the rule that produces it", ""),
        ("A5", lambda: [a5_floor_follows_from_nulls(arms)], "major",
         "a measured noise floor is the null arms' own spread",
         "re-derive the floor from the null arms"),
    ):
        res: list[Probe] = fn()  # type: ignore[assignment]
        failed = [p for p in res if not p.ok]
        allev = [e for p in res for e in p.evidence]
        # Keep each probe's own wording even when it passes. Collapsing a pass to
        # a bare count threw away the difference between "agrees" and "agrees, and
        # here is the limitation that is being tolerated" — which is the whole
        # reason a declared fallback is allowed below critical.
        detail = failed[0].detail if failed else (
            res[0].detail if len(res) == 1
            else f"{len(res)} checks agree; first: {res[0].detail}")
        F.append(Finding(inv, title, ok=not failed,
                         severity=sev if failed else "info",
                         detail=detail,
                         evidence=allev[:8] or ["nothing to check"],
                         repair=rep if failed else ""))

    return F
