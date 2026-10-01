"""The self-evolution loop: propose, preregister, run, gate, keep or retire.

EDITABLE — ours, not upstream. This is the part RSI-Jev runs internally and does
not ship, reconstructed from what its public record says the loop does:

  1. PROPOSE a hypothesis on one axis (model / data / training).
  2. PREREGISTER the prediction BEFORE the experiment runs, with a null floor
     measured against the current champion. This is the discipline that makes the
     result a result: a number written down after seeing the outcome is not a
     prediction. Upstream's CONTRIBUTING.md says an outside report "becomes a
     registered prediction with a null floor measured against it, and ships as a
     version — pass or fail"; the same rule applies to the loop's own arms.
  3. RUN the arm through the same code path a release uses
     (`scripts/run_arm_lib.run_arm`), on the same seed, against a control
     measured in the same call on the same weights.
  4. GATE. The bar is the suite's own noise, upstream's rule: +0.006 on the
     primary metric, no benchmark down by more than its own seed noise, and the
     general-knowledge guard within 0.030. An arm that clears exactly one guard
     is NOT discarded — upstream v2.1 exists because someone refused to drop an
     arm that failed one guard; it gets a diagnosis and one repair attempt.
  5. KEEP or RETIRE. A kept arm becomes the new champion and the bar to clear.

Everything is appended to a JSONL log, one record per arm, so the trajectory is
readable afterwards rather than reconstructed from console output.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Axis = Literal["model", "data", "training"]

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
RECORDS = ROOT / "records"
CKPT = ROOT / "ckpt"
LOGS = ROOT / "logs"
for _d in (STATE, RECORDS, CKPT, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# Upstream BENCHMARKS.md says a result is "+0.006 on the suite mean, no benchmark
# down by more than its own seed noise, MMLU-Pro within 0.030".
#
# 0.006 IS NOT THE BAR HERE. It was copied from a different project's write-up
# and left as a module constant, so the gate used 0.006 while the power analysis
# beside it measured a noise floor of 0.0665 from a null arm. The two were never
# connected: every arm for weeks was judged against a bar an order of magnitude
# below the noise it was supposed to exclude.
#
# So the bar now comes from `power.load()` — re-read on every gate call, because
# a bar that is correct at import time and stale after the first arm is exactly
# the defect this replaces. This constant survives ONLY as the fallback for a
# machine with no power analysis yet, and it is named for what it is.
UPSTREAM_BAR = 0.006


def bar() -> float:
    """The bar, read fresh. Falls back to the upstream number, loudly."""
    global BAR
    out = UPSTREAM_BAR
    try:
        import power
        r = power.load()
        if r is not None and r.recommended_bar:
            out = max(r.recommended_bar, UPSTREAM_BAR)
    except Exception:
        pass
    BAR = out
    return out


# Kept as a name so existing readers do not break, and refreshed on every
# `bar()` call so it cannot drift. A float subclass that re-evaluated itself was
# the first attempt and it raised inside float.__new__ — a proxy is the wrong
# tool when one function is the whole answer.
# Resolved AT IMPORT, so every existing reader — `Prereg.delta_floor`'s default,
# agenda's floor, the daemon's log line — sees the measured bar without having to
# remember to call `bar()` first. Leaving this as the raw fallback and relying on
# a side effect meant `next_prereg` stamped 0.006 onto a fresh prediction while
# the gate measured 0.133: the prediction and the test that judged it disagreed,
# which is precisely the drift this module exists to end.
BAR = bar()
SEED_SD = 0.011          # per-seed sd on a single benchmark
GUARD_TOL = 0.030        # MMLU-Pro must stay within this of the champion
# An arm that clears the bar but fails exactly one guard earns a repair attempt,
# not a death. Upstream: v2.1 started as a refusal to drop such an arm.
MAX_REPAIRS = 1


# --------------------------------------------------------------------- records
@dataclass
class Prereg:
    """A prediction written down before the GPU is touched."""
    arm: str
    axis: Axis
    change: str                    # what differs from the champion, in one thing
    prediction: str                # the claim, stated so it can be wrong
    delta_floor: float = BAR       # the smallest gap we would call a keeper
    direction: Literal["up", "down"] = "up"
    null_floor_vs: str = ""        # the champion this floor was measured against
    # What the reference work measured for this exact change, and where that
    # number came from. Carried so a null can be reported as "no effect we
    # could see" rather than "no effect".
    expected_effect: float | None = None
    source: str = ""
    # calibration | decisive | exploratory, decided by the power analysis BEFORE
    # the run. And how many runs this arm is owed.
    role: str = ""
    seeds_required: int = 1
    # The seed this arm is promised to run on. It belongs in the preregistration
    # because it is part of the prediction, not a detail of execution: the null
    # series exists to sample the SEED axis, so a run on the wrong seed is not a
    # noisier measurement, it is a different measurement wearing the same name.
    # `null2` once ran at seed 17 while 37 was written down here, and came back
    # bit-identical to the seed-17 arm beside it. The promise has to be on disk to
    # be checkable.
    seed: int | None = None
    ts: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArmResult:
    arm: str
    axis: Axis
    change: str
    champion: str = ""
    # primary
    pooled_top1_candidate: float | None = None
    pooled_top1_control: float | None = None
    min_decision_score_candidate: float | None = None
    min_decision_score_control: float | None = None
    pooled_aurc_candidate: float | None = None
    pooled_aurc_control: float | None = None
    ece_candidate: float | None = None
    ece_control: float | None = None
    # provenance of the measurement itself
    n_questions: int = 0
    targets_used: list[str] = field(default_factory=list)
    # The contamination report for this arm, verified before it ran. An empty
    # string means it was never checked, which is NOT the same as clean.
    contamination: str = ""
    # The artifact verification: the checkpoint reloaded from disk reproduced the
    # training run's per-question predictions, or it did not.
    artifact: str = ""
    # The champion this arm was measured against, on the same scale (delta over
    # the shared control). None means there was no champion and this arm
    # establishes the bar. Set by the daemon from state/champion.json so the
    # gate is judging against the incumbent and not against nothing.
    champion_delta: float | None = None
    champion_arm: str = ""
    # What the power analysis said this arm was worth, decided before the run.
    role: str = ""
    expected_effect: float | None = None
    seeds_required: int = 1
    # How many runs of this arm have actually completed, including confirmations.
    seeds_done: int = 0
    # The effective bar this arm's verdict was decided at — the gate uses
    # max(preregistered floor, measured bar), and the bar MOVES every time a null
    # lands. Recorded per row because a verdict is unreadable without it: the same
    # +0.0665 is a keeper at +0.0450 and a reject at +0.1043, and a record that
    # does not say which bar applied looks self-consistent under either.
    bar_used: float | None = None
    # per-benchmark, for the "no benchmark down by more than its seed noise" rule
    per_target_top1: dict[str, float] = field(default_factory=dict)
    per_target_top1_control: dict[str, float] = field(default_factory=dict)
    # gate
    passed: bool = False
    verdict: str = ""              # kept | rejected | needs_repair | error
    reason: str = ""
    guards_failed: list[str] = field(default_factory=list)
    n_failed_guards: int = 0
    # provenance
    prereg: dict[str, Any] = field(default_factory=dict)
    train_seconds: float = 0.0
    final_loss: float | None = None
    error: str = ""
    checkpoint: str = ""
    ts: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------- the gates
def _is_guard(tname: str) -> bool:
    """Is this target the general-knowledge guard rather than a decision benchmark?

    Upstream gives MMLU-Pro a 0.030 tolerance and every other benchmark 0.011,
    because the guard is not trying to measure the same thing: it exists to
    catch general knowledge being traded away, and its job is to veto, not to
    compete on the suite mean.
    """
    t = tname.lower()
    return "mmlu" in t or "guard" in t


def gate(cand: ArmResult) -> tuple[bool, list[str], str]:
    """Apply the bar. Returns (passed, failed_guards, reason).

    A guard is a named condition. Upstream's bar is "+0.006 on the suite mean,
    no benchmark down by more than its own seed noise, MMLU-Pro within 0.030".

    WHAT THE BAR IS AGAINST -- and this was a real defect, caught by reading the
    code rather than by a failing number.

    The first version tested `delta = candidate - control` against a FIXED floor
    of 0.006, where the control is the frozen model's zero-shot logprob readout.
    That is the wrong reference. Every arm that trains at all clears it, because
    training IS the effect and the effect is large: the very first null landed at
    +0.0665. So an arm at +0.05 passed while the champion stood at +0.13, and a
    well-trained arm at +0.10 passed over a champion at +0.20 -- promoting a much
    worse model and moving the bar DOWN.

    The control is a constant across arms (same frozen model, same evaluation
    set, same call), so subtracting it is a monotone transform of the absolute
    score: ordering by `delta` is ordering by score, which is why the
    trajectory's champion line was right while this threshold was not. But the
    THRESHOLD has to move with the champion, because upstream's own releases do:
    0.662, then 0.796, then 0.7056, then 0.7291 -- each had to clear the one
    before it, and a fixed floor cannot express a rising bar.

    So: with no champion, the floor is the noise (the first arm establishes the
    bar). With one, the arm must clear `champion + noise`. A bar that cannot fall.
    """
    failed: list[str] = []

    # --- guard 1: the bar itself, measured against the champion when there is one
    c, k = cand.pooled_top1_candidate, cand.pooled_top1_control
    if c is None or k is None:
        return False, ["no_measurement"], "arm produced no primary metric"
    delta = c - k
    prereg = cand.prereg or {}
    # The MEASURED bar always wins over a preregistered one. A floor fixed at
    # preregistration cannot know the noise floor yet — by definition the null
    # arms have not run — so an arm preregistered under the old 0.006 would
    # otherwise keep being judged by a number the experiment has since
    # disproved. Preregistration pins the HYPOTHESIS; the bar is a measurement.
    floor = max(float(prereg.get("delta_floor", 0.0)), bar())
    direction = prereg.get("direction", "up")
    champ = getattr(cand, "champion_delta", None)
    if champ is not None and direction == "up":
        need = champ + floor
        if delta < need:
            failed.append("bar")
        reason_bar = (f"cleared champion {champ:+.4f} + {floor:.4f} = {need:+.4f}"
                      if not failed else
                      f"delta {delta:+.4f} is below champion {champ:+.4f} "
                      f"+ {floor:.4f} = {need:+.4f}")
    else:
        need = floor
        if direction == "up" and delta < need:
            failed.append("bar")
        if direction == "down" and delta > -need:
            failed.append("bar")
        reason_bar = (f"cleared the bar +{floor:.4f} (no champion yet: this arm "
                      f"establishes it)" if not failed else
                      f"delta {delta:+.4f} is below the bar +{need:.4f} "
                      f"(no champion yet)")

    # --- guard 2: no target down by more than its own seed noise
    # The general-knowledge guard is EXCLUDED here and governed by guard 3
    # instead. Upstream gives MMLU-Pro a 0.030 tolerance and every other
    # benchmark 0.011; applying both to the same target makes the wider rule
    # dead code and turns a tolerated 0.02 wobble into a failed arm.
    for tname, ctv in (cand.per_target_top1 or {}).items():
        if _is_guard(tname):
            continue
        ctl = (cand.per_target_top1_control or {}).get(tname)
        if ctl is None:
            continue
        if ctv - ctl < -SEED_SD:
            failed.append(f"regression:{tname}")

    # --- guard 3: general knowledge must not be bought with forgetting
    # Upstream's MMLU-Pro guard: within 0.030 of the control. Matched by
    # substring so a differently-named target in our own suite still hits it.
    for tname, ctv in (cand.per_target_top1 or {}).items():
        if not _is_guard(tname):
            continue
        ctl = (cand.per_target_top1_control or {}).get(tname)
        if ctl is None:
            continue
        if abs(ctv - ctl) > GUARD_TOL:
            failed.append(f"guard:{tname}")

    if not failed:
        reason = reason_bar
    else:
        reason = f"{reason_bar}; failed {len(failed)} guard(s): {failed}"
    return (not failed), failed, reason


# --------------------------------------------------------------- champion book
class ChampionBook:
    """Who is currently the model to beat, and what it scored.

    The champion is a directory of weights plus the numbers it was measured at.
    The first arm establishes it; after that, an arm that clears the bar replaces
    it and the bar moves up with it.
    """
    def __init__(self) -> None:
        self.path = STATE / "champion.json"
        self.log = STATE / "champions.jsonl"

    def current(self) -> dict | None:
        if not self.path.is_file():
            return None
        return json.loads(self.path.read_text())

    def set(self, record: dict) -> None:
        self.path.write_text(json.dumps(record, indent=2))
        with open(self.log, "a") as fh:
            fh.write(json.dumps(record) + "\n")


def append_record(rec: ArmResult) -> None:
    with open(RECORDS / "arms.jsonl", "a") as fh:
        fh.write(json.dumps(rec.to_json()) + "\n")


def append_prereg(p: Prereg) -> None:
    with open(RECORDS / "prereg.jsonl", "a") as fh:
        fh.write(json.dumps(p.to_json()) + "\n")


def read_records() -> list[dict]:
    p = RECORDS / "arms.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue        # a torn final line from a kill is not a record
    return out


def read_prereg() -> list[dict]:
    p = RECORDS / "prereg.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
