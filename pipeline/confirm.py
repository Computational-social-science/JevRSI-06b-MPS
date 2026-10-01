"""Confirmation on a fresh seed. Upstream's rule, implemented.

EDITABLE — ours. This closes the gap where an arm that cleared the bar was
promoted to champion on the strength of the single run it was SELECTED on.

`BENCHMARKS.md`, "How a result becomes a result":

    An arm starts at one seed. A gap of about +0.015 or more [...] is confirmed
    by **one fresh seed the arm has never run on**. A smaller gap near the bar
    earns up to three seeds, plus that fresh confirmation seed. Spending four
    seeds on a difference you can already see is compute that buys nothing;
    **the confirmation seed is what is load-bearing, because it is the only
    number the arm was not selected on.**

That last clause is the whole point. An arm's own run is a number it was chosen
for: you proposed the change because you expected it to help, and it cleared the
bar you set. That number is optimistically biased by construction. The fresh
seed is the only estimate of the arm's effect that no decision touched.

So:

  * an arm that clears the bar does NOT become champion. It becomes
    `kept_pending_confirm` and a confirmation run is enqueued on a seed the arm
    has never used;
  * the confirmation is gated by the SAME bar, on the SAME control measured in
    the same call;
  * only a confirmed arm becomes champion;
  * a confirmation that FAILS does not discard the arm. It is recorded as
    `not_confirmed`, which is a different and more informative claim than
    "rejected": the effect was real enough to clear a preregistered bar once and
    not once more, which is what seed sensitivity looks like.

Seeds are assigned from a pool that an arm's own run can never draw from, so
"fresh" is structural rather than a promise.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"

# A confirmation seed is drawn from here, and an arm's OWN seed is allocated from
# the same pool at preregistration time and recorded, so the two can never
# collide. Offsets keep the pools visibly distinct in the log.
BASE_SEED = 17
CONFIRM_SEED_BASE = 9001


def arm_seed(index: int) -> int:
    return BASE_SEED + index * 10


def confirm_seed(index: int) -> int:
    return CONFIRM_SEED_BASE + index * 10


def is_confirmation(arm: str) -> bool:
    return arm.startswith("confirm__")


def base_arm(arm: str) -> str:
    return arm[len("confirm__"):] if is_confirmation(arm) else arm


def pending() -> dict | None:
    """The confirmation the loop owes, if any.

    Read from the arm log, not from an in-memory flag, so it survives the
    process that created it. A confirmation owed but not scheduled is a bar
    that was moved on incomplete evidence.
    """
    p = STATE / "pending_confirm.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def schedule(arm: str, run_index: int, *, delta: float, floor: float,
             steps: int, batch_size: int, seeds_required: int = 2,
             champion_delta: float | None = None) -> dict:
    """Enqueue the next fresh-seed run of an arm that cleared the bar.

    `run_index` is how many runs have FINISHED, so the seed for run n+1 is a
    function of the record rather than of how many times the loop was
    interrupted. A confirmation on a seed the arm already ran is not a
    confirmation, which is why the seed is derived and not reused.
    """
    n = run_index + 1
    rec = {
        "arm": arm,
        "confirm_arm": f"confirm__{arm}" + (f"_r{n}" if n > 1 else ""),
        "base_arm": arm,
        "seed": CONFIRM_SEED_BASE + (abs(hash(arm)) % 97) * 100 + n * 10,
        "run": n,
        "runs_required": seeds_required,
        "why": ("a number this arm was not selected on; the run that cleared "
                "the bar is optimistically biased by construction"),
        "selected_delta": delta,
        "champion_delta": champion_delta,
        "floor": floor,
        "steps": steps,
        "batch_size": batch_size,
        "ts": __import__("time").time(),
    }
    (STATE / "pending_confirm.json").write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def clear() -> None:
    p = STATE / "pending_confirm.json"
    if p.is_file():
        p.unlink()


def next_seed_for(arm: str, role: str) -> int:
    """The seed for run number `n+1` of an arm, where n runs already finished.

    Derived from the arm's own seed and the run index, so it is a function of
    the record rather than of how many times the loop has been interrupted. A
    seed is never reused for the same arm, which is the whole point: a
    "confirmation" on a seed the arm already ran is not a confirmation.
    """
    n = seeds_done(arm)
    return CONFIRM_SEED_BASE + (abs(hash(arm)) % 97) * 100 + n * 10


def seeds_done(arm: str) -> int:
    """How many runs of this arm have a recorded result."""
    p = ROOT / "records" / "arms.jsonl"
    if not p.is_file():
        return 0
    n = 0
    for line in p.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("arm") == arm or base_arm(r.get("arm", "")) == arm:
            if r.get("verdict") not in (None, "", "error"):
                n += 1
    return n


def seeds_for(role: str) -> int:
    """How many runs an arm of this role gets. Read from the run config.

    `decisive` gets three: it is the only tier where a null is informative, and
    3 seeds takes the minimum detectable effect from ~0.019 to ~0.016 — the
    difference between `train_lower_layers_soft` at +0.020 being resolvable and
    not. An `exploratory` arm's expected effect sits below the noise at any
    budget this machine can afford, so extra seeds there cost an arm each and
    buy nothing.
    """
    import json
    p = ROOT / "config" / "run.json"
    if p.is_file():
        try:
            return int(json.loads(p.read_text())
                       .get("seeds_per_role", {}).get(role, 1))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return 3 if role == "decisive" else 1


@dataclass
class ConfirmationOutcome:
    confirmed: bool
    verdict: str
    reason: str


def judge(*, original_arm: str, original_delta: float,
          confirm_delta: float | None, floor: float,
          champion_delta: float | None = None) -> ConfirmationOutcome:
    """Decide what a confirmation run means.

    The confirmation is judged on ITS OWN delta against the bar, not on whether
    it reproduces the original number to some tolerance. Upstream's bar is a
    difference, and a confirmation that is positive but under the bar has not
    cleared it. What we do then is keep both numbers, because the gap between
    them is the finding: two runs of the same recipe, one over the bar and one
    under, is a direct measurement of how much a single run at this budget can
    be trusted.
    """
    if confirm_delta is None:
        return ConfirmationOutcome(False, "not_confirmed",
                                   "the confirmation run produced no primary metric")
    # The bar the SEED is judged on is the same one the arm was: champion + noise
    # when there is a champion, noise alone for the first. A confirmation that
    # only clears the bare floor while the arm cleared champion+floor is not a
    # confirmation -- it is a seed that happened to beat nothing.
    need = floor if champion_delta is None else champion_delta + floor
    ref = ("the bar" if champion_delta is None
           else f"champion {champion_delta:+.4f} + {floor:.4f}")
    both = (original_delta >= need) and (confirm_delta < need)
    if confirm_delta >= need:
        return ConfirmationOutcome(True, "kept",
                                   f"confirmed on a fresh seed: {confirm_delta:+.4f} "
                                   f">= {ref} = {need:+.4f} (selected run "
                                   f"{original_delta:+.4f})")
    return ConfirmationOutcome(
        False, "not_confirmed",
        f"selected run {original_delta:+.4f} cleared {ref} = {need:+.4f} but the "
        f"fresh seed came in at {confirm_delta:+.4f}, under it"
        + (" — the effect did not survive a seed it was not selected on, which is "
           "seed sensitivity and not an effect" if both else ""))
