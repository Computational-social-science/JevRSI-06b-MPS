"""Power analysis: measure the resolution limit, then set the bar there.

EDITABLE — ours, and the piece upstream says it needed and did not have.

Upstream's own account of why its first multi-agent run found nothing
(`EXPLORE.md`, jevtr_v1, "What the territory turned out to be like"):

    **The bar was above the resolution.** A difference of two 3-seed means has
    an sd of about 0.010, so a realistic single-axis effect of +0.005 is
    invisible against a +0.020 threshold. The run was not powered to find what
    it was looking for.

That is a self-diagnosis of a methodological failure, and it is the most useful
sentence in the file for anyone repeating the experiment. A search whose
threshold sits above its own resolution cannot distinguish "this lever does
nothing" from "this lever does something smaller than I could see" — and it will
report the first as a finding. Upstream spent 24 arms discovering that.

So the first thing this pipeline does, before any hypothesis is proposed, is
MEASURE the resolution:

  1. Null arms. Reruns of an arm that is provably identical to the control,
     verified by object identity before any GPU time. Their spread IS the noise
     floor — measured, not assumed. `CONTRIBUTING.md`: "Null floors are measured,
     not assumed ... Their spread *is* the noise floor, so a difference smaller
     than it is not a result."
  2. A minimum detectable effect from that spread, for the seed budget we can
     actually afford.

## WHAT THIS SEARCH CAN AND CANNOT CONCLUDE

Stated here because it governs how every null in this project may be read, and
because getting it wrong turns a weaker claim into a stronger one.

    Upstream reports effect E at 2B. We test E at 0.6B.

      E confirmed here  ->  the effect survived a 3.3x scale reduction and a
                            different kernel stack. That is STRONG evidence it
                            is a law rather than an artefact of one setup.

      E null here       ->  AMBIGUOUS, and consistent with two different worlds:
                            (i)  E was never real and upstream cleared a bar its
                                 own noise could not support, or
                            (ii) E is real and simply smaller at 0.6B.

So this search can CONFIRM positives. It cannot REFUTE them. Every null it
produces is a null *at this scale*, and must be reported that way. A null here
is not evidence against upstream's finding; it is evidence that the finding did
not reproduce here, which is a much weaker sentence — and the only kind of
sentence this design is entitled to.

## WHY THE NULL SPREAD IS A PAIRED MEASUREMENT, AND WHY THAT MATTERS

The control is the frozen model's zero-shot logprob readout on the evaluation
set, measured inside each arm. It is the SAME NUMBER for every arm — observed
0.4645 on all three discarded nulls — because the tower is frozen and the set is
fixed. So when the nulls' deltas are spread out, the evaluation's own variance
cancels completely and what remains is TRAINING variance only.

That makes the measured null spread a *paired* sd, and a paired sd is smaller
than a per-seed sd computed the unpaired way. Upstream's 0.011–0.016 is an
unpaired per-seed figure, so using it as the fallback OVERSTATES our noise.

That is the safe direction to be wrong in: an overstated floor rejects some real
effects, whereas an understated one accepts noise. But it also means the bar in
force until the nulls finish is higher than it needs to be, and the record says
so rather than quietly benefiting from it.

The design consequence: power is bought by SEEDS, not by more evaluation data.
The evaluation set is already the full 2,000-question test split, so there is
nothing left to widen — only seeds to add, and each costs a full arm.

  3. The bar set ABOVE what doing nothing buys — not at the resolution limit, and
     not above the largest null by a flat factor. The null series is a set of
     replicas of the champion recipe, so every one of them trains and every one
     of them reports the training effect; their mean is the delta a hypothesis arm
     gets for free. A bar that ignores that is cleared by an arm that changed
     nothing, which is not a threshold, it is a rubber stamp.

The honest alternative — a bar that an arm clearing by training alone can pass —
is the failure this whole project exists to avoid. It was also this project's
own failure, for one day, on 2026-10-01: the bar stood at +0.0450 while doing
nothing was worth +0.0680, and the first null arm was recorded as
`kept_pending_confirm`.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"


@dataclass
class PowerResult:
    n_null_arms: int
    null_deltas: list[float] = field(default_factory=list)
    # The measured noise floor: sd of the null deltas, and the observed spread.
    noise_sd: float = 0.0
    noise_max_abs: float = 0.0
    # What DOING NOTHING buys. The null series is a set of replicas of the
    # champion recipe, so every one of them trains and every one of them reports
    # the training effect. Their MEAN is therefore the delta that a hypothesis arm
    # receives for free, before its own change has done anything. A bar that
    # ignores this is a bar that any arm clears by training.
    do_nothing_mean: float = 0.0
    # The GENERAL-KNOWLEDGE guard's own noise, measured from the same null series.
    # It is a separate quantity from `noise_sd` because it is measured on a
    # different target with a different spread, and because it is the one that
    # decides admissibility: the guard tolerance is a fixed 0.030 borrowed from
    # upstream, and at 0.6B the recipe's expected cost sits 0.65 sd inside it, so
    # without this number the gate is a coin flip. See `loop.gate` guard 3.
    guard_mean: float = 0.0
    guard_sd: float = 0.0
    # Per-seed sd implied by the null arms, for a paired design.
    paired_sd: float = 0.0
    # What we can detect, per seed budget, at alpha=0.05 one-sided and 80% power.
    mde: dict[int, float] = field(default_factory=dict)
    # The bar this pipeline will use.
    recommended_bar: float = 0.0
    # The seed budget the bar was derived at. Recorded because the bar DEPENDS on
    # it, and a probe that has to guess this constant will eventually guess wrong
    # and report a healthy pipeline as tampered with. The state file has to be
    # self-describing for anything to be checkable against it.
    planned_seeds: int = 2
    fallback_sd: float = 0.011
    # Effect sizes from the reference work this is being compared against.
    reference_effects: dict[str, float] = field(default_factory=dict)
    detectable: dict[str, bool] = field(default_factory=dict)
    verdict: str = ""
    # This design confirms positives; it cannot refute them. A null is a null at
    # THIS scale, not evidence against upstream. See the module docstring.
    can_refute: bool = False
    note_scale: str = ""    # WHY the bar has the value it has, in words. A number without its reason is
    # an assumption, and this bar has been wrong once already.
    bar_rationale: str = ""
    # Which rule produced `recommended_bar`, by name. A bar whose rule cannot be
    # named cannot be re-derived, and a bar that cannot be re-derived is a number
    # someone typed — which is exactly what adversary/A4 exists to catch.
    bar_rule: str = ""

    def to_json(self) -> dict:
        """Valid JSON only.

        `mde[1]` is genuinely infinite — one seed cannot detect anything — and
        `json.dumps` writes that as the bare token `Infinity`, which no strict
        reader accepts. `state/power.json` carried it for days. The honest
        spelling of "no such value" is null, and the `allow_nan=False` in
        `save()` means a second one cannot appear silently.
        """
        d = asdict(self)
        d["mde"] = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                    for k, v in self.mde.items()}
        return d


# Effect sizes the reference work reports, so the power analysis is asked a
# concrete question: can this design detect what upstream actually found?
# Upstream's own numbers (README table, EXPLORE.md), not ours.
REFERENCE_EFFECTS = {
    # "train on the benchmark's own train split" -> typed-decisions +0.12
    "upstream_data_train_split": 0.120,
    # "the same + 15% general-knowledge replay" -> 0.796 vs 0.7794 = +0.017
    "upstream_replay_15pct": 0.017,
    # "bottom 8 layers at one tenth the learning rate" -> MMLU-Pro 0.339 -> 0.359
    "upstream_lower_layers_slow": 0.020,
    # "a two-layer MLP combining the option and cross-attention features" -> +0.002
    "upstream_mlp_combine": 0.002,
    # "wider coverage (data-scale-2)" -> suite +0.011
    "upstream_wider_coverage": 0.011,
    # "the best any arm in the first 24-direction run achieved" -> +0.0035
    "upstream_best_of_24_arms": 0.0035,
    # "label smoothing / head wd against logit blowup" -> not a metric effect
    "upstream_data_axis_worst": 0.0015,
}


def mde_for_seeds(paired_sd: float, n_seeds: int, *, alpha: float = 0.05,
                 power: float = 0.80) -> float:
    """Smallest true effect this design can detect, one-sided, at `power`.

    The standard one-sample-t result for a PAIRED design: the statistic is the
    mean of per-seed deltas and its variance is paired_sd^2/n, because common
    random numbers (upstream's `seed_everything`: "arms in one comparison share
    seed, init and data order, and differ in exactly one thing ... the paired
    difference then has far lower variance than either arm") remove the
    between-seed variance that dominates an unpaired comparison.

    z_{1-alpha} + z_{power}, from the normal approximation. At n=1 the statistic
    has no variance estimate, so the design is not powered at all and this
    returns inf — which is the honest answer and the reason one seed can only
    ever support a large effect.
    """
    if n_seeds < 1:
        return float("inf")
    if n_seeds == 1 or paired_sd <= 0:
        return 0.0 if paired_sd <= 0 else float("inf")
    z_a = 1.6449          # one-sided alpha = 0.05
    z_b = 0.8416          # power = 0.80
    return (z_a + z_b) * paired_sd / math.sqrt(n_seeds)


def increment_resolvable(paired_sd: float, n_seeds: int, *,
                         alpha: float = 0.05, power: float = 0.80) -> float:
    """How far above a known reference a hypothesis arm must land to be believable.

    The question the bar actually has to answer is NOT "what can this machine
    detect from zero?" It is "what would an arm that changed NOTHING look like,
    and how far above that must a real effect land to be distinguishable?"

    So the reference is the do-nothing mean (measured by the null series) and the
    margin is the resolvable increment over it. One seed has no variance estimate
    of its own, so it is judged against the reference with a plain one-sided 5%
    test — the reference distribution comes from the replicas, not from repeating
    the arm. From two seeds up, the mean of n paired deltas has sd
    paired_sd/sqrt(n), so the margin shrinks as 1/sqrt(n), which is where buying
    power with seeds comes from.

    `NULL_MARGIN` used to live here: a factor of 2 on the largest observed null.
    It was wrong for a reason worth keeping. These null arms are replicas of a
    recipe that TRAINS, so their deltas are not noise around zero — they are the
    training effect itself, around +0.068. Multiplying that by 2 does not "put the
    bar beyond the noise that has been seen"; it demands that a hypothesis beat
    plain training by more than training is worth, which set the bar at +0.188 —
    above every effect upstream reports, including its headline +0.120.

    That is upstream's own diagnosed failure reproduced in a new disguise: a
    threshold above the design's resolution, which cannot tell "this lever does
    nothing" from "this lever does something I cannot see", and reports the
    first. A margin belongs on the VARIANCE. It never belonged on the mean.
    """
    if paired_sd <= 0:
        return 0.0
    z_a = 1.6449          # one-sided alpha = 0.05
    z_b = 0.8416          # power = 0.80
    z = z_a if n_seeds <= 1 else z_a + z_b
    return z * paired_sd / math.sqrt(max(n_seeds, 1))


def analyse(null_deltas: Sequence[float], *, planned_seeds: int = 2,
            fallback_sd: float = 0.011,
            guard_deltas: Sequence[float] = ()) -> PowerResult:
    """Turn measured null arms into a bar.

    `null_deltas` are the control-minus-candidate differences from arms that are
    provably identical to the control. They should all be ~0; what matters is
    their spread.

    When there are not enough null arms to estimate a sd, we say so and fall
    back to the per-seed sd upstream measured on a single benchmark
    (0.011-0.016, EXPLORE.md), taking the CONSERVATIVE end. A fallback that
    flatters the design would defeat the purpose of computing this.
    """
    null_deltas = [d for d in null_deltas if d is not None and math.isfinite(d)]
    res = PowerResult(n_null_arms=len(null_deltas), null_deltas=list(null_deltas))
    res.reference_effects = dict(REFERENCE_EFFECTS)

    # The largest null MAGNITUDE is available whenever there is a null at all.
    # It is a magnitude, not a spread, so one observation is enough — and
    # skipping it for n < 2 is exactly how a bar ended up a third of a null that
    # had already run. `power.json` read n_null_arms 1, null_deltas [0.0665],
    # noise_max_abs 0.0 and a bar of 0.020.
    res.noise_max_abs = max((abs(d) for d in null_deltas), default=0.0)

    # The guard's own spread, from the same replicas. A spread needs >= 2 points,
    # and with fewer this stays 0.0 — which `loop.gate` reads as "no band", i.e.
    # the plain tolerance, i.e. upstream's rule unchanged. That is the right
    # degradation: an unmeasured band must not silently become a wide one.
    g = [d for d in guard_deltas if d is not None and math.isfinite(d)]
    if g:
        res.guard_mean = statistics.fmean(g)
    if len(g) >= 2:
        res.guard_sd = statistics.stdev(g)

    if len(null_deltas) >= 2:
        res.noise_sd = statistics.stdev(null_deltas)
        # A single null rerun measures the unpaired spread. Under CRN the
        # paired sd is smaller, and we cannot know by how much without a paired
        # null arm. So we take the unpaired sd as an UPPER bound on the paired
        # sd: that inflates the MDE, which makes the bar conservative, which is
        # the direction that cannot flatter a null verdict.
        res.paired_sd = res.noise_sd
    else:
        res.paired_sd = fallback_sd

    for n in range(1, 9):
        res.mde[n] = round(mde_for_seeds(res.paired_sd, n), 6)

    # THE BAR. Two rules, and which one applies is recorded in `bar_rule`.
    #
    # With no null arm there is nothing here that measures what doing nothing
    # buys, so the only defensible number is the resolution limit — and the
    # record must say out loud that this bar does NOT exclude the training
    # effect. It is provisional by construction and is replaced by the first
    # null.
    #
    # Once the null series exists the bar answers the question that actually
    # matters: an arm that changed nothing lands at `do_nothing_mean`, so the bar
    # is that mean plus the increment this design can resolve above a known
    # reference. Setting the bar AT the resolution limit instead — which is what
    # the rule did until 2026-10-01 — puts it at +0.0450 when doing nothing is
    # worth +0.0680, so the first null arm cleared it and was recorded as
    # `kept_pending_confirm`: the search had promoted plain training to champion
    # and called it an improvement.
    bar_mde = bar_from_mde(res.mde.get(planned_seeds, float("inf")), fallback_sd)
    if null_deltas:
        res.do_nothing_mean = statistics.fmean(null_deltas)
        inc = increment_resolvable(res.paired_sd, planned_seeds)
        res.recommended_bar = res.do_nothing_mean + inc
        res.bar_rule = "do_nothing_mean + resolvable increment"
        res.bar_rationale = (
            f"set ABOVE what doing nothing buys, not at the resolution limit. "
            f"The {len(null_deltas)} null arm(s) are replicas of the champion "
            f"recipe, so they train: their mean delta {res.do_nothing_mean:+.4f} "
            f"is the gain every hypothesis arm receives for free. A bar at the "
            f"detectable minimum ({bar_mde:.4f}) sits BELOW that and would be "
            f"cleared by an arm that changed nothing at all — which is exactly "
            f"what happened before this rule. The bar is therefore the "
            f"do-nothing mean plus the increment resolvable at {planned_seeds} "
            f"seed(s) ({inc:.4f}).")
    else:
        res.bar_rule = "MDE at the planned seed budget (no null arm has run)"
        res.bar_rationale = (
            f"PROVISIONAL. No null arm has run, so nothing measures what doing "
            f"nothing buys; this is the resolution limit ({bar_mde:.4f}) and it "
            f"does NOT exclude the training effect. It is replaced as soon as "
            f"the first null lands.")
    res.recommended_bar = round(res.recommended_bar, 6)
    res.planned_seeds = planned_seeds
    res.fallback_sd = fallback_sd

    res.detectable = {
        name: (res.recommended_bar <= effect)
        for name, effect in res.reference_effects.items()
    }

    vis = [n for n, ok in res.detectable.items() if ok]
    invis = [n for n, ok in res.detectable.items() if not ok]
    lines = [
        f"null arms: {res.n_null_arms}",
        f"measured noise sd: {res.noise_sd:.4f}"
        + ("" if res.n_null_arms >= 2 else f"  (FALLBACK {fallback_sd}, <2 null arms)"),
        f"paired sd used:     {res.paired_sd:.4f}",
        f"doing nothing buys: {res.do_nothing_mean:+.4f}"
        + ("" if null_deltas else "  (NOT MEASURED — no null arm has run)"),
        f"guard cost      : {res.guard_mean:+.4f}"
        + (f"  sd {res.guard_sd:.4f} (tolerance 0.0300, so the recipe's "
           f"admissibility is {'UNDECIDABLE at one seed' if res.guard_sd and abs(res.guard_mean) < 0.030 + res.guard_sd else 'decidable'})"
           if g else "  (NOT MEASURED)"),
        f"MDE by seed budget: " + ", ".join(f"{n}s={v:.4f}" for n, v in res.mde.items()
                                           if math.isfinite(v) and v > 0),
        f"bar at {planned_seeds} seeds: +{res.recommended_bar:.4f}  [{res.bar_rule}]",
        f"detectable: {vis or 'none'}",
        f"NOT detectable: {invis or 'none'}",
    ]
    res.verdict = "\n    ".join(lines)
    res.can_refute = False
    res.note_scale = ("every null here is a null AT 0.6B on MPS; it is consistent "
                      "with both 'upstream's effect was never real' and 'the effect "
                      "is real and smaller at this scale', and this design cannot "
                      "distinguish those")
    return res


def bar_from_mde(mde: float, fallback_sd: float) -> float:
    """The bar, as a pure function of the detectable effect and the fallback.

    Extracted from the measurement body so the rule has exactly ONE definition,
    callable by a probe. An earlier version of the adversarial check restated it
    inline as `max(0.020, paired_sd * 2.8)` — a rule this pipeline has never used.
    The real one is the MDE at the planned seed budget, rounded UP to two
    significant figures so a number like 0.0044 does not read as more precise
    than a 2,000-question estimate deserves. The probe therefore raised a
    critical violation against a perfectly healthy pipeline, having invented a
    rule instead of reading one.

    That is the entire argument for this function existing. A checker that
    re-derives a rule from prose will eventually re-derive it wrongly, and it
    will be loud about it; a checker that calls the rule cannot.
    """
    if math.isfinite(mde) and mde > 0:
        exp = math.floor(math.log10(mde))
        return math.ceil(mde * 10 ** (1 - exp)) / 10 ** (1 - exp)
    return fallback_sd


def save(res: PowerResult) -> Path:
    p = STATE / "power.json"
    # allow_nan=False makes invalid JSON an exception rather than a surprise in
    # whatever reads this file next week.
    p.write_text(json.dumps(res.to_json(), indent=2, allow_nan=False) + "\n")
    return p


def load() -> PowerResult | None:
    p = STATE / "power.json"
    if not p.is_file():
        return None
    raw = json.loads(p.read_text())
    # Round-trips a null mde back to inf, so an old file and a new one behave the
    # same and the comparison between them means something.
    raw["mde"] = {int(k): (float("inf") if v is None else v)
                  for k, v in raw.get("mde", {}).items()}
    return PowerResult(**raw)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=0,
                    help="seed budget to derive the bar at; 0 = the budget a "
                         "decisive arm actually runs, read from the run config")
    ap.add_argument("--from-arms", default="",
                    help="read null deltas from an arms.jsonl instead of stdin")
    a = ap.parse_args()

    if a.seeds:
        planned = a.seeds
    else:
        # The same source the daemon uses. A CLI that defaulted to its own constant
        # would be a second implementation of the bar's seed budget, and A4 exists
        # because second implementations of this rule are how the recorded bar and
        # the real one drift apart.
        import confirm as _c
        planned = _c.seeds_for("decisive")

    deltas: list[float] = []
    gdeltas: list[float] = []
    if a.from_arms:
        import agenda as _ag
        recs = []
        for line in Path(a.from_arms).read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
        # Only the null series, by the agenda's definition -- not every record.
        deltas = _ag.null_deltas(recs)
        # The guard's own series goes through the same path. It did not, once: the
        # CLI re-derived the bar without it and overwrote the state file, which
        # silently disarmed the guard band in `loop.gate` until the next null arm
        # landed. A second implementation of the rule is not a convenience, it is
        # a way for the recorded numbers and the applied ones to part company.
        gdeltas = _ag.null_guard_deltas(recs)
        print(f"  read {len(recs)} record(s), {len(deltas)} of them null-series, "
              f"{len(gdeltas)} with a guard delta")
    else:
        import sys
        deltas = [float(x) for x in sys.stdin.read().split() if x.strip()]

    res = analyse(deltas, planned_seeds=planned, guard_deltas=gdeltas)
    print("POWER ANALYSIS")
    print("  " + res.verdict)
    p = save(res)
    print(f"  -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
