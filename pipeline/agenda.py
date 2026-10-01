"""Hypothesis generation: the loop's own research agenda.

EDITABLE — ours. Upstream does not ship the proposer (it runs the search
internally), so this reconstructs one from the directions its public record says
the loop actually explored, and keeps the discipline that makes an arm comparable:
**an arm changes exactly one thing.**

From upstream's README table, the directions that got from one release to the
next, and the ones that did not:

  model axis
    option_xattn readout (the champion's readout — kept)
    a two-layer MLP combining option and cross-attention features (+0.002, kept)
    per-mode readout layers (explored)
    a learned mixture over several hidden-state layers (explored)

  data axis
    train on the benchmark's own train split (+0.12, but costs general knowledge)
    + 15% general-knowledge replay (repays the cost — the first keeper)
    more sources, 3k per (the lever is not benchmark-specific)
    wider coverage, +84k questions (+0.011)
    recast replay to ten options (barely moves anything: +0.008)

  training axis
    a per-question confidence head that never changes which option wins (kept)
    freeze the tower's lower third (recovers some knowledge, breaks 3 benchmarks)
    the bottom 8 layers at ONE TENTH the learning rate (the real fix: +0.359)
    label smoothing / head weight decay against logit blowup
    listwise reranking RL (kept in v3.0: +60% R@1)

The proposer walks this agenda in order, skipping anything whose axis has already
been shown not to move, and its `propose()` is deterministic given the state, so
a 24/7 run is reproducible: the same history yields the same next arm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loop import BAR, Axis, Prereg, read_prereg, read_records


@dataclass
class Hypothesis:
    name: str
    axis: Axis
    change: str
    spec: dict[str, Any]
    prediction: str
    direction: str = "up"
    delta_floor: float = BAR
    # What this arm is a REPAIR of, if any.
    repair_of: str = ""
    # The effect size this arm is testing, from the reference work. It is what
    # the power analysis is asked about: if this is below the MDE, a null
    # verdict means "no effect we could see", NOT "no effect", and the record
    # has to say which one it is.
    expected_effect: float = 0.0
    source: str = ""
    # How much this arm is worth, decided BEFORE the run from the power analysis
    # rather than after from whatever it produced.
    #   calibration  measures the noise floor; establishes the bar
    #   decisive     expected effect ABOVE the detectable minimum -- a null here
    #                is informative, so these get the seed budget
    #   exploratory  expected effect BELOW the detectable minimum -- it can only
    #                ever confirm, never refute, and is labelled as such
    role: str = "exploratory"


# Roles, applied from the power analysis rather than from taste.
#
# The power analysis says the 2-seed minimum detectable effect is ~0.019 at the
# fallback sd, and only TWO of upstream's measured effects clear it: fine-tuning
# the tower (+0.13) and the slow-lower-layers change (+0.020). Everything else in
# this agenda sits between 0.000 and +0.005, which is the same size as the noise.
#
# So the agenda is ORDERED by what the design can actually resolve, and the seed
# budget follows: `decisive` arms get three seeds, `exploratory` get one. That is
# the difference between "we looked and found nothing" and "we could not have
# seen it either way" -- and only the first of those is worth anything.
#
# The agenda order is therefore: 4 calibration arms, then the 2 decisive ones,
# then the 13 exploratory. The exploratory arms are still run and still
# published -- upstream's most valuable output is a measured negative -- but they
# cannot be mistaken for a refutation, and the loop does not spend its seed
# budget on them.
def _role_for(arm_name: str, expected: float) -> str:
    """Decided by the design, before any run, and never by the outcome.

    The first version of this keyed `calibration` off `expected <= 0`, which
    swept every metric-neutral arm into it -- thirteen "calibration" arms, of
    which only four actually measure the floor. An arm aimed at STABILITY
    (logit_cap, label_smoothing) has an expected effect near zero because that
    is the point of it, not because it is a replicate of the control, and
    calling it calibration would promise a number it cannot produce.

    So calibration is identified by what the arm IS -- a replica of the
    champion recipe, varying only the seed -- and everything else by how its
    expected effect compares with the detectable minimum.
    """
    from loop import BAR
    if arm_name.startswith("null"):
        return "calibration"
    import power
    p = power.load()
    mde = p.recommended_bar if p and p.recommended_bar else BAR
    return "decisive" if abs(expected) >= mde else "exploratory"


# The champion recipe: upstream's v1.0 spec, verbatim from rsijev/README.md,
# with `sources` left to the corpus loader and `model` set by the caller.
# `freeze_base=False` because upstream's v1.0 is a full end-to-end fine-tune
# ("a Qwen3.5 Base tower fine-tuned end to end with a trained cross-attention
# scorer on top"), not a frozen-tower probe.
CHAMPION_SPEC: dict[str, Any] = {
    "readout": "option_xattn",
    "objective": "soft_ce",
    "steps": 1500,
    "residual": False,
    "option_order": "shuffled",
    "sources": "synth",
    "freeze_base": False,
    "readout_layer": -1,
    "lr_head": 1e-4,
    "eval_option_orders": ["canonical", "reversed"],
    "seed": 17,
    "xattn_heads": 4,
}


def _spec(**over: Any) -> dict[str, Any]:
    s = {**CHAMPION_SPEC, **over}
    return s


# The agenda, in the order the loop should walk it. Each entry changes ONE thing
# relative to CHAMPION_SPEC, so the paired difference has a single cause.
AGENDA: list[Hypothesis] = [
    # --- a NULL ARM FIRST, before any hypothesis.
    # Identical to the control, run through the identical path. Its spread IS
    # the noise floor, and every later bar is set against it. Upstream's rule:
    # "Null floors are measured, not assumed — arms provably identical to the
    # control, verified by object identity before any GPU time. Their spread
    # *is* the noise floor, so a difference smaller than it is not a result."
    # Running this first is what stops the whole search repeating upstream's
    # documented failure: a bar set above the resolution, which turns every
    # small true effect into a reported non-finding.
    Hypothesis(
        name="null_floor",
        axis="training",
        change="NOTHING: the champion spec, run again. A null arm, to measure the "
               "noise floor the bar is set against.",
        spec=_spec(),
        prediction=(
            "This arm is a REPLICA of the champion recipe, not a do-nothing: it "
            "trains, so its delta against its own control is the champion's effect "
            "(measured at +0.0665 on the first run), NOT zero. What the series buys is "
            "the SPREAD of that delta across seeds, and the spread is the noise floor — "
            "the smallest difference this setup can honestly call a result. A single "
            "replica cannot estimate a spread, which is why null1..null3 follow it."),
        expected_effect=0.0,
        source="CONTRIBUTING.md: null floors are measured, not assumed",
    ),
    # --- baseline: the champion itself. The first real arm establishes the bar.
    Hypothesis(
        name="champion_base",
        axis="training",
        change="champion recipe: option_xattn readout, soft_ce, shuffled options, "
               "tower tuned end to end",
        spec=_spec(),
        prediction="A trained cross-attention readout over a tuned 0.6B tower beats the "
                   "zero-shot logprob control on pooled top-1. Upstream measured this at "
                   "+0.13 on a 2B tower; at 0.6B the same mechanism should still be the "
                   "largest single effect available, because upstream's own conclusion is "
                   "that what works here is capability (fine-tuning the tower), not fit. "
                   "If it does not clear the bar, the readout is not learning the task at "
                   "this size and every later arm is moot.",
        expected_effect=0.130,
        source="README: fine-tuning the tower was worth +0.13",
    ),
    # --- MODEL AXIS
    Hypothesis(
        name="model_mlp_combine",
        axis="model",
        change="xattn_combine=mlp: a two-layer MLP over [v, v*ctx, v-ctx] instead of "
               "proj(v+ctx), which upstream notes cancels the context in the softmax",
        spec=_spec(arch_extra={"xattn_combine": "mlp", "xattn_mlp_hidden": 512}),
        expected_effect=0.002,
        source="README: a two-layer MLP combining the option and cross-attention features, +0.002",
        prediction="Upstream measured the same change at +0.002 on the 2B. At 0.6B the "
                   "readout is a larger share of the model, so the correction should matter "
                   "at least as much — but +0.002 is below the bar on its own, so this is "
                   "expected to land UNDER the bar and is run to establish that ceiling here.",
    ),
    Hypothesis(
        name="model_bilinear",
        axis="model",
        change="xattn_combine=bilinear: proj(v) + <Wv, ctx> with W zero-initialised, so the "
               "arm STARTS exactly at the champion and can only depart from it deliberately",
        spec=_spec(arch_extra={"xattn_combine": "bilinear"}),
        expected_effect=0.0,
        source="no upstream measurement; zero-init means it starts AT the champion",
        prediction="Because W starts at zero this arm begins at the champion's function. If "
                   "it does not beat the champion, the cross-attention context carries nothing "
                   "the option value does not already carry.",
    ),
    Hypothesis(
        name="model_layer_mix",
        axis="model",
        change="layer_mix over hidden-state indices 8/12/16/20/-1, initialised one-hot on the "
               "champion's own tap (-1) so the arm starts AT the champion, with lr_mix raised "
               "because a one-hot start behind a softmax cannot move at the head's lr",
        spec=_spec(arch_extra={"layer_mix": [8, 12, 16, 20, -1], "layer_mix_init": -1},
                   fit_extra={"lr_mix": 1e-1}),
        expected_effect=0.005,
        source="EXPLORE.md: the best any of 24 arms reached was +0.0035",
        prediction="A learned mixture can weight shallow and deep features without being told "
                   "which distribution a request came from. Upstream's cross-target "
                   "disagreement (MMLU-Pro prefers shallow, typed-decisions mid-stack) is not "
                   "resolvable by picking one tap, so a mixture should help if the disagreement "
                   "is real at 0.6B too.",
    ),
    Hypothesis(
        name="model_head_input_norm",
        axis="model",
        change="head_input_norm: parameter-free RMS normalisation of the readout's inputs, so "
               "logit growth from a few directions cannot dominate the head",
        spec=_spec(arch_extra={"head_input_norm": True}),
        expected_effect=0.0,
        source="no upstream measurement; expected metric-neutral, aimed at stability",
        prediction="The logit blowup upstream traced to a few rows or directions is a per-row "
                   "normalisation problem. If the instability is present at 0.6B, this should "
                   "stabilise training; it is expected to be roughly metric-neutral.",
    ),
    # --- DATA AXIS
    Hypothesis(
        name="data_no_shuffle",
        axis="data",
        change="option_order=canonical: train on the option order the benchmark presents, "
               "dropping the per-example shuffle",
        spec=_spec(option_order="canonical"),
        expected_effect=0.0,
        source="no upstream measurement; an augmentation/diagnostic change, expected neutral",
        prediction="Upstream's own note: a causal encoder gives option k a representation "
                   "that has seen options 1..k-1 and not k+1..K, so position carries "
                   "information the task does not intend, and a head trained on shuffled "
                   "options converges onto that off-distribution. Shuffling is the AUGMENTATION "
                   "and canonical is what the benchmark looks like at eval — so this should be "
                   "roughly neutral in-distribution and better out of it.",
    ),
    Hypothesis(
        name="data_include_adjacent",
        axis="data",
        change="add the synth_adjacent corpus: the nine keyword-adjacent domains upstream "
               "quarantined, so the result does not depend on near-workflow data",
        spec=_spec(sources="synth,synth_adjacent"),
        expected_effect=0.0015,
        source="EXPLORE.md: the data axis' best of 8 arms was -0.0015, i.e. nothing",
        prediction="Upstream's generalist rule: no synth domain is one of typed-decisions' four "
                   "workflows, and nine are keyword-ADJACENT. If near-workflow data were doing "
                   "the work, adding it back should be worth much more than a small positive. "
                   "A small gain means the signal is in the general decision structure, not in "
                   "the domain vocabulary.",
    ),
    # --- TRAINING AXIS
    Hypothesis(
        name="train_lower_layers_soft",
        axis="training",
        change="lower_layers_n=8, lower_layers_lr_scale=0.1: the bottom decoder layers train at "
               "a tenth the learning rate, in their own parameter group",
        spec=_spec(fit_extra={"lower_layers_n": 8, "lower_layers_lr_scale": 0.1}),
        expected_effect=0.02,
        source="README: the bottom 8 layers at one tenth the lr, MMLU-Pro 0.339 -> 0.359",
        prediction="Upstream's single most important optimiser change, and its explanation is "
                   "the one to test here: fitting decisions through the WHOLE tower at one "
                   "rate overwrites what the general knowledge sat in, and FROZEN layers cannot "
                   "adapt while SLOW ones can. At 0.6B (28 layers) the same trade should exist. "
                   "Expected: decision accuracy holds, general knowledge returns.",
    ),
    Hypothesis(
        name="train_residual",
        axis="training",
        change="residual=True: the logit is the base model's own option logprob plus a "
               "zero-initialised learned correction, so step 0 IS the control",
        spec=_spec(residual=True),
        expected_effect=0.0,
        source="no upstream metric; residual start did not protect the prior out of distribution",
        prediction="A residual start makes any drop below the control overfitting you can SEE "
                   "rather than a cold start being paid for. Upstream notes residual start did "
                   "not stop a head overwriting a prior out of distribution — so this is "
                   "expected to help in-distribution and not to protect the prior.",
    ),
    Hypothesis(
        name="train_label_smoothing",
        axis="training",
        change="label_smoothing=0.05: train on (1-e)*gold + e*uniform over the question's "
               "options, so the soft-CE optimum is a finite margin instead of an infinite one",
        spec=_spec(fit_extra={"label_smoothing": 0.05}),
        expected_effect=0.0,
        source="no upstream measurement; aimed at logit stability, expected neutral",
        prediction="Exactly-one-hot gold makes the soft-CE optimum an unbounded margin, which "
                   "is upstream's named cause of logit blowup (max logit 1304 under the stable "
                   "recipe). A finite-margin target should cap the logits. Expected: stabilise, "
                   "roughly metric-neutral.",
    ),
    Hypothesis(
        name="train_logit_cap",
        axis="training",
        change="logit_cap=20: soft-cap the scorer's logits as cap*tanh(logit/cap)",
        spec=_spec(arch_extra={"logit_cap": 20.0}),
        expected_effect=0.0,
        source="no upstream measurement; if the blowup is not the limit, this reads as the champion",
        prediction="The same instability addressed in the architecture rather than the "
                   "objective. Near-identity for |logit| << cap and bounded beyond it. Expected: "
                   "stabilise; if the blowup is NOT what limits this arm, the cap is inert and "
                   "the arm reads as exactly the champion — which is itself the finding.",
    ),
    Hypothesis(
        name="train_head_cosine",
        axis="training",
        change="head_schedule=cosine: decay the head's learning rate to zero instead of holding "
               "it constant after warmup",
        spec=_spec(fit_extra={"head_schedule": "cosine"}),
        expected_effect=0.0,
        source="no upstream measurement; aimed at late-training stability",
        prediction="Upstream: with a tuned tower the logit spikes arrived AFTER the tower's lr "
                   "had decayed, while only the head's constant lr was still large. Decaying "
                   "the head too should remove that window. Expected: stabilise late training.",
    ),
    Hypothesis(
        name="train_prior_kl",
        axis="training",
        change="prior_kl=0.1: KL(base || model) on the option distribution, anchoring the head "
               "to the frozen base's own prior",
        spec=_spec(prior_kl=0.1),
        expected_effect=0.0,
        source="no upstream measurement; puts the prior in the loss",
        prediction="Residual start did not protect the prior out of distribution; this puts the "
                   "prior IN the loss so departing costs something. Upstream's direction note "
                   "is load-bearing: KL(base || model) has the largest pull precisely where "
                   "the head has drifted furthest, which is the case the term exists for.",
    ),
    Hypothesis(
        name="train_calibration_head",
        axis="training",
        change="cal_method=oof_head_scorefloor: fit a per-question confidence head on withheld "
               "dev cases, dividing each question's logits by one positive number",
        spec=_spec(fit_extra={"cal_method": "oof_head_scorefloor"}),
        expected_effect=0.0,
        source="README: the confidence head moves ECE a great deal and pooled top-1 not at all",
        prediction="Argmax-preserving by construction: every row is divided by one positive "
                   "scalar, so the chosen option never changes and a decision is still ONE "
                   "forward pass. Upstream shipped this as free at inference. Expected: ECE "
                   "moves a great deal, pooled top-1 not at all. Reading it as a capability "
                   "gain would be reading a monotone transform as one.",
    ),
]


# Three more NULL arms, inserted right after the first, before the champion.
# A floor is a SPREAD and a spread needs more than one point: one null arm gives
# a delta but no variance, so the bar would still be a guess. Upstream's rule is
# explicit -- "Null floors are measured, not assumed -- arms provably identical
# to the control [...] Their spread *is* the noise floor, so a difference
# smaller than it is not a result."
#
# Each repeat uses a DIFFERENT SEED, so the spread they measure samples the seed
# axis of the variance and not only the process axis. They run at the same
# budget as every other arm, because a null measured at a different budget is
# not a null for this comparison.
AGENDA[1:1] = [
    Hypothesis(
        name=f"null{i}",
        axis="training",
        change=f"NOTHING, again: the champion spec, run point {i+1} of the null series.",
        spec=_spec(seed=17 + i * 10),
        prediction=(
            f"Replica {i+1} of 4 of the champion recipe, on seed "
            f"{17 + i * 10}. Its delta should land near the champion's own effect, "
            "not at zero — it trains. What must differ between the four is the "
            "seed, and what they jointly measure is the SPREAD of the delta, which "
            "is the noise floor the bar is then set at rather than above. If these "
            "four came back identical, that would not mean the pipeline is "
            "deterministic; it would mean the seeds were not applied."),
        expected_effect=0.0,
        source="CONTRIBUTING.md: null floors are measured, not assumed",
    )
    for i in (1, 2, 3)
]


# ---- assign roles, then ORDER THE AGENDA BY WHAT THE DESIGN CAN RESOLVE.
#
# Calibration first: the bar must exist before anything is judged against it.
# Then the decisive arms, in descending expected effect -- these are the ones a
# null would mean something about. Then the exploratory ones, where a null means
# only "not visible here" and a positive is a bonus.
#
# The order is derived, not hand-written, so a change in the power analysis
# re-orders the queue instead of quietly making it wrong.
def _assign_roles_and_order() -> None:
    for h in AGENDA:
        h.role = _role_for(h.name, h.expected_effect)
    def _cal_first(h):
        # The four null replicas run first, in series order, because they are
        # what establishes the bar everything else is judged against.
        # `null_floor` is the first of the series; `null1..3` are 1,2,3. The
        # name is parsed rather than assumed, because a first version that did
        # int(name[4:]) raised on `null_floor` at import -- which would have
        # taken the whole loop down on a naming detail.
        n = h.name[4:].lstrip("_")     # "null_floor"[4:] is "_floor", not "floor"
        return (0, int(n) if n.isdigit() else 0 if n == "floor" else 99)
    AGENDA.sort(key=lambda h: (
        {"calibration": 0, "decisive": 1, "exploratory": 2}[h.role],
        _cal_first(h) if h.role == "calibration" else 0,
        -abs(h.expected_effect),
        h.name,
    ))


_assign_roles_and_order()


def propose(history: list[dict] | None = None) -> Hypothesis:
    """The next arm: the first agenda entry with no result recorded against it.

    Deterministic given the state, so a 24/7 run is reproducible: an interrupted
    run resumes on the same arm, and two runs from the same state propose the
    same thing.

    The distinction that matters is preregistered-vs-finished, not
    tried-vs-not. An arm with a prediction on disk and no result was proposed and
    never resolved — the machine went down, the runner was killed, the process
    was OOM-killed. Its prediction is still unresolved, so it is re-run: that is
    the only honest way to close it out. Skipping it would leave a dangling
    prediction the log never resolves.
    """
    history = history if history is not None else read_records()
    finished = {r.get("arm") for r in history}
    for h in AGENDA:
        if h.name not in finished:
            return h
    return AGENDA[0]        # agenda exhausted; re-measure the champion


def next_prereg(h: Hypothesis, champion: dict | None) -> Prereg:
    """Write the prediction down, with a floor measured against the champion."""
    floor = BAR
    base = "none (first arm: no champion yet)"
    if champion:
        base = f"{champion.get('arm')} @ top1={champion.get('pooled_top1'):.4f}" \
            if champion.get("pooled_top1") is not None else str(champion.get("arm"))
        # The floor scales with the gap the loop is trying to resolve. Upstream:
        # "a gap of about +0.015 or more is confirmed by one fresh seed"; below
        # that, up to three. We keep the published +0.006 floor and say so.
    import confirm as _c
    return Prereg(
        arm=h.name, axis=h.axis, change=h.change, prediction=h.prediction,
        delta_floor=floor, direction=h.direction, null_floor_vs=base,
        expected_effect=h.expected_effect, source=h.source,
        role=h.role, seeds_required=_c.seeds_for(h.role),
    )
