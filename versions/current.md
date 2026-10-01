# RSI-Jev on Qwen3-0.6B — research record

*Generated 2026-10-01 13:49:58 CST from `records/prereg.jsonl`, `records/arms.jsonl` and*
*`state/champions.jsonl`. Not maintained by hand: it cannot drift from the*
*data, because it is a projection of the data.*

## 1. What this is

A replication of [RSI-Jev](https://github.com/Shanghua-Gao/RSI-Jev)'s *method* rather than its checkpoints: a
self-improving loop that proposes a hypothesis, **registers the prediction before spending compute**, runs
the experiment, and retires its own champion when the measurement says to. The backbone is
`Qwen/Qwen3-0.6B-Base` on local Apple MPS, against upstream's `Qwen3.5-2B` on an NVIDIA GB10.

Upstream code is vendored **verbatim** and unmodified except for one performance line in
`rsijev/evaluate.py`, proven bit-identical by `tests/test_eval_parity.py`. So a difference
between our numbers and theirs cannot be attributed to a different implementation — only to a
different backbone, a different kernel stack, or a real effect.

## 2. The question

> Are RSI-Jev's conclusions about *what changes a decision model* general laws, or artefacts of one
> particular backbone and machine?

Upstream reports 204 arms across four releases. Its own most useful output is a **negative**:
in its first 24-arm run, nothing cleared the bar, and the reason it concluded is quoted in §5.

### What this search can and cannot conclude

Stated before any result, because it governs how every result below may be read. Upstream reports effect *E* at 2B; we test *E* at 0.6B.

| | |
|---|---|
| **E confirmed here** | the effect survived a 3.3x scale reduction and a different kernel stack — **strong evidence it is a law**, not an artefact of one setup |
| **E null here** | **ambiguous**: consistent both with *E was never real* and with *E is real and smaller at 0.6B*. This design cannot distinguish them. |

> **This search can confirm positives. It cannot refute them.** Every null below is a null *at this scale*. A null here is not evidence against upstream's finding; it is evidence that the finding did not reproduce here — a much weaker sentence, and the only one this design is entitled to.

A second limit, on the power itself: the control is the frozen model's logprob readout, identical for every arm, so the nulls' spread measures **training** variance with the evaluation's variance cancelled. It is a *paired* sd and is therefore smaller than upstream's unpaired per-seed 0.011 — so the fallback in force until the nulls finish **overstates** our noise and sets the bar too high. That errs safe, and the record says so rather than quietly benefiting.

## 3. Environment

| | |
|---|---|
| backbone | `Qwen/Qwen3-0.6B-Base`, 300 steps (sweep) / 1500 (confirmation) |
| device | `mps`, torch 2.14.0, fp32 |
| kernel stamp | `torch-reference/torch-*/mps` — **not** comparable with a CUDA or fla run |
| eval target | typed-decisions test, 400 cases / 2,000 questions, soft gold |
| guard | MMLU-Pro, 200 of 1,000 rows (general-knowledge veto) |
| corpus | `synth.jsonl` 6977 cases sha `8db84b39ab61`, `synth_adjacent.jsonl` 437 cases sha `625c3cd87698` |

Numbers only compare within one kernel stack. This is upstream's rule, applied rather than
assumed: every record here carries the device in its kernel stamp, and no MPS number is
pooled with a CUDA one.

## 4. Power — what this setup could have seen

**1 null arms so far.** Until there are at least two, the noise floor is
not measured and the fallback is upstream's per-seed sd of 0.011, taken at the conservative
end. This is stated rather than hidden because it is a weaker claim.

| seeds | minimum detectable effect |
|---|---|
| 2 | 0.0193 |
| 3 | 0.0158 |
| 4 | 0.0137 |
| 5 | 0.0122 |
| 6 | 0.0112 |
| 7 | 0.0103 |
| 8 | 0.0097 |

**Bar in force: +0.0200.**

Of the effects upstream reports, this design could detect: upstream_data_train_split, upstream_lower_layers_slow.

It could **not** detect: upstream_replay_15pct, upstream_mlp_combine, upstream_wider_coverage, upstream_best_of_24_arms, upstream_data_axis_worst.

This is the single most important number in this document. A null verdict on an arm whose
expected effect is below the MDE means *no effect we could see*, not *no effect*, and the
trajectory labels those arms `underpowered` rather than calling them published negatives.

## 5. Why the bar is where it is

Upstream diagnosed its own first run's failure, and the diagnosis is the reason this
pipeline measures its noise floor *before* proposing any hypothesis:

> **The bar was above the resolution.** A difference of two 3-seed means has an sd of about
> 0.010, so a realistic single-axis effect of +0.005 is invisible against a +0.020 threshold. The
> run was not powered to find what it was looking for.

A search whose threshold sits above its own resolution cannot distinguish "this lever does
nothing" from "this lever does something smaller than I could see" — and reports the first.
Upstream spent 24 arms learning that. The first arm here is a null arm for exactly this reason.

### The agenda, ordered by what this design can resolve

Minimum detectable effect at the planned budget: **0.0200**. An arm's *role* is decided from its expected effect against that number, **before** it runs — never from what it produced. That is what stops a null from being read as a refutation when the design could not have seen one.

| role | meaning | seed budget | arms |
|---|---|---|---|
| **calibration** | replicas of the champion recipe; their SPREAD is the noise floor the bar is set at | 1 | `null_floor`, `null1`, `null2`, `null3` |
| **decisive** | expected effect above the detectable minimum — a null here is informative | 3 | `champion_base`, `train_lower_layers_soft` |
| **exploratory** | expected effect below it — can only ever confirm, never refute | 1 | 12 arms |

The order is derived from the power analysis, not hand-written, so a change in the measured floor re-orders the queue instead of quietly making it wrong. The seed budget follows the role: a `decisive` arm is the only one where more seeds change what is knowable, so it is the only one they are spent on.

## 6. Results

1 arms: **0 kept**, 0 published negatives, 0 needing repair, 0 errors.

| arm | role | Δ vs control | expected | contamination | verdict |
|---|---|---|---|---|---|
| `null_floor` | — | +0.0665 | +0.0000 | **NOT CHECKED** | kept_pending_confirm |

## 7. Reference points

Read these as ceilings, not targets. The comparison that counts is inside each arm, between its
candidate and the control measured in the same call on the same weights.

| | value | what it is |
|---|---|---|
| majority baseline | 0.52 | typed-decisions test |
| latent factor bound | 0.704 | typed-decisions test |
| teacher self agreement | 0.735 | typed-decisions test |
| quirk line | 0.75 | typed-decisions test |
| Jev | 0.727 | the reference System One model |
| RSI-Jev v3.0 2B | 0.791 | upstream, typed-decisions |
| RSI-Jev v1.0 2B | 0.662 | upstream, pooled top-1 over 2,000 decisions |

_Qwen3.5-2B tower on an NVIDIA GB10. Context only. A 0.6B tower is ~3.3x smaller, so these are context, not a target: upstream's own 0.8B and 2B differ by 0.048._

## 8. Discipline

- **Held-out set**: `tasksource/tasksource` pinned at `7bfbd76c`, frozen 2026-10-01 12:03:53, **not yet read**. Upstream scores it "once, after a release model has been chosen, never during the search", and it is consumable — so `pipeline/held_out.py` refuses a second read and the champion does not take the crown until a fresh seed has confirmed it.
- **Fresh-seed confirmation**: an arm that clears the bar is recorded as `kept_pending_confirm` and does **not** become champion until the same recipe clears it on a seed it was not selected on. Upstream: the confirmation seed "is the only number the arm was not selected on".
- **Contamination is verified per arm, not asserted at build time**: `pipeline/contamination.py` re-checks the *loaded* corpus against the *loaded* targets before any compute — exact state hashes and 12-word phrase overlap, upstream's own rule. An arm whose corpus is not clean is refused rather than scored. A shared question *key* is deliberately NOT a criterion: the synthetic corpus is the same task family as the benchmark by design, and a key naming the same kind of question about a different document is the task being the task.
- **Artifacts are verified before publication**: a champion's checkpoint is reloaded from disk and must reproduce its training run's per-question predictions (floor 1.0000). Upstream: a checkpoint is "published only if it reproduces its training run's per-question predictions exactly", and a near-agreement is reported rather than rounded into a pass. An arm that clears the bar but whose artifact fails is recorded as `kept_unverified_artifact` — a save problem, not evidence the effect is absent.

## 9. Cost

- compute: local MPS, **1.1 h per arm** at 300 steps (205.3s x2 eval + 11.79s/step)
- **money: 0.00** — no rented GPU, no inference endpoint, no paid API
- network: HuggingFace Hub, anonymous, for the open-weights backbone only
- enforced by `tests/test_no_cost.py`, which fails the build on any metered-provider
  reference or credential literal in our code

## 10. Discarded measurements

**5 measurement(s) were taken, found invalid, and thrown away.** They are written up in [`DISCARDED.md`](DISCARDED.md), because upstream's rule is that the record is the evidence — "failures ship, including the ones that killed our own champion" — and a failure that leaves no trace is indistinguishable from one that never happened.

The most instructive: three null arms that turned out to be three copies of a single run, because a config value had overridden the per-arm seed. They returned a **measured noise sd of exactly 0.0000** and set the bar at +0.0110. A noise-free measurement system would have been a remarkable finding; it was a bug, and the giveaway was that the sd was *exactly* zero.

## 11. Reproduction

```bash
.venv/bin/python tests/test_gates.py        # the bar, the proposer
.venv/bin/python tests/test_eval_parity.py  # the one patched line is inert
.venv/bin/python tests/test_no_cost.py      # rule 2, enforced
.venv/bin/python pipeline/measure_cost.py   # what an arm costs here
.venv/bin/python pipeline/daemon.py         # one arm, resumable
.venv/bin/python pipeline/trajectory.py     # this record + the chart
```

Artifacts: `corpus/manifest.json` (sha256 per training file), `records/arms.jsonl` (one row per
arm), `records/prereg.jsonl` (predictions, fsynced before each run), `state/champions.jsonl`.
