# Run configuration — what these numbers are, and why

`config/run.json` is the machine-readable form read by `pipeline/daemon.py`
before every arm. This file is why each value has the value it has. Every number
here is measured on this machine, not assumed.

## Measured cost (`state/cost.json`)

MPS, fp32, torch 2.14.0, Qwen3-0.6B-Base:

| quantity | value |
|---|---|
| one eval pass, 400 cases / 2,000 questions | 205 s (9.7 q/s) |
| one optimiser step, batch 16, tower training | 11.2 s |
| **eval tax per arm** | **~1,830 s (30 min)**, see below |

An arm does not evaluate once. `run_arm` scores three targets — the primary
(2,000 questions), an `in_distribution` holdout it adds itself (~2,100 questions,
which exists to separate *"the pipeline is broken"* from *"the head fits its
training sources but does not transfer"*), and the MMLU-Pro guard (200 rows) —
and it does all three under **both** option orders (the champion spec sets
`eval_option_orders: [canonical, reversed]`) for **both** roles (the control and
the candidate). That is `3 × 2 × 2` passes, ~17,000 question-evaluations, before
a single optimiser step.

This was measured wrong the first time. The first version of this file counted
two passes and reported 411 s of tax, under-counting by ~3× — enough to promise
an overnight finish that does not happen. The corrected number is why the sweep
budget is 300 steps and not more.

| steps | per arm |
|---|---|
| 50 | 0.58 h |
| 300 | **1.26 h** |
| 1500 | 4.97 h |

## Why `steps: 300`

The step count is **fixed across the sweep**, which is upstream's own rule:

> The step count is fixed rather than the epoch count so that a data-axis arm
> that changes corpus size does not silently also change the amount of
> optimisation.

So every arm in a comparison differs in exactly one thing, and the paired
difference has a single cause.

300 is the **sweep** budget. It is chosen so the trajectory fills with real
verdicts overnight rather than producing one champion in three days — and 1.26 h
per arm is also what makes multi-seed confirmation affordable, which matters
more here than a long single run (see *Power* below).

A 20-step smoke run was used to verify the chain end to end and is **not**
science: 20 steps is 320 training examples against upstream's 24,000, and its
candidate scores *below* the majority baseline, which says only that the readout
was untrained. A null arm has to run at the **same budget as the arms it
calibrates** — a 20-step null floor says nothing about a 300-step arm's noise,
and using one would have made the bar wrong in the most confident way possible.

It is **not** the release recipe. Upstream's v1.0 is 1500 steps, and an arm that
clears the bar at 300 is re-run at `confirmation_steps` before anything is called
a release. Every record carries the step count it ran at, so a 300-step number
can never be read as a 1500-step one.

## Why `guard_n: 200`

MMLU-Pro is the general-knowledge **guard**, not a headline. Its job is to catch
decision gains being bought by forgetting. 200 rows resolves a 0.03 move against
a ~0.011 seed sd, which is the tolerance the guard is held to. Upstream scores
the full 1,000; the record names the subsample so the two are not confused, and
the guard's rule in `loop.gate` is a separate, wider tolerance (0.030) than the
one applied to decision benchmarks (0.011) — for exactly the reason that it is
not measuring the same thing.

## Power: why one seed is not enough, and what that forces

`pipeline/power.py` computes the minimum detectable effect from the measured
noise floor. At the fallback sd of 0.011 (upstream's measured per-seed sd):

| seeds | MDE |
|---|---|
| 1 | **∞** — a single seed has no variance estimate |
| 2 | 0.019 |
| 3 | 0.016 |
| 4 | 0.014 |

**One seed has no power at all.** A single-arm sweep can therefore only report
"this lever moved the metric by less than we could see", never "this lever does
nothing" — and the trajectory labels those arms *underpowered* rather than
*published negative*, using each arm's `expected_effect` against the MDE.

Only two agenda arms have an expected effect above the 2-seed MDE:

| arm | expected | source |
|---|---|---|
| `champion_base` | +0.130 | fine-tuning the tower was worth +0.13 |
| `train_lower_layers_soft` | +0.020 | bottom 8 layers at ⅒ lr, MMLU-Pro 0.339→0.359 |

Everything else in the agenda has an expected effect between 0.000 and +0.005 —
the same size as the noise. Those arms are run anyway, and published, because
upstream's own value is in the negatives:

> A variant you measured **removes a branch**. [...] A well-measured negative
> from outside is worth more to us than a small positive.

But they are reported as underpowered, and any of them that clears the bar is
the one that earns a multi-seed confirmation.

## Why `interval_s: 7200`

An arm is ~1.1 h at the sweep budget. A 2 h tick leaves headroom and **cannot
overlap**: launchd will not start a second instance of a label that is already
running, so a slow arm delays the next tick rather than doubling up on the GPU.

## Why fp32 on MPS, not bf16

`pipeline/dev.py` uses bf16 on CUDA (upstream's published stack) and fp32 on
MPS. bf16-on-MPS is supported but has a different reduction order from
bf16-on-CUDA, so enabling it would create a **third** numeric stack rather than
a second. Upstream's rule is that numbers only compare within one kernel stack
and every record carries a `linear_attn_kernel` stamp; we append the device to
that stamp (`torch-reference/torch-2.14.0/mps`) rather than pretending MPS and
CUDA are the same measurement. At 0.6B on a 64 GB machine, fp32 costs little.

## One local change to a protected file

`rsijev/evaluate.py` is upstream's PROTECTED evaluation path. One line in
`predict()` was changed: the per-row device-to-host copy became one copy per
batch.

On CUDA the original is cheap. On MPS every per-row `.tolist()` calls
`MPSStream::synchronize` and stalls the pipeline until the batch drains, so a
2,000-row pass paid 2,000 full syncs. It is the same computation producing
bit-identical floats — the values move at a different time, not differently —
and `tests/test_eval_parity.py` asserts that on real data, against the original
expression, before anything downstream is scored. If it ever disagreed, the
protected boundary would have been crossed and every number would be suspect,
so the test fails loudly rather than the pipeline continuing quietly.
