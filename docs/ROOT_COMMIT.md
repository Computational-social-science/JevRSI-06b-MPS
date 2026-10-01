# JevRSI-06b-MPS

> **The RSI-Jev self-improvement loop on a Qwen3-0.6B backbone, run on Apple MPS,
> with a critical agent that is allowed to stop it.**

## What this is

A 24/7 unattended reproduction of the RSI-Jev loop — agents propose hypotheses,
register predictions before spending GPU time, run the arm, and retire their own
champions when the evidence says to — against a **0.6B** backbone on an **M4 Pro**.

The deliverable is the **curve and the record**, not a score. The part that is
usually thrown away is the point: every arm is kept, including the ones that
failed, and the failures are what the trajectory is made of.

## Relationship to JevRSI

This is **not** a fork of [JevRSI](https://github.com/Computational-social-science/JevRSI)
and does not supersede it. That project runs the same objective on different
hardware with a different corpus and a different discipline stack. They are two
lines of work, kept apart on purpose:

| | JevRSI | this repository |
|---|---|---|
| Accelerator | RTX 4070 12 GB, CUDA | Apple M4 Pro, **MPS** |
| torch | 2.10 | 2.14 |
| Corpus | `n4ze3m/typed-decisions-synth` | upstream `build_synth_corpus` |
| Guard rail | `EDIT_GUARD`, sha256-pinned | 12 invariants + 5 adversarial probes |
| Formal methods | — | Lean 4, kernel-checked |
| Cost of one arm | ~9.6 h | ~1.1 h |

Their numbers are **not** reproducible on this hardware and are not claimed to be.
Every figure here carries the machine that produced it. The two sets may share a
table; they are never averaged or differenced.

## The part that is different: a critical agent, not a dashboard

Every unattended loop can report that it is alive. The failure that costs the
most is the one nobody notices: **a loop that is alive and publishing numbers
that violate its own protocol.** A dead process announces itself. A live one
that is wrong does not, and will not stop on its own.

So this pipeline has five independent lenses, an arbiter that cannot be talked
out of a critical, and a stop mechanism:

| lens | asks |
|---|---|
| **vitals** | is the machine running the loop? |
| **provenance** | is every number traceable to a prediction that existed first? |
| **methodology** | is this the science we said we would run, in the order we said? |
| **integrity** | were the guards actually applied? |
| **adversary** | do the published numbers survive being **recomputed from the raw per-question rows**? |

Three properties are non-negotiable, and each is tested rather than asserted:

- **A critical is never diluted.** Four lenses agreeing "fine" and one saying
  "the bar is untraceable" is *one* critical, not 25% critical. Severity is
  unioned, never averaged.
- **Wrong outranks broken.** A dead loop is loud and cheap to fix. A live loop
  publishing invalid results is neither, so it is reported first and reserves the
  loudest treatment.
- **A critical in the science class halts the loop.** Not the operational class:
  a dead dashboard is broken, not wrong, and halting a search over a monitor
  outage inverts every priority here. A halt blocks *new* proposals — the arm in
  flight is left to finish — does **not** clear itself, and requires a human to
  release it with a written reason that is appended to `records/halt_audit.jsonl`.

## Why the numbers can be believed

Because they are recomputed, not just recorded.

- **Every headline is re-derived from the rows underneath it.** `adversary.py`
  A1 takes each arm's per-target top-1 and recomputes it from ~18,500 per-question
  predictions. A record whose headline cannot be reproduced is not "a number we
  are unsure about" — it is a number with no provenance.
- **A checker that has never fired is not a checker.** Every invariant and every
  probe has a negative control that plants the violation and requires the catch.
  Four separate times a guard fired on its own detection literals; that history is
  in `pipeline/detectors.py` rather than papered over.
- **The gate is proved, not tested.** `formal/Gate.lean` states the load-bearing
  claim — *a passing arm is strictly better than the champion* — for an arbitrary
  ordered additive group, and proves the bar never moves down. `formal/kernel_test.py`
  asks the Lean kernel what each proof term rests on, and cross-checks the real
  `loop.gate()` against the model's `#eval` table.
- **One number at a time.** The noise floor is measured from null replicas before
  any hypothesis runs; a hypothesis is labelled `decisive` only if its expected
  effect clears the detectable minimum; a decisive arm is promoted only after
  fresh seeds it was never selected on reproduce it; the held-out set is read once.

## Cost

Nothing here rents a GPU, calls a paid API, or requires a credential. Enforced
by `tests/test_no_cost.py` on every run — a guard that bills is a guard nobody
trusts. Backbone weights come from the HuggingFace Hub, anonymously.

## Layout

```
pipeline/     the loop, the gate, the discipline, the doctor, the publisher
formal/       Lean 4: the gate, the kernel test, the conformance check
rsijev/       upstream code, vendored verbatim
records/      the claims: preregistrations, results, halts  (append-only)
versions/     current.md and DISCARDED.md — the record and the negatives
tests/        13 files; every guard has a negative control
state/        power, champion, doctor, trajectory, dashboard
```

## Licence and provenance

Code MIT. [RSI-Jev](https://github.com/Shanghua-Gao/RSI-Jev) is MIT and not
affiliated with TypeSafe AI. This project reuses its ideas, loop structure and
public data, and claims none of its work as ours. Base weights follow their own
licences.
