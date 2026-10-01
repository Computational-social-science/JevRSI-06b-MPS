# Discarded measurements

A measurement that was taken, found invalid, and thrown away. This file exists
because upstream's rule is that the record is the evidence:

> **Failures ship**, including the ones that killed our own champion.

and a failure that leaves no trace is indistinguishable from a failure that never
happened. Everything here was real compute that produced a number; the number is
wrong for a stated reason, and the reason is the part worth keeping.

---

## D1 — three null arms that were three copies of one run

**When:** 2026-09-30 night → 2026-10-01 12:00
**Status:** discarded; the null series was re-run with the fix
**Cost:** ~3 arms of compute, roughly 4 hours of MPS

### What was measured

The first three arms of the loop were the null series — replicas of the champion
recipe, whose *spread* is the noise floor the bar is then set at. All three
returned byte-identical numbers:

| arm | candidate | control | delta |
|---|---|---|---|
| `null_floor` | 0.5310 | 0.4645 | +0.0665 |
| `null1` | 0.5310 | 0.4645 | +0.0665 |
| `null2` | 0.5310 | 0.4645 | +0.0665 |

The power analysis then reported a **measured noise sd of exactly 0.0000** and
set the bar at **+0.0110**.

### Why it was wrong

Two separate defects, one of which hid the other.

**The proximate cause: the config seed overrode the arm seed.** `daemon.py` did

```python
spec["seed"] = CONFIG.get("seed", spec.get("seed", 17))
```

The agenda assigns each null a different seed (17, 27, 37, 47) precisely so the
series measures the *seed* axis of the variance. The config supplied `seed: 17`
and won unconditionally, so all four nulls ran at seed 17. Three copies of one
run is not a spread.

**The reason the bug was invisible: a trained null's delta is not zero.** The
preregistration for these arms claimed "its delta against its own control is 0 up
to run-to-run variation". That was wrong. These arms *train* — they are replicas
of the champion recipe — so their delta is the champion's effect, +0.0665, and
that is exactly what they should report. Only the *spread across seeds* is
meaningful. A prediction text that said "≈0" would have looked refuted by a
perfectly correct +0.0665, and the instinct to go looking for a bug would have
been wasted on the wrong thing.

Together these made a pipeline defect look like a remarkable scientific result:
a noise-free measurement system, implying a very high-powered design.

### What was done

* `spec["seed"] = spec.get("seed") or CONFIG.get("seed", 17)` — the config seed
  is a **default for an arm that names none**, not an override of one that does.
* The null arms' preregistration text rewritten to say what the series is
  actually for: the delta is the champion's effect, the SPREAD is the floor.
* Four guard tests added to `tests/test_gates.py`: every null names its own
  seed; the seeds are all different; three of the four differ from the
  champion's; and no null claims its delta is zero.
* The three records were deleted rather than kept with a flag, because
  `power.analyse` derives the floor from every null in the log — keeping them
  would have silently contaminated the re-run floor with the same artifact.
* The held-out set was **not** spent during any of this, which is why the fix
  could be verified at no cost to the project's one independent check.

### The transferable lesson

A measured sd of exactly zero is not a finding. It is the shape a bug takes when
the thing being measured is not varying — and the most likely reason a set of
"independent" replicates is identical is that they were not independent.

The power analysis was correct to be suspicious-worthy here only by accident: it
happened to be implemented. A pipeline that had simply assumed a noise floor would
have taken +0.0110 as measured and used it for the rest of the search, with a
number in every downstream record and no indication it was an artifact.

Corollary, and the reason the guard tests exist: **the check that would have
caught this is cheap and specific** — "are the seeds in these specs actually
different?" — and it should be written before the run that motivated it, not
after.

---

## D4 — the gate measured arms against the control, not the champion

**When:** 2026-10-01 12:30
**Status:** fixed before any arm completed; no result was produced under the
defective gate, so nothing had to be retracted
**Found by:** reading the code during a self-review, not by a failing number

### The defect

```python
delta = candidate - control          # control = the frozen model's zero-shot readout
pass if delta >= 0.006               # a FIXED floor
```

The control is the right thing to *subtract* and the wrong thing to *compare
against*. Every arm that trains at all clears a fixed floor of 0.006, because
training is the effect and the effect is large — the first null landed at
**+0.0665**. So with a champion already at +0.13:

| arm | delta vs control | defective gate | correct gate |
|---|---|---|---|
| worse but well-trained | +0.050 | **PASS** → promoted | FAIL |
| worse but well-trained | +0.100 | **PASS** → promoted | FAIL |

The loop would have replaced a much better model with a much worse one and
**moved the bar down**. Every arm that trains would have been a "keeper", which
is precisely the failure upstream's v2.0 record describes and criticises.

### Why reading the code found it and no test did

The gate had 34 passing tests. Every one of them constructed an `ArmResult` with
no champion — because the champion concept did not exist in the gate. The tests
were correct about the code and blind to the concept. **A test suite can only
check the concepts its author already had.**

### The fix

The bar moves with the incumbent and cannot fall:

```
no champion  ->  delta >= noise                     (the first arm sets the bar)
champion     ->  delta >= champion_delta + noise     (upstream's 0.662 -> 0.796 -> 0.7291)
```

`champion.json` stores an absolute top-1, so `daemon.py` converts it into the same
delta-over-control space before gating, and `ArmResult` gained
`champion_delta` / `champion_arm` so the record names what the arm was measured
against. `tests/test_champion_bar.py` (11 checks) covers the concept the old
suite could not: worse-than-champion rejected, a bar that cannot move down, the
first arm still judged on its own delta, and the other guards unchanged.

### Transferable lesson

A gate is a claim about what counts as an improvement, and it is easy to write
one that measures "did training work" when the question is "is this better than
what we already have". The two are identical for the first arm and diverge for
every one after it — which is exactly when it matters.

The tell: **a fixed threshold in a system that has a moving state.** Where there
is a champion, a constant floor is a bug even when every test passes.

---

## D5 — a contamination criterion that was wrong, not a finding

**When:** 2026-10-01 12:20
**Status:** criterion removed; the real corpus was clean under the correct rules

The first `contamination.py` reported **1500 "collisions"**. On inspection they
were ten shared question *names* — `score:urgency`, `choice:action`,
`noul:needs_review` — asked about **different documents** in the two corpora.

That is the design, not contamination. v1.0 trains on "a separate synthetic
corpus only, with no shared **state** and no shared **12-word phrase**": states
and phrases. The synthetic corpus is deliberately the same task family as the
benchmark; a key naming the same kind of question about a different document is
the task being the task.

The criterion was removed and the count kept at 0 in the record, so a reader can
see it was considered and why it does not apply. `tests/test_contamination.py`
then plants an exact duplicate and a 12-word-phrase overlap to prove the two
criteria that remain actually fire — because **a check that has never been seen
to fire is not known to work**, and a check that always returns zero is worse
than no check.

---

## D2 — the first cost measurement, under-counted by ~3×

**When:** 2026-10-01 00:50
**Status:** corrected before it was used for scheduling; the wrong number is
recorded here because it nearly caused a false promise

`pipeline/measure_cost.py` first reported an eval tax of 411 s per arm, from
"2 passes". The truth is **3 targets × 2 option orders × 2 roles**:

* primary target (2,000 questions),
* the `in_distribution` holdout `run_arm` adds itself (~2,100 questions, which
  exists to separate *"the pipeline is broken"* from *"the head fits its training
  sources but does not transfer"*),
* the MMLU-Pro guard (200 rows),

each scored under both `canonical` and `reversed` option order, for both the
control and the candidate.

Corrected: **~1,830 s (30 min)** of fixed tax per arm, not 411 s. A 300-step arm
is 1.26 h, not 0.44 h.

The wrong number would have promised a 15-arm agenda finishing overnight when it
needed about a day. The correction is written into the code as a comment
explaining the multiplier, because the multiplier is the part that is easy to
re-derive wrongly.

---

## D3 — a 20-step smoke run, kept out of the record on purpose

**When:** 2026-10-01 00:30
**Status:** ran, verified, deliberately not kept

A 20-step arm was used to prove the chain end to end before committing the
machine to an unattended run. It worked, and its result was then **deleted**:
20 steps is 320 training examples against upstream's 24,000, and its candidate
scored 0.338 — *below* the 0.520 majority baseline, which says only that the
readout was untrained.

It is recorded here rather than in `records/arms.jsonl` for a specific reason
beyond untidiness: a null arm must run at the **same budget as the arms it
calibrates**. A 20-step null floor says nothing about a 300-step arm's noise, and
using it would have set the bar wrong in the most confident way available.

The only thing it was for — proving that train → eval → gate → record → power →
trajectory runs without a human — it did establish, and the proof is the
`EXIT rc=0 wall=2563s` line and the four-guard rejection in the daemon log.
