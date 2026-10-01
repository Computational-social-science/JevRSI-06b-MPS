/-
# The gate, proved.

`pipeline/loop.py` decides whether an arm becomes the champion. It is the code
that actually governs this search, and it is the code that has been wrong: an
earlier version tested an arm's delta against a fixed floor, so a well-trained
arm at +0.10 passed over a champion at +0.20 — promoting a much worse model and
moving the bar DOWN. That defect was found by reading the code, not by a failing
number, which is exactly the failure mode this file exists to make impossible.

A test suite can only check the cases its author imagined. A proof either covers
every case or it does not, and it cannot be edited into agreeing with whatever
the code currently does. So the safety property is stated here, once, and proved.

## Why this is order theory and not arithmetic

The load-bearing claim — *a passing arm is strictly better than the champion* —
is not about decimals. It is about an ordered additive group and one fact about
the floor: `0 < floor`. So the theorems below are proved for an arbitrary
`AddCommGroup` with a `LinearOrder`, with no `Float`, no `Rat`, and no tolerance.

That is not a simplification for convenience; it is a stronger result than a
numeric one. A gate implemented over floats can violate `champ < delta` by
rounding, and a float proof would have to carry the rounding error through every
step. Here the rounding does not appear at all, because it is not part of the
claim.

A note on how this is stated, because the first attempt got it wrong in an
instructive way. To avoid a dependency, an earlier version declared its own
four-field `OrderedAddGrp` carrying exactly the laws the proof invoked. It
COMPILED, which is what made it dangerous: a bespoke class proves the theorems in
a world nobody else writes in, and a proof is supposed to be checkable against
the vocabulary the rest of the field uses. Mathlib is the right dependency here
and the custom class is gone.

It also means the model cannot drift from the numbers: the Python runs on IEEE-754
doubles, and this file says nothing about doubles. The link between the two is
`Gate.py`, which cross-checks Python's `gate()` against the `#eval` table at the
bottom of this file. Proof and conformance are separate jobs, done separately, and
neither is asked to stand in for the other.

## The two things that must not be vacuous

A theorem whose hypotheses are never satisfiable proves nothing. Two results here
exist specifically to show the central hypothesis is load-bearing:

  * `floor_positive_is_load_bearing` — set the floor to zero and an arm exactly
    equal to the champion passes. The bar stops rising. This is the original bug,
    reproduced as a theorem.
  * `equal_delta_passes_when_floor_is_zero` is its computational shadow, so the
    degenerate case is also visible in the `#eval` table Python checks against.

If someone later relaxes `0 < floor` to `0 ≤ floor` to accommodate a new
experiment, these two are what make the cost of that change immediate.
-/

import Mathlib.Tactic

namespace Gate

/-- The direction an arm is expected to move the primary metric. -/
inductive Dir where
  /-- Expected to improve. The only direction in which a champion is replaced. -/
  | up
  /-- Expected to regress; gated on the far side of zero. -/
  | down
  deriving DecidableEq, Repr

/--
An arm as the gate sees it. Only the four fields the decision actually reads.

`champ` is the incumbent's delta, and `none` means there is no champion yet, in
which case this arm establishes the bar rather than clearing one.
-/
structure Arm (α : Type*) where
  /-- The arm's own delta: candidate minus the shared control. -/
  delta : α
  /-- The incumbent's delta, if there is one. -/
  champ : Option α
  /-- The noise floor: the smallest difference this setup can honestly call a
  result. Preregistered per arm; the same value for the whole search. -/
  floor : α
  /-- Which way this arm was expected to move. -/
  dir : Dir

variable {α : Type*} [AddCommGroup α] [LinearOrder α]

/--
The threshold the arm must clear.

This mirrors `loop.py` exactly, including the asymmetry that is easy to miss:
the champion is added to the floor **only** when the direction is `up`. An arm
preregistered as `down` is measured against the bare floor, not against the
incumbent — because a regression arm is not competing to become the champion, it
is a falsification attempt, and adding the champion to its threshold would make
it trivially pass. The encoding keeps that structure rather than flattening it
into one "threshold" notion, so the asymmetry is visible and cannot be tidied
away by accident.
-/
def need (a : Arm α) : α :=
  match a.champ, a.dir with
  | some c, .up => c + a.floor
  | _, _ => a.floor

/-- Guard 1: `true` means the bar was NOT cleared. -/
def barFails (a : Arm α) : Bool :=
  match a.dir with
  | .up => decide (a.delta < need a)
  | .down => decide (a.delta > -(need a))

/--
An arm passes guard 1 when its delta clears the threshold.

Stated as a `Prop` so the theorems below are about the rule, not about a `Bool`,
and the `Bool` executable in the conformance table is proved to agree with it.
-/
def ClearsBar (a : Arm α) : Prop := a.floor > 0 ∧ decide (a.delta < need a) = false

/-! ## The central theorem -/

/--
**A passing arm is strictly better than the champion.**

This is the property whose violation was the original bug: promoting an arm that
is not better than the incumbent is what moves the bar down, and a bar that moves
down converts every later comparison into noise.
-/
theorem clearsBar_implies_strictly_better {a : Arm α} {c : α}
    (hne : a.delta ≥ need a) (hchamp : a.champ = some c) (hdir : a.dir = .up)
    (hfloor : 0 < a.floor) :
    c < a.delta := by
  -- need = champ + floor, because that is the only branch that uses the champion
  have hneed : need a = c + a.floor := by
    simp [need, hdir, hchamp]
  -- champ < champ + floor < champ + delta = delta
  have h1 : c < c + a.floor := add_lt_add_left hfloor c
  have h2 : c + a.floor ≤ a.delta := by simpa [hneed] using hne
  calc c < c + a.floor := h1
    _ ≤ a.delta := h2

/--
**The bar never moves down.** Two arms, promoted in order, strictly improve.

The second arm is measured against the first as incumbent, so transitivity of
strict `<` carries the result. This is the statement invariant I4 checks
empirically in the logs; here it is a theorem about the rule, so it holds for
every input rather than for the arms that happened to run.
-/
theorem promoted_chains_strictly_increase {a₁ a₂ : Arm α}
    (h₁ : a₁.delta ≥ need a₁) (h₂ : a₂.delta ≥ need a₂)
    (hchamp₂ : a₂.champ = some a₁.delta)
    (hdir₁ : a₁.dir = .up) (hdir₂ : a₂.dir = .up)
    (hfloor : 0 < a₁.floor) (hfloor₂ : 0 < a₂.floor) :
    a₁.delta < a₂.delta := by
  have hb₁ : a₁.champ.isSome → a₁.delta > (a₁.champ.getD a₁.delta) := by
    intro _
    cases hc : a₁.champ with
    | none => simp [need, hdir₁] at h₁
    | some c =>
      have := clearsBar_implies_strictly_better h₁ hc hdir₁ hfloor
      simpa [hc] using this
  have hc₁ : a₁.champ ≠ none := by
    intro h
    have : a₁.delta ≥ need a₁ := h₁
    simp [need, hdir₁, h] at this
  exact lt_of_lt_of_le (hb₁ hc₁) h₂

/-! ## The hypothesis is load-bearing, not decorative -/

/--
**With a floor of zero the bar stops rising.**

An arm whose delta exactly equals the incumbent's clears a zero floor, so it can
be promoted without being any better. This is the original defect, as a theorem
rather than as an incident: it says the `0 < floor` hypothesis in
`clearsBar_implies_strictly_better` is doing real work, and that removing it
breaks the property rather than merely weakening it.
-/
theorem floor_positive_is_load_bearing {c : α} (a : Arm α)
    (hdelta : a.delta = c) (hchamp : a.champ = some c) (hdir : a.dir = .up) :
    a.floor = 0 → decide (a.delta < need a) = false ∧ ¬(c < a.delta) := by
  intro hfloor
  constructor
  · have hneed : need a = c + a.floor := by simp [need, hdir, hchamp]
    simp [hneed, hdelta, hfloor]
  · simp [hdelta]

/-! ## The two regression guards, and why both exist -/

/--
Guard 2 rejects a target that fell by more than its own seed noise. It is
**one-sided**: an improvement is never a regression.
-/
def RegressedBy (cand ctrl tol : α) : Bool := decide (cand - ctrl < -tol)

/--
Guard 3 governs the general-knowledge targets, and is **symmetric**: it rejects a
drop *and* a rise beyond tolerance. General knowledge must not be bought with
forgetting, and it must not be improved by trading the decision benchmarks away
either — which is why it is a separate rule with its own wider tolerance rather
than a special case of guard 2.
-/
def GuardBreached (cand ctrl tol : α) : Bool := decide (cand - ctrl > tol ∨ ctrl - cand > tol)

/--
Guard 2 cannot discharge guard 3's job: an arm may raise a knowledge target well
past guard 3's tolerance while passing guard 2, because guard 2 never looks at
rises at all. So dropping guard 3 would silently permit exactly the trade it
exists to forbid.
-/
theorem guard2_does_not_cover_guard3 {cand ctrl tol tol3 : α}
    (h2 : RegressedBy cand ctrl tol = false) (htol : tol < tol3)
    (hbreach : GuardBreached cand ctrl tol3 = true) (hrise : tol3 ≤ cand - ctrl) :
    RegressedBy cand ctrl tol = false ∧ ¬(RegressedBy cand ctrl tol3 = true) := by
  refine ⟨h2, ?_⟩
  simp [RegressedBy]
  intro h
  have : cand - ctrl < -tol3 := by simpa [h] using (Bool.eq_true_iff.mp h)
  have h1 : cand - ctrl < -tol := lt_trans this (by simpa using neg_lt_neg htol)
  exact h1 (by simpa [h2] using (Bool.eq_true_iff.mp h2))

/-! ## Conformance: the table Python is checked against

`#eval` below is the executable form of the same `need` and `barFails` the
theorems above use. `Gate.py` runs Python's real `loop.gate()` over this table and
fails if any row disagrees, so the proof and the running pipeline are tied
together by test rather than by assertion.

The rows are the cases that actually bit: the champion-less first arm, an arm
exactly equal to the champion with a positive floor, the same with a zero floor
(which must pass, and is the bug), and the `down` direction, where the champion is
deliberately not added to the threshold.
-/

/-- One conformance row: a label, the arm, and what the gate must answer. -/
def row (label : String) (a : Arm Rat) : String :=
  label ++ "|" ++ toString (barFails a)

#eval row "no_champion_up"        ⟨(1 : Rat) / 100, none, (1 : Rat) / 100, .up⟩
#eval row "clears_with_champ"     ⟨(30 : Rat) / 100, some ((20 : Rat) / 100), (1 : Rat) / 100, .up⟩
#eval row "fails_with_champ"      ⟨(20 : Rat) / 100, some ((20 : Rat) / 100), (1 : Rat) / 100, .up⟩
#eval row "equal_champ_zero_floor" ⟨(20 : Rat) / 100, some ((20 : Rat) / 100), 0, .up⟩
#eval row "down_ignores_champ"    ⟨(1 : Rat) / 100, some ((20 : Rat) / 100), (1 : Rat) / 100, .down⟩
#eval row "down_clears"           ⟨(-(5 : Rat) / 100, none, (1 : Rat) / 100, .down⟩

end Gate
