/-
# The kernel test.

`lean Gate.lean` compiling is not sufficient evidence, because a Lean file also
compiles when it contains `sorry`, and `sorryAx` is an axiom the kernel accepts
silently. A development that is one `sorry` away from proving nothing still type-
checks, still runs its `#eval`s, and still produces the conformance table — so
"it compiles" is exactly the claim that cannot be trusted.

This file is the check that cannot be faked. It re-declares every theorem from
`Gate.lean` and asks the kernel what each one actually rests on:

  * `#print axioms` walks the proof term and reports every axiom in it.
  * A theorem that says "does not depend on any axioms" rests on nothing.
  * A theorem that names axioms is acceptable ONLY for Lean's three standard
    ones — `propext`, `Classical.choice`, `Quot.sound` — which are part of what
    `lean` means, not shortcuts around it.
  * `sorryAx` appearing anywhere is a hard failure, and so is `sorry` as a term.

`kernel_test.py` runs this, parses the output, and fails on anything outside that
set. A green run means the Lean kernel re-checked every proof term in `Gate.lean`
and found each one either axiom-free or resting only on the three foundational
axioms of the logic itself.
-/
import Gate

open Gate

#print axioms Gate.clearsBar_implies_strictly_better
#print axioms Gate.promoted_chains_strictly_increase
#print axioms Gate.floor_positive_is_load_bearing
#print axioms Gate.guard2_does_not_cover_guard3

-- The definitions the theorems are about, so the audit covers the rules too.
#print axioms Gate.need
#print axioms Gate.barFails
#print axioms Gate.RegressedBy
#print axioms Gate.GuardBreached
