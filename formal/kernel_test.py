#!/usr/bin/env python3
"""The Lean kernel test, and the link from the proof to the running pipeline.

Two jobs that cannot substitute for each other, so they are done separately and
neither is asked to cover the other.

**1. THE KERNEL TEST.** `lean Gate.lean` compiling is not sufficient evidence.
A Lean file also compiles when it contains `sorry`, and `sorryAx` is an axiom the
kernel accepts without complaint. A development one `sorry` away from proving
nothing still type-checks, still runs its `#eval`s, and still emits the
conformance table below — so "it compiles" is precisely the claim that cannot be
trusted. `Axioms.lean` asks the kernel what each proof term actually rests on,
and this script fails on anything beyond Lean's own three foundational axioms.

**2. CONFORMANCE.** A proof about a model of the gate says nothing about the gate
that runs, because the running one is Python on IEEE-754 doubles. The theorems
are stated for an arbitrary ordered additive group, where rounding cannot occur —
which is what makes them strong, and also what makes them silent about the
implementation. So the model emits a table of cases via `#eval`, this script runs
the REAL `loop.gate()` over the same cases, and any disagreement is a failure.

The second check is the one that would catch a regression introduced by a future
edit to `loop.py`. The first is the one that would catch a proof that was quietly
hollowed out. Neither alone is worth much.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORMAL = ROOT / "formal"
LEAN = Path.home() / ".elan" / "bin" / "lean"

# Lean's own foundational axioms. Resting on these is not a shortcut around the
# kernel; it is what `lean` means. Anything else is a defect.
BASIC_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def run_lean(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(LEAN), "--root", str(FORMAL), str(path.name)],
        cwd=FORMAL, capture_output=True, text=True, timeout=900,
    )


# ------------------------------------------------------------------ 1. kernel
def kernel_test() -> None:
    print("THE KERNEL TEST — what does each proof term actually rest on?")
    if not LEAN.is_file():
        check("lean is installed", False, f"{LEAN} not found")
        return
    r = run_lean(FORMAL / "Axioms.lean")
    if r.returncode != 0:
        check("Axioms.lean compiles", False, r.stderr.strip()[:200])
        return
    check("Axioms.lean compiles", True)

    out = r.stdout
    # "does not depend on any axioms"  |  "depends on axioms: [a, b]"
    ax = re.findall(r"'([^']+)' (does not depend on any axioms|depends on axioms: \[([^\]]*)\])",
                    out)
    check("the kernel reported on every declaration", len(ax) >= 8,
          f"{len(ax)} declarations audited")

    axiom_free = [n for n, kind, _ in ax if kind == "does not depend on any axioms"]
    with_axioms = [(n, [a.strip() for a in deps.split(",") if a.strip()])
                   for n, _, deps in ax if kind.startswith("depends")]

    print(f"        {len(axiom_free)} axiom-free, {len(with_axioms)} on the three basics")
    for n in axiom_free:
        print(f"          axiom-free  {n}")

    bad = [(n, a) for n, a in with_axioms if not set(a) <= BASIC_AXIOMS]
    check("no theorem rests on anything but Lean's three foundational axioms",
          not bad, str(bad[:2]))
    for n, a in with_axioms:
        print(f"          on {a}  {n}")

    check("no sorry anywhere (sorryAx is an axiom the kernel accepts silently)",
          "sorryAx" not in out and "sorry" not in out)
    check("no declarations, warnings or errors emitted",
          "error" not in out.lower() and "declaration uses 'sorry'" not in out,
          out.strip()[:120])

    # And the theorems themselves must exist under the names the audit claims.
    for t in ("clearsBar_implies_strictly_better", "promoted_chains_strictly_increase",
              "floor_positive_is_load_bearing", "guard2_does_not_cover_guard3"):
        check(f"{t} was audited", any(n.endswith(t) for n, _ in ax))


# ------------------------------------------------------- 2. proof ↔ pipeline
def conformance() -> None:
    print("\nCONFORMANCE — does the Python gate agree with the proved model?")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import loop  # noqa: E402

    r = run_lean(FORMAL / "Gate.lean")
    if r.returncode != 0:
        check("Gate.lean compiles", False, r.stderr.strip()[:300])
        return
    check("Gate.lean compiles", True)

    rows = dict(re.findall(r'"([a-z_]+)\|(true|false)"', r.stdout))
    check("the model emitted its conformance table", len(rows) >= 6,
          f"{len(rows)} rows: {sorted(rows)}")

    # Same cases, expressed against the REAL gate. `champ` is None where the
    # model says none, because `loop.gate` reads it off the ArmResult.
    cases = {
        "no_champion_up": (Fraction(1, 100), None, Fraction(1, 100), "up"),
        "clears_with_champ": (Fraction(30, 100), Fraction(20, 100), Fraction(1, 100), "up"),
        "fails_with_champ": (Fraction(20, 100), Fraction(20, 100), Fraction(1, 100), "up"),
        "equal_champ_zero_floor": (Fraction(20, 100), Fraction(20, 100), Fraction(0), "up"),
        "down_ignores_champ": (Fraction(1, 100), Fraction(20, 100), Fraction(1, 100), "down"),
        "down_clears": (Fraction(-5, 100), None, Fraction(1, 100), "down"),
    }
    for name, (delta, champ, floor, direction) in cases.items():
        if name not in rows:
            check(f"{name} present in the model", False)
            continue
        c = float(delta)
        ctl = 0.0                      # arbitrary: only the difference is read
        res = loop.ArmResult(
            arm=name, axis="training", change="formal-conformance",
            pooled_top1_candidate=c + ctl, pooled_top1_control=ctl,
            prereg={"delta_floor": float(floor), "direction": direction},
            champion_delta=(float(champ) if champ is not None else None),
        )
        passed, failed, _ = loop.gate(res)
        py = "false" if (not passed and "bar" in failed) else "true"
        agree = py == rows[name]
        check(f"{name}: model says barFails={rows[name]}, Python agrees",
              agree, f"model={rows[name]} python={py}")

    # The case that matters most, stated on its own so a regression is unmissable.
    check("the zero-floor case still passes the bar in BOTH (the original bug, "
          "reproduced rather than assumed away)",
          rows.get("equal_champ_zero_floor") == "false",
          "a bar that stops rising when the floor is zero")


def main() -> int:
    kernel_test()
    conformance()
    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
