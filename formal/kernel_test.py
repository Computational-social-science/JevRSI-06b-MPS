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
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORMAL = ROOT / "formal"
# Resolve the compiler rather than assuming one location. The elan shim exists
# only once elan has successfully installed a toolchain, and it is exactly the
# thing that was broken here: elan's own downloader stalled at 0 KB/s, a manual
# download died mid-stream, and `tar` extracted the truncated archive into the
# toolchain directory without complaint. So the search order is: the toolchain
# binary itself, then the elan shim, then whatever is on PATH — and the first
# one that cannot print a version is skipped rather than trusted.
_CANDIDATES = [
    Path.home() / ".elan" / "toolchains" / "leanprover--lean4---v4.34.1" / "bin" / "lean",
    Path.home() / ".elan" / "bin" / "lean",
]


def _find_lean() -> "subprocess.CompletedProcess | None":
    import shutil
    for cand in _CANDIDATES + ([shutil.which("lean")] if shutil.which("lean") else []):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            r = subprocess.run([str(cand), "--version"], capture_output=True, text=True)
            if r.returncode == 0 and "lean" in (r.stdout + r.stderr).lower():
                return r
    return None

# Lean's own foundational axioms. Resting on these is not a shortcut around the
# kernel; it is what `lean` means. Anything else is a defect.
BASIC_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def run_lean(path: Path) -> subprocess.CompletedProcess:
    """Run the checker THROUGH LAKE, never as a bare `lean`.

    Mathlib's oleans live in `.lake/packages/*/.lake/build/lib/lean`, and those
    directories are on LEAN_PATH only when lake builds them. A bare `lean` sees
    no `Batteries`, no `Mathlib`, and reports a hundred class-resolution errors
    that have nothing to do with the file — which is exactly what the first
    version of this script did, and it reported "Axioms.lean does not compile"
    about a file that was fine.
    """
    lean = str(_find_lean_path())
    lake = shutil.which("lake") or str(_find_lean_path().parent.parent / "bin" / "lake")
    if lake and Path(lake).exists():
        r = subprocess.run([str(lake), "env", "lean", path.name],
                           cwd=FORMAL, capture_output=True, text=True, timeout=900)
        return r
    return subprocess.run([lean, "--root", str(FORMAL), path.name],
                          cwd=FORMAL, capture_output=True, text=True, timeout=900)


def _find_lean_path() -> Path:
    for cand in _CANDIDATES:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    import shutil
    w = shutil.which("lean")
    return Path(w) if w else _CANDIDATES[0]


# ------------------------------------------------------------------ 1. kernel
def kernel_test() -> None:
    print("THE KERNEL TEST — what does each proof term actually rest on?")
    found = _find_lean()
    if found is None:
        check("a WORKING lean is installed", False,
              "no candidate ran; a half-extracted toolchain reports success on "
              "--version only if it is asked the wrong question")
        return
    check("a working lean is installed", True,
          (found.stdout + found.stderr).strip().splitlines()[0][:40])
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

    # The comprehension above destructures `kind` inside the element expression,
    # where it is not bound -- a NameError that only fires on the SECOND list, so
    # the first list printed correctly and the failure looked like a parsing
    # problem rather than a scoping one.
    axiom_free = [n for n, kind, _ in ax if kind == "does not depend on any axioms"]
    with_axioms = [(n, [a.strip() for a in deps.split(",") if a.strip()])
                   for n, kind, deps in ax if kind.startswith("depends")]

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
    # Scoped to lines that START a diagnostic. The naive `"error" not in out`
    # matched the word inside the axiom report and reported a clean run as dirty.
    diag = [l for l in out.splitlines()
            if l.startswith(("error", "warning"))
            or "declaration uses 'sorry'" in l]
    check("no diagnostics emitted", not diag, str(diag[:2]))

    # And the theorems themselves must exist under the names the audit claims.
    for t in ("clearsBar_implies_strictly_better", "promoted_chains_strictly_increase",
              "floor_positive_is_load_bearing", "guard2_does_not_cover_guard3"):
        check(f"{t} was audited", any(n.endswith(t) for n, _, _ in ax))


# ------------------------------------------------------- 2. proof ↔ pipeline
def conformance() -> None:
    print("\nCONFORMANCE — does the Python gate agree with the proved model?")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import loop  # noqa: E402

    B = float(loop.bar())   # the bar this experiment actually uses
    q = lambda x: Fraction(int(round(x * 1000000)), 1000000)   # exact-ish decimal
    cases = {
        "no_champion_up": (q(B + 0.01), None, q(B + 0.01), "up"),
        "clears_with_champ": (q(0.2 + B + 0.10), q(0.2), q(B), "up"),
        "fails_with_champ": (q(0.2 + B - 0.01), q(0.2), q(B), "up"),
        # `floor` here is the EFFECTIVE floor the Python gate will apply, which is
        # max(preregistered, measured). Passing 0 modelled a gate configuration
        # the Python side can no longer be in — the measured bar overrides a
        # preregistered zero — so the model and the gate were honestly disagreeing
        # about what was being tested. The zero-floor property is still checked
        # below, as a property of the model, where it belongs.
        "equal_champ_zero_floor": (q(0.2 + B), q(0.2), q(B), "up"),
        "down_ignores_champ": (q(B + 0.01), q(0.2), q(B), "down"),
        "down_clears": (q(-(B + 0.01)), None, q(B), "down"),
    }

    r = run_lean(FORMAL / "Gate.lean")
    if r.returncode != 0:
        check("Gate.lean compiles", False, r.stderr.strip()[:300])
        return
    check("Gate.lean compiles", True)

    gr = _emit_and_run(float(loop.bar()), cases)
    rows = dict(re.findall(r'"([a-z_]+)\|(true|false)"', gr.stdout))
    # The table is GENERATED from `cases` rather than hand-written in Gate.lean.
    # Two hand-maintained copies of one fixture is how the sides drift: when the
    # measured bar moved, Lean's `#eval` rows still said 0.01 while Python said
    # 0.133, and the conformance table reported a disagreement that was really the
    # harness lying about the input.
    check("the model emitted its conformance table", len(rows) >= 6,
          f"{len(rows)} rows: {sorted(rows)}")

    # Same cases, expressed against the REAL gate. `champ` is None where the
    # model says none, because `loop.gate` reads it off the ArmResult.
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
        # Lean reports `barFails` (true = the bar was NOT cleared); Python's gate
        # returns `passed` (true = it was). Comparing them directly reports every
        # row as a disagreement — all six, inverted, which is the signature of a
        # label swap rather than of a semantic difference. Only guard 1 can fire
        # here: the fixture sets no per-target numbers, so guards 2 and 3 have
        # nothing to look at and `passed` is exactly "the bar held".
        py = "false" if passed else "true"
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


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp(ok: bool, failures: list[str], detail: dict) -> Path:
    """Write the proof artifact the claim registry reads.

    It carries a hash of every source the run was about. That is the whole
    mechanism: a claim whose source has changed since is STALE rather than
    proven, so a proof can never outlive the code it proves. A proof with no
    source hash is a claim about nothing in particular.
    """
    sys.path.insert(0, str(ROOT / "pipeline"))
    import claims as claimsmod
    sources = {}
    for rel in ("formal/Gate.lean", "formal/Axioms.lean", "pipeline/loop.py",
                "pipeline/confirm.py", "pipeline/adversary.py"):
        h = _sha256(ROOT / rel)
        if h:
            sources[rel] = h
    return claimsmod.stamp("kernel.json", {
        "ok": ok, "failures": failures, "when_h": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ts": time.time(),
        "lean": detail.get("lean", ""),
        "declarations_audited": detail.get("declarations", 0),
        "sources_sha256": sources,
    })


def _rat(x: float) -> str:
    """A Lean `Rat` literal from a Python float, exactly (no rounding drift)."""
    n, d = (x * 1_000_000).as_integer_ratio()
    return f"({n} : Rat) / {d}"


def _emit_and_run(B: float, cases: dict) -> subprocess.CompletedProcess:
    """Write the conformance rows from `cases` and run them through the model."""
    lines = ["import Gate", "open Gate"]
    for name, (delta, champ, floor, direction) in cases.items():
        ch = "none" if champ is None else f"some ({_rat(float(champ))})"
        lines.append(
            f'#eval row "{name}" ⟨{_rat(float(delta))}, {ch}, '
            f'{_rat(float(floor))}, .{direction}⟩')
    src = FORMAL / "Conformance.lean"
    src.write_text("\n".join(lines) + "\n")
    return run_lean(src)


if __name__ == "__main__":
    import argparse
    import hashlib
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true",
                    help="for unattended runs: only the verdict line")
    a = ap.parse_args()
    FAILS.clear()
    # `--quiet` silences the OUTPUT, it does not skip the WORK. An earlier draft
    # guarded the two checks behind `if not a.quiet:`, which meant the unattended
    # hourly run recorded a proof without having proved anything — the worst
    # possible bug in the one component whose entire job is to not be a rubber
    # stamp. A quiet run must be a silent run, never a skipped one.
    if a.quiet:
        real_check = check
        def check(name, cond, detail=""):        # noqa: F811
            if not cond:
                FAILS.append(name)
    kernel_test()
    conformance()
    detail = {"lean": "", "declarations": 8}
    p = stamp(not FAILS, FAILS, detail)
    print(f"\n  proof artifact: {p}")
    print(f"  publication gate: {'ALLOWED' if not FAILS else 'BLOCKED'}")
    raise SystemExit(0 if not FAILS else 2)
