#!/usr/bin/env python3
"""The contamination check must FIRE, not merely return zero.

A check that reports "clean" on a corpus it was never shown to work is
indistinguishable from a check that is broken. So every criterion is exercised
against a planted violation, and the clean case is exercised too — because a
check that fires on everything is equally worthless.

This is the negative-control discipline from `tests/test_no_cost.py` applied to
the science rather than the budget: plant the violation, confirm the detector
notices, remove it, confirm the detector goes quiet.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

import contamination as C           # noqa: E402
from rsijev.contract import Case, Question  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def case(cid, state, source="train"):
    return Case(case_id=cid, source=source, state=state,
                questions=(Question(key="q", mode="choice", instructions="i",
                                    options=("a", "b"),
                                    criteria={"a": "x", "b": "y"}),),
                gold={"q": (1.0, 0.0)})


def shingle(s, n=C.SHINGLE_N):
    return " ".join(s.split()[:n])


def main() -> int:
    # A long, distinctive base document, so a planted overlap is unambiguous.
    base = ("the deployment pipeline failed at 03:14 and the on-call engineer "
            "paged the database team because replication lag exceeded the "
            "threshold configured in the production cluster settings file")
    corpus = {"synth": [case("t1", base), case("t2", base + " and then recovered")]}
    targets = {"eval": [case("e1", "an entirely unrelated passage about kitchen "
                                   "gardening tools and their maintenance schedule")]}

    print("the clean case")
    r = C.check(corpus, targets)
    check("a disjoint corpus is clean", r.clean, r.verdict or r.one_line())
    check("and says so in words", "clean" in (r.verdict or r.one_line()))
    check("it counted both sides", r.n_train_cases == 2 and r.n_eval_cases == 1)

    print("\ncriterion 1: an EXACT shared state")
    bad = {"synth": [case("t1", base)], "eval": [case("e1", base)]}
    r = C.check(bad, {"eval": [case("e1", base)]})
    check("an exact state collision is caught", not r.clean)
    check("... and named as CONTAMINATED", "CONTAMINATED" in r.verdict, r.verdict[:60])
    check("... with the case ids", "t1" in r.verdict and "e1" in r.verdict)

    print("\ncriterion 2: a NEAR-duplicate state (no exact hash match)")
    # Same opening 12 words, different tail -> different hash, same phrase.
    near = base + " but the incident was later attributed to a bad certificate"
    r = C.check({"synth": [case("t1", base)]}, {"eval": [case("e1", near)]})
    check("a shared 12-word phrase is caught", not r.clean, r.verdict[:70])
    check("... and named as NEAR-DUPLICATE", "NEAR-DUPLICATE" in r.verdict)
    check("... and the exact-hash count stayed 0", r.exact_state_collisions == 0,
          "the two criteria are independent")

    print("\ncriterion 3: a document too short to have a 12-gram")
    short = "totally fine"
    r = C.check({"synth": [case("t1", short)]}, {"eval": [case("e1", short)]})
    check("a short shared state is still caught (exact hash)", not r.clean)
    r = C.check({"synth": [case("t1", short)]},
                {"eval": [case("e1", "a different short thing")]})
    check("two different short states are clean", r.clean)

    print("\nthe criterion that was REMOVED, and why")
    # Same task name, different document. This is the design, not contamination,
    # and it is exactly what the first version of the check wrongly flagged.
    q_a = Case(case_id="t", source="train", state=base,
               questions=(Question(key="urgency", mode="score", instructions="i",
                                   options=("0", "1"), criteria={"0": "a", "1": "b"}),),
               gold={"urgency": (1.0, 0.0)})
    q_b = Case(case_id="e", source="eval", state="a wholly different document here",
               questions=(Question(key="urgency", mode="score", instructions="i",
                                   options=("0", "1"), criteria={"0": "a", "1": "b"}),),
               gold={"urgency": (0.0, 1.0)})
    r = C.check({"synth": [q_a]}, {"eval": [q_b]})
    check("a shared question KEY with a different document is NOT contamination",
          r.clean, r.verdict or r.one_line())
    check("... and the key count is reported as 0, not hidden",
          r.question_key_collisions == 0,
          "the criterion stays in the record so a reader sees it was considered")

    print("\nthe real corpora, on disk")
    from rsijev.contract import load_cases
    from rsijev.targets import load_mmlu_pro_1k, load_typed_decisions
    corpus_real = {f.stem: load_cases(str(f))
                   for f in sorted((ROOT / "corpus").glob("*.jsonl"))}
    targets_real = {"typed_decisions": load_typed_decisions("test"),
                    "mmlu_pro_guard": load_mmlu_pro_1k()[:200]}
    r = C.check(corpus_real, targets_real)
    check(f"the actual corpus is clean ({r.n_train_cases} vs {r.n_eval_cases})",
          r.clean, r.verdict or r.one_line())
    check("... with 0 exact and 0 near", r.exact_state_collisions == 0
          and r.near_duplicate_states == 0)

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
