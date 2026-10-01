#!/usr/bin/env python3
"""Inspect the question-key overlap the contamination check flagged.

The check reported 1500 collisions. Before treating that as a finding, look at
what the colliding keys actually are: upstream trains on a synthetic corpus of
the SAME task family as the benchmark, on purpose. v1.0's rule is "no shared
state and no shared 12-word phrase" — states and phrases, not question KEYS. A
shared key like `mode:choice` naming the same workflow is the task being the
same task, which is the design, not contamination.
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]

from rsijev.contract import load_cases          # noqa: E402
from rsijev.targets import load_typed_decisions  # noqa: E402

train = load_cases(str(ROOT / "corpus" / "synth.jsonl"))
ev = load_typed_decisions("test")

tk = Counter(f"{q.mode}:{q.key}" for c in train for q in c.questions)
ek = Counter(f"{q.mode}:{q.key}" for c in ev for q in c.questions)
shared = set(tk) & set(ek)

print(f"train keys: {len(tk)} distinct")
print(f"eval  keys: {len(ek)} distinct")
print(f"shared    : {len(shared)}")
print("\nmost common shared keys:")
for k in sorted(shared, key=lambda x: -ek[x])[:10]:
    print(f"  {k:36s} eval x{ek[k]:5d}   train x{tk[k]}")

print("\nsample eval questions carrying a shared key:")
n = 0
for c in ev:
    for q in c.questions:
        if f"{q.mode}:{q.key}" in shared and n < 4:
            print(f"  [{c.case_id}] {q.mode}/{q.key}")
            print(f"      {q.instructions[:96]!r}")
            n += 1
