#!/usr/bin/env python3
"""Backfill the contamination verdict onto records written before the check ran.

`null_floor` was scored, gated, written to `versions/current.md` and shown on the
dashboard with no evidence that the training corpus and the evaluation targets
were disjoint — `contamination.py` and invariant I9 both existed, and neither was
ever connected to the daemon. I9 caught it on the first real record.

Rewriting an append-only log is normally the wrong move, so this is deliberately
narrow and deliberately visible:

  * it only ADDS a `contamination` field to records that have none;
  * it never touches a score, a verdict, a delta or a timestamp;
  * the backfilled value is prefixed `verified retroactively`, so a reader can
    tell a check that ran before the arm from one run afterwards;
  * it refuses to touch a record that already has a verdict from a real run.

The check is a fact about two fixed files (corpus/*.jsonl and the eval targets),
so running it now says the same thing it would have said then. What is being
backfilled is the *evidence*, not a changed conclusion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import arm as armlib            # noqa: E402
import contamination            # noqa: E402
import loop                     # noqa: E402

RECORDS = ROOT / "records" / "arms.jsonl"
CONFIG = json.loads((ROOT / "config" / "run.json").read_text())


def main() -> int:
    if not RECORDS.is_file():
        print("no records file; nothing to backfill")
        return 0
    rows = [json.loads(l) for l in RECORDS.read_text().splitlines() if l.strip()]
    todo = [r for r in rows if not r.get("contamination")]
    if not todo:
        print(f"all {len(rows)} record(s) already carry a contamination verdict")
        return 0

    targets = armlib.load_targets(CONFIG.get("guard_n", 200))
    train = armlib.load_corpus(ROOT / "corpus")
    rep = contamination.check(train, targets)
    line = rep.one_line()

    for r in todo:
        r["contamination"] = f"verified retroactively: {line}"
        print(f"  {r.get('arm')}: {r['contamination'][:100]}")

    tmp = RECORDS.with_suffix(".jsonl.bak")
    RECORDS.replace(tmp)
    with RECORDS.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"\n  {len(todo)} record(s) backfilled; previous file kept at {tmp.name}")
    print(f"  contamination verdict: {line}")

    # And confirm the invariant that caught it now holds.
    import importlib
    import invariants
    importlib.reload(invariants)
    i9 = next(f for f in invariants.check_all() if f.invariant == "I9")
    print(f"\n  I9 now: {'HOLDS' if i9.ok else 'STILL FAILING'} — {i9.detail[:70]}")
    return 0 if i9.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
