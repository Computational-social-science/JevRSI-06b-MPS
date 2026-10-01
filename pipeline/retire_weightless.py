#!/usr/bin/env python3
"""Retire a keeper that has no weights behind it. Append-only, idempotent, logged.

## Why this exists

`null_floor` cleared a bar and was recorded `kept_pending_confirm` having been run
with `ckpt=no`. Four arms later `ckpt/` was empty, `champion.json` did not exist,
and invariant I14 reported a critical: a kept arm with no weights on disk cannot be
re-scored from disk, cannot be verified per question, cannot be published, and
cannot be spent on the held-out set.

The finding is true and it is also permanent, because the run it is about is in the
past. Left alone it halts the loop forever on something it cannot act on — the
shape of failure this project has now hit three times in three different forms
(a halt outliving its cause, a loop not restarted after a clear, an invariant
claimed by no auditor).

So the record is corrected the only way a record should be: by appending. A
superseding row is written to `records/arms.jsonl` with the verdict the evidence
supports, the reason in words, and a pointer to `records/supersessions.jsonl`.
Nothing is edited in place, because an edited record cannot be distinguished from
one that was never wrong.

This is NOT the same as deleting the arm. The measurement stands: it ran, it
produced 2,000 scored questions, and its delta is a real point in the null series.
What does not stand is the VERDICT — it was assigned to an arm with no artifact,
which is not a keeper. The arm stays in the floor; the crown is withdrawn.

Usage:  python pipeline/retire_weightless.py [--dry-run]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORDS = ROOT / "records" / "arms.jsonl"
AUDIT = ROOT / "records" / "supersessions.jsonl"
CKPT = ROOT / "ckpt"
KEEPERS = ("kept", "kept_pending_confirm")


def rows() -> list[dict]:
    if not RECORDS.is_file():
        return []
    out = []
    for line in RECORDS.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def has_weights(arm: str) -> bool:
    d = CKPT / arm
    if not d.is_dir():
        return False
    return any(d.rglob("*.pt")) or any(d.rglob("*.safetensors"))


def main() -> int:
    dry = "--dry-run" in sys.argv
    # The last row per arm wins, because arms.jsonl is append-only and a superseded
    # row is itself a row. Reading only the final verdict is what keeps this
    # idempotent: run it twice and the second run finds nothing to do.
    latest: dict[str, dict] = {}
    for r in rows():
        latest[str(r.get("arm", ""))] = r
    already = set()
    if AUDIT.is_file():
        for line in AUDIT.read_text().splitlines():
            if line.strip():
                try:
                    already.add(json.loads(line)["arm"])
                except (json.JSONDecodeError, KeyError):
                    pass

    targets = []
    for arm, r in sorted(latest.items()):
        if r.get("verdict") not in KEEPERS or arm in already:
            continue
        if has_weights(arm):
            continue
        targets.append((arm, r))

    if not targets:
        print("  nothing to retire: every kept arm has weights on disk")
        return 0

    for arm, r in targets:
        delta = ""
        c, k = r.get("pooled_top1_candidate"), r.get("pooled_top1_control")
        if c is not None and k is not None:
            delta = f"{c - k:+.4f}"
        reason = (
            f"verdict withdrawn, measurement retained. Recorded "
            f"{r.get('verdict')} with no checkpoint on disk, so there was nothing "
            f"to re-score from disk, verify per question, publish, or spend on the "
            f"held-out set. Its delta ({delta}) remains a real point in the null "
            f"series; what does not stand is the crown. Retired by "
            f"pipeline/retire_weightless.py on 2026-10-01 under invariant I14."
        )
        print(f"  {'would retire' if dry else 'retiring'} {arm} "
              f"({r.get('verdict')} -> rejected, delta {delta})")
        if dry:
            continue
        sup = dict(r)
        sup["verdict"] = "rejected"
        sup["passed"] = False
        sup["reason"] = reason
        sup["superseded_verdict"] = r.get("verdict")
        sup["superseded_by"] = "pipeline/retire_weightless.py (I14)"
        sup["ts"] = time.time()
        with RECORDS.open("a") as fh:
            fh.write(json.dumps(sup) + "\n")
            fh.flush()
        with AUDIT.open("a") as fh:
            fh.write(json.dumps({"arm": arm, "action": "verdict_withdrawn",
                                 "from": r.get("verdict"), "to": "rejected",
                                 "delta": delta, "invariant": "I14",
                                 "ts": sup["ts"]}) + "\n")
            fh.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
