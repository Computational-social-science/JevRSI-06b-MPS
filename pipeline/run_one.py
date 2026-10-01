#!/usr/bin/env python3
"""Run one arm in its own process and write `result.json`.

EDITABLE — ours, and deliberately minimal. Everything it does is: read the spec,
call `pipeline.arm.run_arm`, serialise the result. The arm itself is upstream's
`run_arm_lib.run_arm`, unmodified.

A separate process per arm is the point. A long-lived worker holding the
backbone in memory would save a model load per arm, but an arm that segfaults,
OOMs or diverges would take the worker with it and the trajectory would lose its
clock. At 0.6B the load costs seconds, and a clean exit code is what launchd's
`StartInterval` and the daemon's subprocess isolation actually need.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "pipeline"))

import dev as devmod          # noqa: E402
from agenda import AGENDA, Hypothesis  # noqa: E402
from arm import run_arm       # noqa: E402
from loop import Prereg, RECORDS  # noqa: E402
import confirm as confirmmod    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--guard-n", type=int, default=200)
    a = ap.parse_args()

    by_name = {h.name: h for h in AGENDA}
    if a.arm not in by_name:
        # A confirmation arm is not in the agenda -- it is synthesised from the
        # pending-confirmation record, with the fresh seed the loop owes. Its
        # spec arrives in --spec, so all that is needed here is a placeholder
        # whose name matches, so the run is recorded and gated like any other.
        pend = confirmmod.pending()
        if not (pend and pend.get("confirm_arm") == a.arm):
            print(f"unknown arm {a.arm!r}; known: {sorted(by_name)} "
                  f"(plus a pending confirmation, if one is owed)", file=sys.stderr)
            return 1
        by_name[a.arm] = Hypothesis(
            name=a.arm, axis="training",
            change=f"confirmation of {pend['arm']} on fresh seed {pend['seed']}",
            spec={}, prediction="", source="pending confirmation")
    h = by_name[a.arm]
    spec = json.loads(Path(a.spec).read_text())

    # Reconstruct the prereg from the log the daemon fsynced BEFORE launching us.
    # Reading it back rather than re-deriving it is the point: the prediction
    # that gates this result is the one that was on disk first.
    prereg = None
    p = RECORDS / "prereg.jsonl"
    if p.is_file():
        for line in reversed(p.read_text().splitlines()):
            if line.strip():
                r = json.loads(line)
                if r.get("arm") == a.arm:
                    prereg = r
                    break
    if prereg is None:
        prereg = Prereg(arm=a.arm, axis=h.axis, change=h.change,
                        prediction=h.prediction).to_json()
        print(f"  [warn] no prereg found for {a.arm}; using the agenda's text")

    # THE SEED IS CHECKED, NOT TRUSTED. The promise is the preregistration on disk
    # (falling back to the agenda for rows written before the seed was pinned);
    # the spec is the thing about to be executed. If they disagree, this arm is
    # not the arm that was predicted, and an hour of compute would produce a
    # number that cannot be compared with anything — including the floor it is
    # supposed to help measure.
    #
    # This is not hypothetical. `null2` ran at seed 17 while 37 was written down,
    # and returned rows bit-identical to the seed-17 `null_floor` beside it: two
    # arm names, one measurement, sitting in `records/` where a reader would take
    # them for two points of a spread. The refusal costs a second; the duplicate
    # cost a day of the record's credibility.
    promised = prereg.get("seed")
    if promised is None:
        promised = (h.spec or {}).get("seed")
    if promised is not None and spec.get("seed") != promised:
        print(f"REFUSING {a.arm}: the prediction names seed {promised} but "
              f"{a.spec} says {spec.get('seed')!r}. Refusing rather than "
              f"measuring the wrong thing.", file=sys.stderr)
        return 1

    res = run_arm(
        arm=a.arm, spec=spec, prereg=Prereg(**prereg),
        corpus_dir=Path(a.corpus), model_id=a.model,
        save_dir=Path(a.save_dir) if a.save_dir else None,
        guard_n=a.guard_n,
    )
    out = RECORDS / a.arm / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(res), indent=2, default=str) + "\n")
    print(f"\nwrote {out}  verdict={res.verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
