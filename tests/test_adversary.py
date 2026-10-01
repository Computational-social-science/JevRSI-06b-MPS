#!/usr/bin/env python3
"""Do the adversarial probes fire? Plant the violation, confirm it is caught.

`adversary.py` was written, wired into the doctor, and reported all-green on the
first run of a healthy pipeline. That is exactly the state a checker should never
be trusted in, and it is the state this project has now been bitten by three times
in three different forms:

  * the cost guard flagged its own detection literals,
  * the invariants module flagged itself on I12,
  * and A4 — written minutes earlier — invented a bar rule the pipeline has never
    used and raised a critical against a healthy run.

A probe that has only ever agreed with the system is indistinguishable from a
probe that does nothing. So each one gets a violation planted in a temporary
state directory, and each must catch it.

The A1 case is the expensive one and the reason this file exists: it corrupts a
*headline number* in the record while leaving every internal field — the delta,
the verdict, the per-target table — perfectly consistent with each other. A
transcription check cannot see it, because the record is coherent. Only
recomputing from the 18k raw rows underneath can.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def plant(mutate_records=None, mutate_power=None, records_src=None, power_src=None):
    """Build an isolated state dir, plant a violation, run the probes on it."""
    import adversary
    import loop
    import power

    tmp = Path(tempfile.mkdtemp(prefix="adv-"))
    try:
        rows = [json.loads(l) for l in
                (records_src or ROOT / "records" / "arms.jsonl").read_text().splitlines()
                if l.strip()]
        if mutate_records:
            mutate_records(rows)
        (tmp / "arms.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

        # The per-question rows are megabytes and must NOT be copied, but A1 is
        # precisely the probe that needs them -- so link the real arm directories
        # into the temp tree rather than redirecting ROOT away from them.
        (tmp / "records").mkdir(exist_ok=True)
        for d in (ROOT / "records").iterdir():
            if d.is_dir():
                (tmp / "records" / d.name).symlink_to(d)

        pw = json.loads((power_src or ROOT / "state" / "power.json").read_text())
        if mutate_power:
            mutate_power(pw)
        (tmp / "power.json").write_text(json.dumps(pw, default=str))

        saved = (power.STATE, adversary.ROOT)
        power.STATE = tmp
        adversary.ROOT = tmp
        try:
            return adversary.probe(rows)
        finally:
            power.STATE, adversary.ROOT = saved
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def find(F, inv):
    return next((f for f in F if f.invariant == inv), None)


def main() -> int:
    real = ROOT / "records" / "arms.jsonl"
    power_json = ROOT / "state" / "power.json"

    print("the healthy case: the real, untouched state")
    F = plant(records_src=real, power_src=power_json)
    bad = [f.invariant for f in F if not f.ok]
    check("every probe agrees on the real record", not bad, str(bad))

    print("\nA1: a headline number that does not come from its own raw rows")
    # Corrupt ONLY the per-target table. The pooled numbers, the delta and the
    # verdict all stay mutually consistent, so a coherence check sees nothing.
    def bump(rows):
        for r in rows:
            if r.get("per_target_top1"):
                r["per_target_top1"] = {k: min(0.99, v + 0.01)
                                         for k, v in r["per_target_top1"].items()}
    F = plant(mutate_records=bump, records_src=real, power_src=power_json)
    a1 = find(F, "A1")
    check("A1 fires on a headline that cannot be reproduced", not a1.ok,
          a1.detail[:70])
    check("... and is CRITICAL", a1.severity == "critical")
    check("... and shows recomputed beside recorded",
          any("recomputed" in str(e) and "recorded" in str(e) for e in a1.evidence),
          str(a1.evidence[:1]))

    print("\nA2: a delta that does not follow from its own parts")
    def bad_delta(rows):
        for r in rows:
            r["champion_delta"] = 0.5      # candidate - control is +0.0665
    F = plant(mutate_records=bad_delta, records_src=real, power_src=power_json)
    a2 = find(F, "A2")
    check("A2 fires on an inconsistent delta", not a2.ok, a2.detail[:70])
    check("... and is CRITICAL", a2.severity == "critical")

    print("\nA3: the control drifting between arms")
    def drift(rows):
        rows.append(dict(rows[0], arm="synthetic_second", champion_delta=0.01))
        rows[-1]["pooled_top1_control"] = 0.50      # was 0.4645
        rows[-1]["per_target_top1_control"] = dict(rows[0]["per_target_top1_control"])
    F = plant(mutate_records=drift, records_src=real, power_src=power_json)
    a3 = find(F, "A3")
    check("A3 fires when the control is not shared", not a3.ok, a3.detail[:70])
    check("... and is CRITICAL", a3.severity == "critical")
    check("... and names both control values",
          any("0.464500" in str(e) for e in a3.evidence)
          and any("0.500000" in str(e) for e in a3.evidence), str(a3.evidence[:3]))

    print("\nA4: a bar that was adjusted by hand")
    def hand_tuned(pw):
        pw["recommended_bar"] = 0.005        # below the rule's 0.020
    F = plant(mutate_power=hand_tuned, records_src=real, power_src=power_json)
    a4 = find(F, "A4")
    check("A4 fires on a hand-tuned bar", not a4.ok, a4.detail[:70])
    check("... and shows the rule's answer beside the recorded one",
          any("re-derived" in str(e) for e in a4.evidence), str(a4.evidence[:3]))

    print("\nA5: a floor that is not the null arms' own spread")
    def invented(pw):
        pw["n_null_arms"] = 4
        pw["null_deltas"] = [0.0665, 0.0700, 0.0710, 0.0720]   # sd ~0.0024
        pw["paired_sd"] = 0.011                               # claims the fallback
    F = plant(mutate_power=invented, records_src=real, power_src=power_json)
    a5 = find(F, "A5")
    check("A5 fires on a claimed-measured floor that is not the nulls' spread",
          (not a5.ok), f"{a5.ok} {a5.detail[:60]}")
    check("... and is at least major", a5.severity in ("major", "critical"), a5.severity)

    print("\nthe healthy path is not vacuous: the fallback is NOT a violation")
    def still_fallback(pw):
        pw["n_null_arms"] = 1
        pw["null_deltas"] = [0.0665]
    F = plant(mutate_power=still_fallback, records_src=real, power_src=power_json)
    a5b = find(F, "A5")
    check("A5 is quiet while the floor is the declared fallback", a5b.ok,
          a5b.detail[:60])
    check("... and says so as a limitation, not a pass",
          "limitation" in a5b.detail or "fallback" in a5b.detail, a5b.detail[:60])

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
