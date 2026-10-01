#!/usr/bin/env python3
"""Does the claim gate actually BLOCK, and does staleness actually bite?

A publication gate that has only ever said yes is a comment. So this does the
thing that matters: it takes a system where every claim is proven, and then makes
each of the three blocking conditions true in turn, requiring the gate to refuse.

The case that matters most is the third. A proof that ran before the code changed
is the formal version of reporting a measurement taken before you changed the
instrument — and it is the failure that a naive "is there a proof file?" check
waves through forever. So the source is edited by one byte here, and the claim
must go STALE even though the proof artifact is untouched and still says `ok`.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    import claims as C

    print("1. the healthy case: every claim proven, publication allowed")
    s = C.summary()
    check("all claims proven", s["may_publish"], str(s["unproven"]))
    check("the proof artifact records a source hash",
          bool(json.loads((C.CLAIM_DIR / "kernel.json").read_text())
               .get("sources_sha256")), "otherwise a claim is about nothing")
    check("a proven claim names when it was proven",
          all(c["proved_at"] for c in s["claims"] if c["status"] == "proven"))

    print("\n2. no proof at all -> every claim is made without proof")
    saved = (C.CLAIM_DIR / "kernel.json").read_text()
    try:
        (C.CLAIM_DIR / "kernel.json").unlink()
        s2 = C.summary()
        check("publication is BLOCKED", not s2["may_publish"])
        check("the claims are reported MISSING, not silently fine",
              all(c["status"] == "missing" for c in s2["claims"]), str(s2["counts"]))
        check("... and the reason says the claim is unbacked",
              any("WITHOUT PROOF" in c["detail"] for c in s2["claims"]))
    finally:
        (C.CLAIM_DIR / "kernel.json").write_text(saved)

    print("\n3. the check ran and FAILED -> blocked, even though a proof file exists")
    pr = json.loads(saved)
    pr2 = dict(pr, ok=False, failures=["Gate.lean does not compile"])
    try:
        (C.CLAIM_DIR / "kernel.json").write_text(json.dumps(pr2))
        s3 = C.summary()
        check("publication is BLOCKED by a failed proof", not s3["may_publish"])
        check("the status is UNPROVEN, not MISSING",
              all(c["status"] == "unproven" for c in s3["claims"]), str(s3["counts"]))
        check("... and the failure is surfaced",
              any("does not compile" in c["detail"] for c in s3["claims"]))
    finally:
        (C.CLAIM_DIR / "kernel.json").write_text(saved)

    print("\n4. THE CASE THAT MATTERS: the source moved after the proof")
    # One byte. The proof artifact is untouched and still says ok:true. If the
    # gate only asks "is there a proof?", it waves this through — and that is a
    # report of a measurement taken before the instrument was changed.
    target = ROOT / "formal" / "Gate.lean"
    original = target.read_text()
    try:
        target.write_text(original + "\n-- touched by test_claims\n")
        s4 = C.summary()
        check("publication is BLOCKED after the source changes", not s4["may_publish"],
              str(s4["unproven"]))
        stale = [c for c in s4["claims"] if c["status"] == "stale"]
        check("the affected claim is STALE, not proven", bool(stale), str(s4["counts"]))
        check("... and the reason names the file that changed",
              any("Gate.lean has changed" in c["detail"] for c in stale),
              stale[0]["detail"][:70] if stale else "")
        # And a claim whose source did NOT move must still be proven, or the gate
        # is just a blunt instrument that blocks everything.
        others = [c for c in s4["claims"] if c["source"] != "formal/Gate.lean"]
        check("claims about untouched sources are unaffected",
              all(c["status"] == "proven" for c in others),
              str([(c["name"], c["status"]) for c in others]))
    finally:
        target.write_text(original)

    print("\n5. a proof older than its environment")
    pr3 = dict(pr, ts=time.time() - 60 * 86400)     # 60 days
    try:
        (C.CLAIM_DIR / "kernel.json").write_text(json.dumps(pr3))
        s5 = C.summary()
        check("a 60-day-old proof is refused", not s5["may_publish"])
        check("... and it says how old",
              any("days old" in c["detail"] for c in s5["claims"]),
              s5["claims"][0]["detail"][:60])
    finally:
        (C.CLAIM_DIR / "kernel.json").write_text(saved)

    print("\n6. restored")
    s6 = C.summary()
    check("everything is proven again", s6["may_publish"], str(s6["unproven"]))

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
