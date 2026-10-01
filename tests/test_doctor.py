#!/usr/bin/env python3
"""The arbiter must not be decoration.

Four auditors that have only ever said "healthy" are indistinguishable from four
that always say "healthy". This drives the real `doctor.arbitrate()` with
hand-built auditor outputs, including the case the design exists for: several
auditors agreeing everything is fine while one reports a methodological critical.

The property under test is that a critical is never averaged away. That is the
whole reason the arbiter exists rather than a single fused check: with four
auditors, a naive severity-mean turns one critical into a 25% problem, and a
bar that moved down gets filed as a yellow.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import doctor  # noqa: E402
from invariants import Finding  # noqa: E402

FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def ok(inv, title="fine"):
    return Finding(inv, title, ok=True, severity="info")


def bad(inv, title, severity):
    return Finding(inv, title, ok=False, severity=severity,
                   detail="planted for the test")


def main() -> int:
    print("the healthy case")
    rep = doctor.arbitrate({
        "vitals": [ok("V1"), ok("V2")],
        "provenance": [ok("I1"), ok("I3")],
        "methodology": [ok("I5"), ok("I7")],
        "integrity": [ok("I9"), ok("I12")],
        "adversary": [ok("A1"), ok("A4")],
    })
    check("healthy when everything holds", rep.ok and rep.verdict == "healthy",
          rep.verdict)

    print("\nTHE CASE THIS EXISTS FOR: one methodological critical, three oks")
    rep = doctor.arbitrate({
        "vitals": [ok("V1"), ok("V2"), ok("V3")],
        "provenance": [ok("I1"), ok("I3")],
        "methodology": [bad("I5", "the noise floor", "critical")],
        "integrity": [ok("I9"), ok("I12")],
        "adversary": [ok("A1"), ok("A4")],
    })
    check("the critical is NOT diluted by three agreeing auditors",
          (not rep.ok) and len(rep.criticals) == 1, str(rep.criticals))
    check("... and is reported as WRONG, not merely degraded",
          rep.verdict == "wrong", rep.verdict)
    check("... and the diagnosis names the class of danger",
          "keeps running" in rep.diagnosis or "dangerous" in rep.diagnosis
          or "publishing numbers" in rep.diagnosis, rep.diagnosis[:80])
    check("... and it needs no human decision (it has no repair)",
          not rep.needs_human)

    print("\nseverity ordering between the classes")
    rep_broken = doctor.arbitrate({
        "vitals": [bad("V1", "the loop died", "critical")],
        "provenance": [ok("I1")], "methodology": [ok("I5")], "integrity": [ok("I9")],
    })
    check("an operational critical is BROKEN, not WRONG",
          rep_broken.verdict == "broken", rep_broken.verdict)

    rep_both = doctor.arbitrate({
        "vitals": [bad("V1", "the loop died", "critical")],
        "provenance": [ok("I1")],
        "methodology": [bad("I6", "queue order", "major")],
        "integrity": [ok("I9")],
    })
    check("methodological + operational together is still WRONG, not BROKEN",
          rep_both.verdict == "wrong", rep_both.verdict)
    check("... because wrong is the one nobody notices alone",
          "dangerous" in rep_both.diagnosis or "keeps running" in rep_both.diagnosis,
          rep_both.diagnosis[:80])

    print("\nmajor-only is degraded, not an emergency")
    rep_deg = doctor.arbitrate({
        "vitals": [ok("V1")],
        "provenance": [ok("I1")],
        "methodology": [bad("I5", "floor not measured", "minor")],
        "integrity": [ok("I9")],
    })
    check("a minor failure is degraded", rep_deg.verdict == "degraded", rep_deg.verdict)

    print("\nrepairs are separated by whether a machine may apply them")
    rep_rep = doctor.arbitrate({
        "vitals": [Finding("V7", "wedged", ok=False, severity="major",
                            detail="", repair="kill the wedged arm")],
        "provenance": [ok("I1")], "methodology": [ok("I5")], "integrity": [ok("I9")],
    })
    check("a safe repair is offered unattended",
          any(r["safe_unattended"] for r in rep_rep.repairs), str(rep_rep.repairs))
    check("and nothing needs a human here", not rep_rep.needs_human)

    rep_human = doctor.arbitrate({
        "vitals": [Finding("V4", "sleep", ok=False, severity="minor",
                            detail="", repair="change the power policy")],
        "provenance": [ok("I1")], "methodology": [ok("I5")], "integrity": [ok("I9")],
    })
    # severity minor -> safe is False by the rule; the point is the LABEL exists
    check("every repair carries a safe/unsafe label",
          all("safe_unattended" in r for r in rep_human.repairs))

    print("\nthe real pipeline is diagnosed by the real path")
    r = doctor.diagnose()
    check("diagnose() runs and returns a report", isinstance(r, doctor.Report))
    check("it names all five auditors",
          set(r.auditors) == {"vitals", "provenance", "methodology", "integrity",
                              "adversary"},
          str(sorted(r.auditors)))
    check("the adversary lens recomputes numbers rather than reading the record",
          any(str(f.get("invariant", "")).startswith("A")
              for f in r.auditors["adversary"]["findings"]),
          "provenance and methodology are transcription checks; this one is not")
    check("the live pipeline is currently healthy", r.ok,
          f"{r.verdict}: crit={r.criticals} maj={r.majors}")

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())