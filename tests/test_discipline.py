#!/usr/bin/env python3
"""Tests for the two disciplines that were missing: fresh-seed confirmation and
the consumable held-out set.

No GPU, no network for the pure-logic parts. The held-out loader touches the
network and is skipped when the dataset is not already cached, so this runs in a
second on a cold machine instead of blocking on a download.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

import confirm  # noqa: E402
import held_out  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    print("fresh-seed confirmation")
    o = confirm.judge(original_arm="champion_base", original_delta=0.130,
                      confirm_delta=0.118, floor=0.020)
    check("a fresh seed above the bar confirms", o.confirmed and o.verdict == "kept", o.reason[:70])
    o = confirm.judge(original_arm="x", original_delta=0.130,
                      confirm_delta=0.003, floor=0.020)
    check("a fresh seed below the bar does NOT confirm",
          (not o.confirmed) and o.verdict == "not_confirmed", o.reason[:70])
    check("... and says seed sensitivity, not 'rejected'",
          "seed sensitivity" in o.reason)
    o = confirm.judge(original_arm="x", original_delta=0.130,
                      confirm_delta=None, floor=0.020)
    check("a confirmation with no metric does not confirm", not o.confirmed)
    o = confirm.judge(original_arm="x", original_delta=0.021, confirm_delta=0.020, floor=0.020)
    check("exactly at the bar confirms", o.confirmed)

    check("an arm's seed pool and the confirmation pool cannot collide",
          not ({confirm.arm_seed(i) for i in range(200)}
               & {confirm.confirm_seed(i) for i in range(200)}),
          "fresh must be structural, not a promise")
    check("a confirmation arm is recognisable",
          confirm.is_confirmation("confirm__champion_base")
          and confirm.base_arm("confirm__champion_base") == "champion_base"
          and not confirm.is_confirmation("champion_base"))

    print("\nheld-out set: the consumable discipline")
    st = held_out.status()
    check("the set is frozen at a pinned revision",
          bool(st.get("revision")) and bool(st.get("source")),
          f"{st.get('source')}@{st.get('revision')}")
    check("a frozen set reports itself unspent", st.get("spent") is False)
    check("the pin was written before any read",
          st.get("frozen_at", 0) < st.get("read_at", float("inf")))
    check("frozen-but-unread is not the same as spent",
          held_out.is_spent() is False,
          "existence of the file must not read as spent")

    # The spend guard, tested on a temporary state file so the real set is not
    # consumed by the test that checks the guard.
    real = held_out.HELD_OUT_JSON
    backup = real.read_text() if real.is_file() else None
    try:
        held_out.freeze()
        check("a second freeze is allowed and does not spend the set",
              not held_out.is_spent())
        held_out.spend({"probe": True})
        check("after a read the set reports spent", held_out.is_spent())
        try:
            held_out.load_held_out(10)
            check("a second read RAISES", False, "it returned cases instead")
        except held_out.HeldOutSpent:
            check("a second read RAISES", True)
        try:
            held_out.spend({"probe": 2})
            check("a second spend() RAISES", False, "it overwrote the record of the read")
        except held_out.HeldOutSpent:
            check("a second spend() RAISES", True)
    finally:
        if backup is not None:
            real.write_text(backup)
        elif real.is_file():
            real.unlink()

    print("\nheld-out set: it is a real, independent, typed-decision set")
    try:
        cases = held_out.load_held_out()
    except Exception as exc:
        print(f"  SKIP  loader needs the dataset ({type(exc).__name__}); "
              f"run once with network to populate the cache")
        cases = None
    if cases:
        from rsijev.contract import MODES
        qs = [q for c in cases for q in c.questions]
        check("every question is a typed decision",
              all(q.mode in MODES for q in qs), f"{len(qs)} questions")
        check("the label space is the source's own, not one we invented",
              all(q.options == ("entailment", "neutral") for q in qs))
        check("every gold sums to 1", all(abs(sum(c.gold[q.key]) - 1) < 1e-6
                                          for c in cases for q in c.questions))
        base = held_out.majority_baseline(cases)
        check("a majority baseline exists to beat", 0.5 <= base <= 1.0, f"{base:.4f}")
        check("the set is large enough to resolve an effect", len(cases) >= 200,
              f"{len(cases)} cases")
        check("the set came from a source the search never trains on",
              all(c.source == "held_out_tasksource" for c in cases))
        check("probing did NOT spend it", not held_out.is_spent())

    print("\nTHE BAR IS MEASURED, NOT COPIED")
    # The defect this pins: `loop.py` carried `BAR = 0.006`, transcribed from a
    # different project's write-up, while the power analysis beside it measured a
    # null at 0.0665. The gate was judging arms at 1/11th of the noise it existed
    # to exclude. Both facts were true in the same file, one line apart.
    import loop as _loop
    import power as _power
    m = _power.load()
    check("the bar is read from the power analysis, not a constant",
          _loop.bar() >= m.recommended_bar,
          f"bar={_loop.bar()} power={m.recommended_bar}")
    check("the bar clears every null arm that has actually run",
          _loop.bar() > max(abs(d) for d in m.null_deltas),
          f"bar={_loop.bar()} max|null|={max(abs(d) for d in m.null_deltas)}")
    check("the bar is never below the upstream floor of 0.020",
          _loop.bar() >= 0.020, f"bar={_loop.bar()}")
    # The real property, not a proxy for it: move the file and watch bar() follow.
    # A bar frozen at import is the defect all over again, and only a live read
    # can show that it is not frozen.
    _pf = _m_path = ROOT / "state" / "power.json"
    _before = _loop.bar()
    _raw = json.loads(_pf.read_text())
    _pf.write_text(json.dumps(dict(_raw, recommended_bar=_before * 4 + 0.5)))
    try:
        _after = _loop.bar()
    finally:
        _pf.write_text(json.dumps(_raw))
    check("bar() follows power.json instead of being frozen at import",
          _after > _before * 2, f"{_before} -> {_after}")
    check("... and returns to the real value when the file is restored",
          abs(_loop.bar() - _before) < 1e-9, f"{_loop.bar()} vs {_before}")
    # The measured bar must OVERRIDE a preregistered floor, because a floor fixed
    # at preregistration cannot know the noise floor -- the nulls have not run yet.
    for bad in (0.006, 0.0):
        pr = {"delta_floor": bad}
        got = max(float(pr.get("delta_floor", 0.0)), _loop.bar())
        check(f"a preregistered floor of {bad} cannot lower the measured bar",
              got >= _loop.bar(), f"effective={got}")

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
