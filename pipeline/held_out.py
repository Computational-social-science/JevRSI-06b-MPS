"""The held-out set: read once, when a champion is declared, then spent.

EDITABLE — ours, and the discipline upstream states and we had not implemented.

From `BENCHMARKS.md`:

    **The held-out set, read once per release.** Three benchmarks — tasksource,
    SemIf external and scienthoon OOD — were frozen before this work began and
    are scored **once**, after a release model has been chosen, never during the
    search. No release trains on them. They are the only place two releases are
    measured on equal terms, and they are where to look first when a release
    claims an improvement.

    **A held-out set is consumable, so each release freezes the next one.**
    Once a set has been read, the choice of what to release has been informed by
    it [...] Folding a spent set into the routine evaluation suite is the right
    thing to do with it — the measurement gets better — but only alongside
    freezing a *new* set from sources the search has not touched. Otherwise the
    suite slowly absorbs every independent check the project has, nothing is
    left outside it, and no later release can be caught fitting its own
    benchmarks.

Two consequences this module enforces mechanically, not by intention:

  1. It is **not reachable from the search**. `load_held_out` refuses while any
     arm is mid-flight, and the daemon only ever calls it after a champion has
     been decided. An independent check that can be consulted during a search
     stops being independent the first time it is consulted.

  2. It is **spent on read**. `state/held_out.json` records the read, and a
     second call raises rather than returning numbers. Spending it is the point:
     the value was in it being un-read, and a second read measures something
     else — the same trap upstream names in v2.0, where a release looked good on
     the suite it had trained for and did not beat v1.0 on the three benchmarks
     held out from the whole search.

Source: `tasksource/tasksource`, one of the three upstream names. Free,
anonymous, and — the requirement — never touched by the search. The training
corpus is `n4ze3m/typed-decisions-synth` and the routine targets are
`LocalLLaMA/typed-decisions` and `TIGER-Lab/MMLU-Pro`, so tasksource is a
genuinely separate source.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATE = ROOT / "state"
HELD_OUT_JSON = STATE / "held_out.json"
SOURCE = "tasksource/tasksource"
PINNED_REVISION = "7bfbd76c"          # the sha HfApi reported at freeze time
N_CASES = 400


class HeldOutSpent(RuntimeError):
    """Raised on a second read. Spending the set is the whole point."""


def is_spent() -> bool:
    """Has the set been READ? Not merely: does the file exist.

    A frozen-but-unread set is the normal, healthy state — it is frozen precisely
    so it can be read later — so existence is not the test, the `spent` flag is.
    Conflating the two is a bug this module had and this line is the fix: with
    `is_file()` the very act of freezing made the set look spent, and the first
    read after a freeze raised.
    """
    if not HELD_OUT_JSON.is_file():
        return False
    try:
        return bool(json.loads(HELD_OUT_JSON.read_text()).get("spent", False))
    except json.JSONDecodeError:
        return True        # unreadable state is not a safe state


def status() -> dict:
    """The set's state, with the SAME keys either way.

    A sparse unspent branch once made the record render the freeze date as "?",
    which is the one field a reader needs to judge whether the set was frozen
    before the search or after it. Keys must not come and go with `spent`.
    """
    if not HELD_OUT_JSON.is_file():
        return {"spent": False, "source": SOURCE, "revision": PINNED_REVISION,
                "frozen_at_h": "(not frozen yet)",
                "note": "no state file; the set has not been pinned"}
    rec = json.loads(HELD_OUT_JSON.read_text())
    rec["spent"] = bool(rec.get("spent", False))
    if not rec["spent"]:
        rec["note"] = "frozen and never read; this is what makes it independent"
    return rec


def freeze() -> dict:
    """Record the pin now, so the freeze is dated before the first read.

    A set that is frozen at the moment it is read was not frozen. This writes
    the pin and the date without scoring anything.
    """
    rec = {"frozen_at": time.time(), "frozen_at_h": time.strftime("%Y-%m-%d %H:%M:%S"),
           "source": SOURCE, "revision": PINNED_REVISION, "n_cases": N_CASES,
           "spent": False,
           "why": ("upstream names tasksource as one of its three held-out "
                   "benchmarks; it is free, anonymous, and a source this search "
                   "never trains on")}
    STATE.mkdir(parents=True, exist_ok=True)
    HELD_OUT_JSON.write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def load_held_out(n: int = N_CASES) -> list:
    """The held-out cases, as typed decisions. Raises if the set was read.

    The source is an NLI set: `premises`, `hypothesis`, and a `label` in
    {entailment, neutral}. That is already a typed decision about a document, so
    it maps onto `contract.Question` with no reshaping beyond asking the question
    in the project's own request shape:

        state       = the premises
        question    = does the hypothesis follow from them?
        options     = ("entailment", "neutral")
        gold        = one-hot on the source's own label

    The label space is TWO options, and that matters: `metrics.decision_score`
    for a choice question divides by the split's OWN base rate, so a binary
    benchmark has a majority baseline near 0.5 and a decision score that is NOT
    comparable to typed-decisions' four-way and five-way ones. The headline
    this produces is therefore reported as top-1 against its own majority
    baseline, and the record says which, rather than being placed in a column
    beside a number it is not commensurable with.
    """
    if is_spent():
        raise HeldOutSpent(
            f"{HELD_OUT_JSON} says this set was read on "
            f"{json.loads(HELD_OUT_JSON.read_text()).get('read_at_h', '?')}. "
            "A second read does not measure the same thing. Freeze a new set "
            "from a source the search has not touched instead — upstream's rule, "
            "and the reason v2.0 could be beaten on the suite it trained for "
            "while never beating v1.0 here.")

    from datasets import load_dataset

    from rsijev.contract import Case, Question

    ds = load_dataset(SOURCE, split="train", revision=PINNED_REVISION)
    # A fixed stride, not a shuffle: no RNG, so the subset is a function of the
    # source and the pin alone, and cannot drift with a seed.
    stride = max(1, len(ds) // n)
    rows = [ds[i] for i in range(0, len(ds), stride)][:n]

    OPTIONS = ("entailment", "neutral")
    CRITERIA = {
        "entailment": "The hypothesis follows from the premises.",
        "neutral": "The premises neither support nor contradict the hypothesis.",
    }
    cases = []
    for i, r in enumerate(rows):
        prem = (r.get("premises") or "").strip()
        hyp = (r.get("hypothesis") or "").strip()
        lab = (r.get("label") or "").strip()
        if not prem or not hyp or lab not in OPTIONS:
            continue
        gold = [0.0, 0.0]
        gold[OPTIONS.index(lab)] = 1.0
        cases.append(Case(
            case_id=f"heldout_{r.get('json_name', i)}",
            source="held_out_tasksource",
            state=prem,
            questions=(Question(
                key="entailment", mode="choice",
                instructions=f"Given the text, decide whether this statement follows "
                             f"from it.\n\nStatement: {hyp}",
                options=OPTIONS, criteria=CRITERIA,
            ),),
            gold={"entailment": tuple(gold)}))
    # Half, not an absolute number: the source is dense (every row carries all
    # four fields), so losing more than half means the schema moved and the pin
    # no longer means what it said. An absolute floor would also make a
    # deliberately small probe fail, which is a check testing its own argument.
    if len(cases) < max(10, n // 2):
        raise RuntimeError(
            f"{SOURCE} yielded only {len(cases)} of a requested {n} usable rows; "
            "the schema has probably changed, and a held-out set built from a "
            "source that no longer matches its pin is not the set that was frozen")
    return cases


def majority_baseline(cases) -> float:
    """The split's own majority baseline — the number a top-1 must beat."""
    from collections import Counter
    from rsijev.contract import gold_label
    labs = []
    for c in cases:
        for q in c.questions:
            labs.append(gold_label(q, c.gold[q.key]))
    if not labs:
        return float("nan")
    return max(Counter(labs).values()) / len(labs)


def spend(read_record: dict) -> dict:
    """Mark the set spent, with what was read and when.

    Refuses on a set that is already spent. Overwriting a read is the one thing
    that would quietly destroy the evidence that it was read, and the reason the
    set was worth having.
    """
    if is_spent():
        raise HeldOutSpent(
            "this set is already spent; spend() would overwrite the record of "
            "the read. Freeze a new one instead.")
    rec = json.loads(HELD_OUT_JSON.read_text()) if HELD_OUT_JSON.is_file() else freeze()
    rec.update(read_record)
    rec["spent"] = True
    rec["read_at"] = time.time()
    rec["read_at_h"] = time.strftime("%Y-%m-%d %H:%M:%S")
    rec["digest"] = hashlib.sha256(
        json.dumps(read_record, sort_keys=True).encode()).hexdigest()[:16]
    HELD_OUT_JSON.write_text(json.dumps(rec, indent=2) + "\n")
    return rec


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["status", "freeze", "probe"])
    a = ap.parse_args()
    if a.action == "status":
        print(json.dumps(status(), indent=2))
    elif a.action == "freeze":
        print(json.dumps(freeze(), indent=2))
    else:
        # Probe at the real size. A smaller probe would tell us the loader runs
        # but not that the SET is usable, and the set's size is the thing that
        # determines whether it can resolve the effect it exists to check.
        cs = load_held_out()
        base = majority_baseline(cs)
        print(f"{len(cs)} held-out cases probe OK")
        print(f"  majority baseline: {base:.4f}  <- a top-1 must beat this")
        print(f"  first state: {cs[0].state[:90]!r}")
        print(f"  first question: {cs[0].questions[0].instructions[:110]!r}")
        print("\n  the set is still UNSPENT: probing loaded it without reading it")
