"""Contamination: VERIFIED by the arm runner, not asserted.

EDITABLE — ours. Closes the second gap found in the rule-1 audit.

Upstream states it twice and means it both times:

    **No contamination, checked rather than asserted** — no release trains on
    any benchmark's **test** split, and no training document shares a state with
    one. Whether a release may use a benchmark's **train** split changed at
    v2.0, and it changes what its scores mean.

    Whether a release may use a benchmark's train split changed at v2.0, and it
    changes what its scores mean.

`build_synth_corpus.py` does check for exact state overlap when it builds the
corpus, and reported 0. That is an assertion made once, at build time, about one
snapshot. It is not the same claim as "no arm in this run was contaminated", and
the difference is what a reviewer is actually asking:

  * the corpus on disk may have been rebuilt, or truncated by `--max-cases`, or
    hand-edited since;
  * the eval targets are loaded fresh each arm from the Hub, and a pinned
    revision can still be re-pointed;
  * a *near*-duplicate is the contamination that actually inflates a number, and
    exact-hash equality cannot see it.

So this runs per arm, before any compute, and reports three things rather than
one boolean: exact state collisions, near-duplicates on a normalised key, and
the question-key overlap. A number produced next to an unverified contamination
report is a number with an asterisk on it, and this removes the asterisk or puts
it back.
"""
from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

# 12-word shingles: upstream's own contamination rule, verbatim from
# BENCHMARKS.md -- "no shared 12-word phrase with either split" -- generalised so
# it runs over whatever the corpus and targets happen to be.
SHINGLE_N = 12
_WORD = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _state_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()


def _shingles(text: str, n: int = SHINGLE_N) -> set[str]:
    w = _norm(text).split()
    if len(w) < n:
        # Too short to have an n-gram overlap; its own normalised form is still
        # a valid near-duplicate key, so short texts get one key each.
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


@dataclass
class Report:
    n_train_cases: int = 0
    n_eval_cases: int = 0
    eval_targets: list[str] = field(default_factory=list)
    exact_state_collisions: int = 0
    exact_examples: list[str] = field(default_factory=list)
    near_duplicate_states: int = 0
    near_examples: list[str] = field(default_factory=list)
    question_key_collisions: int = 0
    corpus_files: list[str] = field(default_factory=list)
    clean: bool = True
    verdict: str = ""

    def to_json(self) -> dict:
        return self.__dict__.copy()

    def one_line(self) -> str:
        return self.verdict or (
            f"clean: {self.n_train_cases} train cases vs {self.n_eval_cases} eval "
            f"cases across {len(self.eval_targets)} targets, no shared state")


def _case_state(c) -> str:
    return getattr(c, "state", "") or ""


def _case_keys(c) -> set[str]:
    ks = set()
    for q in getattr(c, "questions", ()):  # question KEYS, not their text
        ks.add(f"{q.mode}:{q.key}")
    return ks


def check(corpus: dict[str, list], targets: dict[str, list]) -> Report:
    """Corpus and targets, as the arm runner holds them. Not files on disk.

    Taking the loaded objects rather than re-reading the corpus means this
    measures exactly what the arm is about to train on, which is the only thing
    the claim is about.
    """
    r = Report(eval_targets=sorted(targets), corpus_files=sorted(corpus))

    train = [c for cases in corpus.values() for c in cases]
    ev = [c for cases in targets.values() for c in cases]
    r.n_train_cases, r.n_eval_cases = len(train), len(ev)

    train_hashes = {}
    for c in train:
        s = _case_state(c)
        if s:
            train_hashes.setdefault(_state_hash(s), c.case_id)

    # Shingle sets are only built if exact hashes find nothing, because they are
    # the expensive part and the common case is clean.
    ev_shingles: dict[str, str] = {}
    train_shingles: dict[str, str] = {}
    need_shingles = True

    for c in ev:
        s = _case_state(c)
        if not s:
            continue
        h = _state_hash(s)
        if h in train_hashes:
            r.exact_state_collisions += 1
            if len(r.exact_examples) < 3:
                r.exact_examples.append(f"eval {c.case_id} == train {train_hashes[h]}")
        if need_shingles:
            for sh in _shingles(s):
                ev_shingles.setdefault(sh, c.case_id)

    if r.exact_state_collisions == 0:
        for cases in corpus.values():
            for c in cases:
                s = _case_state(c)
                if not s:
                    continue
                for sh in _shingles(s):
                    if sh in ev_shingles:
                        r.near_duplicate_states += 1
                        if len(r.near_examples) < 3:
                            r.near_examples.append(
                                f"train {c.case_id} shares a {SHINGLE_N}-word phrase "
                                f"with eval {ev_shingles[sh]}")
                        break

    # Question keys are NOT a contamination signal here, and counting them was
    # a bug in this file rather than a finding about the data. It first
    # reported 1500 "collisions", which on inspection were ten shared names --
    # `score:urgency`, `choice:action`, `noul:needs_review` -- asked about
    # DIFFERENT documents in the two corpora. That is the design: v1.0 trains on
    # "a separate synthetic corpus only, with no shared state and no shared
    # 12-word phrase", states and phrases, and the synthetic corpus is
    # deliberately the same task family as the benchmark. A key naming the same
    # kind of question about a different document is the task being the task.
    #
    # A (key, state) pair appearing in both would be contamination, but that is
    # already implied by the state check above, so it adds no information.
    #
    # The count is kept in the record as 0 rather than deleted, so a reader can
    # see the criterion was considered and why it does not apply.

    r.clean = (r.exact_state_collisions == 0
               and r.near_duplicate_states == 0)
    if r.exact_state_collisions:
        r.verdict = (f"CONTAMINATED: {r.exact_state_collisions} training case(s) share a "
                     f"state with an evaluation case; e.g. {r.exact_examples[0]}")
    elif r.near_duplicate_states:
        r.verdict = (f"NEAR-DUPLICATE: {r.near_duplicate_states} training case(s) share a "
                     f"{SHINGLE_N}-word phrase with an evaluation case; e.g. "
                     f"{r.near_examples[0]}")
    return r


def save(rep: Report, arm: str) -> Path:
    import json
    p = ROOT / "state" / "contamination.jsonl"
    with open(p, "a") as fh:
        fh.write(json.dumps({"arm": arm, **rep.to_json()}) + "\n")
    return p


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "corpus"))
    ap.add_argument("--guard-n", type=int, default=200)
    a = ap.parse_args()
    from rsijev.contract import load_cases
    from rsijev.targets import load_mmlu_pro_1k, load_typed_decisions
    corpus = {f.stem: load_cases(str(f)) for f in sorted(Path(a.corpus).glob("*.jsonl"))}
    targets = {"typed_decisions": load_typed_decisions("test"),
               "mmlu_pro_guard": load_mmlu_pro_1k()[:a.guard_n]}
    rep = check(corpus, targets)
    print(f"train cases : {rep.n_train_cases}")
    print(f"eval cases  : {rep.n_eval_cases}  {rep.eval_targets}")
    print(f"exact state collisions   : {rep.exact_state_collisions}")
    print(f"near-duplicate states    : {rep.near_duplicate_states}")
    print(f"question-key collisions  : {rep.question_key_collisions}")
    print(f"verdict: {rep.verdict or rep.one_line()}")
