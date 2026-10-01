"""The one list of files that are allowed to contain what they look for.

This exists because the problem has now appeared four separate times in this
project, and each time it was fixed by adding a name to a second list that
somebody would then forget to update:

  1. the cost guard matched the word "replicate" in a comment about null arms and
     reported a paid inference vendor;
  2. the snapshot emitted bare `Infinity` and stopped being valid JSON;
  3. invariant I12 flagged `invariants.py`'s own cost-scan line;
  4. invariant I12 flagged `publish.py`'s own credential regex.

The common shape: a checker fires on the text of the thing that checks for that
text. The consequence is always the same and always bad — **a guard that cries
wolf on the healthy case is worse than no guard**, because it is the fastest way
to teach a reader to ignore the panel, and the panel is what has to be believed at
3am.

Two copies of this list is one copy too many. `invariants.py` and
`tests/test_no_cost.py` both import from here, so adding a detector is a
one-line change that cannot drift.

A file belongs here for exactly one reason: **it contains the pattern it searches
for, as a literal, because detecting that pattern is its job.** Nothing else
qualifies — in particular, a file that merely *mentions* a provider in prose does
not belong, and adding it here to silence a real finding is how this list turns
into a way of making the guard agree with the code.
"""

from __future__ import annotations

#: Names only, not paths: both scanners walk different trees but see the same
#: basenames, and a path here would silently fail to match in the other.
DETECTORS: frozenset[str] = frozenset({
    # The cost guard itself, and the invariant that re-implements it.
    "test_no_cost.py",
    "invariants.py",
    "test_invariants.py",
    # The contamination guard, and the test that plants violations as fixtures.
    "contamination.py",
    "test_contamination.py",
    # The publication gate, and the test that stages a fake token to prove the
    # gate bites. `publish.py` earns its place the same way `invariants.py` does:
    # the credential pattern is a compiled regex in its source.
    "publish.py",
    "test_publish.py",
    # The conventions checker, and the test that plants a Chinese docstring and
    # a machine path to prove the checker bites. Fifth occurrence of this class
    # in this project; the reason it is in this file rather than in a local
    # exemption in the test is that the previous four fixes each put it in a
    # different place and each was forgotten by the next detector.
    "test_conventions.py",
    # The multi-lens doctor imports the cost scanner's subject matter.
    "doctor.py",
})


def is_detector(name: str) -> bool:
    """Is this basename allowed to contain the patterns it searches for?"""
    from pathlib import Path
    return Path(name).name in DETECTORS
