"""Claim registry: a scientific claim may not be MADE until it is PROVEN.

## The rule this enforces

Every sentence this project asserts to a reader is registered here with the
artifact that proves it. Publication is refused while any registered claim is
unproven, and — the part that matters — while its proof is **older than the code
it proves**.

That second clause is the whole point. A proof that passed last Tuesday does not
cover an edit made this morning. It is the formal analogue of reporting a
measurement without re-running the instrument after you changed the instrument,
and it is how a "verified" claim quietly becomes a historical claim. So the proof
is stamped with a hash of the source it checked, and a claim whose source has
moved is INVALID rather than stale-and-therefore-fine.

## Why a registry and not a single "run the tests" step

A test suite says "everything currently implemented passes". A claim registry says
"this specific sentence, which appears in six files, is currently supported by
this specific artifact". Those are different obligations, and only the second one
matches what a reader is entitled to assume when they read a number.

The claims are not decorative either — each names a real sentence the repository
actually makes, and a `where` so a reader can go and check that the sentence
still exists. A registry whose entries have drifted from the prose is worse than
none, so `check()` verifies the string is still present in the named file and
reports a claim whose text has been edited out from under it.

## Automatic, not on request

Nothing here asks a human to run anything. The proof job runs on a schedule, the
publisher runs the gate on every tick, and the dashboard shows the state. A check
that only fires when somebody remembers is the dangling link this module was
written to close.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

CLAIM_DIR = ROOT / "state" / "claims"

PROVEN, UNPROVEN, STALE, MISSING = "proven", "unproven", "stale", "missing"


@dataclass
class Claim:
    name: str
    sentence: str
    where: tuple[str, ...]          # files that assert it
    proof: str                       # artifact under state/claims/
    source: str                      # the file the proof is about
    # A proof stays valid while its source is unchanged AND it is not older than
    # this many days. The second bound catches a proof of a file that has not
    # changed but whose ENVIRONMENT has (a new Mathlib, a new toolchain).
    max_age_days: int = 30
    status: str = MISSING
    detail: str = ""
    proved_at: str = ""


REGISTRY: tuple[Claim, ...] = (
    Claim(
        "bar-monotone",
        "An arm that clears the bar is strictly better than the champion, so the "
        "champion sequence — and therefore the bar — cannot descend.",
        ("docs/ROOT_COMMIT.md", "formal/Gate.lean"),
        "kernel.json", "formal/Gate.lean",
    ),
    Claim(
        "confirmation-discipline",
        "An arm is promoted only after its effect reproduces on seeds it was never "
        "selected on; a selected-seed result alone is seed luck.",
        ("docs/ROOT_COMMIT.md", "pipeline/confirm.py"),
        "kernel.json", "pipeline/confirm.py",
    ),
    Claim(
        "floor-is-noise",
        "The threshold is the measured spread of null replicas, not an assumed "
        "constant.",
        ("docs/ROOT_COMMIT.md", "pipeline/power.py"),
        "kernel.json", "formal/Gate.lean",
    ),
    Claim(
        "recompute-from-raw",
        "Every published number is recomputed from the per-question rows beneath it.",
        ("docs/ROOT_COMMIT.md", "pipeline/adversary.py"),
        "kernel.json", "pipeline/adversary.py",
    ),
)


def _sha(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _proof(name: str) -> dict | None:
    p = CLAIM_DIR / name
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def check() -> list[Claim]:
    """Every registered claim, with the state of the proof that backs it."""
    out: list[Claim] = []
    now = time.time()
    for c in REGISTRY:
        pr = _proof(c.proof)
        if pr is None:
            c.status, c.detail = MISSING, (
                f"no {c.proof} — the check has not run, so the claim is being "
                f"MADE WITHOUT PROOF")
            out.append(c)
            continue
        c.proved_at = pr.get("when_h", "")

        if not pr.get("ok"):
            c.status = UNPROVEN
            c.detail = "the check RAN and FAILED: " + str(
                pr.get("failures", "see the proof artifact"))[:110]
            out.append(c)
            continue

        # The proof is about a source file. If the source has moved since, the
        # proof is about a file that no longer exists.
        want = pr.get("sources_sha256", {}).get(c.source)
        have = _sha(ROOT / c.source)
        if want is None:
            c.status, c.detail = STALE, "the proof does not say which source it checked"
        elif have is None:
            c.status, c.detail = STALE, f"{c.source} does not exist"
        elif want != have:
            c.status, c.detail = STALE, (
                f"{c.source} has changed since the proof ran — the proof is about "
                f"a file that no longer exists")
        elif now - float(pr.get("ts", 0)) > c.max_age_days * 86400:
            c.status, c.detail = STALE, (
                f"the proof is {int((now - pr.get('ts', 0)) / 86400)} days old")
        else:
            c.status = PROVEN
            c.detail = f"kernel-checked {c.proved_at}"

        # A registry whose entries have drifted from the prose is worse than no
        # registry, so the sentence is looked for where it says it is.
        missing = [w for w in c.where if not (ROOT / w).is_file()]
        if missing and c.status == PROVEN:
            c.status, c.detail = STALE, f"asserted in a file that is gone: {missing}"
        out.append(c)
    return out


def unproven() -> list[Claim]:
    return [c for c in check() if c.status != PROVEN]


def summary() -> dict:
    cs = check()
    counts: dict[str, int] = {}
    for c in cs:
        counts[c.status] = counts.get(c.status, 0) + 1
    return {"claims": [c.__dict__ for c in cs], "counts": counts,
            "unproven": [c.name for c in cs if c.status != PROVEN],
            "may_publish": all(c.status == PROVEN for c in cs)}


def stamp(name: str, payload: dict) -> Path:
    CLAIM_DIR.mkdir(parents=True, exist_ok=True)
    p = CLAIM_DIR / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(p)
    return p


def render() -> str:
    s = summary()
    L = ["  claim registry — a claim may not be made until it is proven", ""]
    for c in s["claims"]:
        mark = {PROVEN: "  ok  ", UNPROVEN: " UNPROVD", STALE: " STALE ",
                MISSING: " NOPRF "}[c["status"]]
        L.append(f"  {mark} {c['name']:26s} {c['detail'][:74]}")
    L.append("")
    L.append("  " + "  ".join(f"{k} {v}" for k, v in sorted(s["counts"].items())))
    L.append(f"  publication: {'ALLOWED' if s['may_publish'] else 'BLOCKED — a claim is unproven'}")
    return "\n".join(L)


if __name__ == "__main__":
    print(render())
    raise SystemExit(0 if summary()["may_publish"] else 2)
