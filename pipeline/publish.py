"""Publication is a scientific act, so it goes through the same discipline.

A `git push` on a cron is the obvious way to sync a 24/7 pipeline and it is
wrong in a specific, expensive way: **it publishes.** Every other guard in this
system exists to stop a number reaching the world before it has earned the right
to, and a cron push bypasses every one of them by publishing on schedule rather
than on merit. A pipeline that is `wrong` right now would have its violation
committed, its version card updated, and its trajectory rendered — all
automatically, all at 3am, all before the doctor finished forming a view.

So the publisher asks the critical agent for permission first, and refuses by
default. Five gates, each of which has a specific failure it prevents:

  1. NO PUBLISHING FROM A `wrong` VERDICT, OR WITH A HALT STANDING. The halt is
     the same halt the loop obeys. A repository is a more permanent claim than a
     log line, and a halted search that publishes anyway has the halt for show.

  2. NO TORN RECORDS. The daemon appends to `arms.jsonl` with `fsync_append`
     while this runs. A commit taken mid-append captures a partial line, and a
     partial line in the evidence is worse than no evidence — every invariant and
     every probe downstream reads that file. So every line of every append-only
     record must parse before the commit is made, not after.

  3. NO SECRETS. The cost guard scans the source tree; this scans what is about
     to be *staged*, which is a different set, because .gitignore is a policy
     and a policy is only as good as the file that enforces it.

  4. THE COMMIT MESSAGE IS THE AUDIT TRAIL. Every commit records the doctor's
     verdict, the count of invariants and probes that held, and which arms
     completed. `git log` then answers "what did this project believe, and was it
     healthy, on the day that number was published" — which is the question a
     reader of a self-improving system has every right to ask and which a bare
     `auto-sync` message can never answer.

  5. NO FORCE, NO REWRITE. The history is evidence. A pipeline that can rewrite
     its own past has the same problem as a pipeline that can rewrite its own
     evaluation.

The raw per-question rows do not go in git (see .gitignore for why) — they are
cut into a per-arm tarball with a sha256 and uploaded as a Release asset, which
is immutable and downloadable, so the adversarial probe remains reproducible by
a reader without the repository growing by a hundred megabytes.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

RECORDS = ROOT / "records"
STATE = ROOT / "state"

# The durable, COMMITTED record of every publication decision. `logs/` is
# gitignored — correctly, since a log is a debugging artifact — which means an
# unattended sync would leave no answer to "did it publish, and what did it
# decide" once the week is over. This file is the one that gets committed, so the
# refusal history travels with the repository: a reader can see that on some day
# the pipeline declined to publish, and why. A sync whose decisions vanish is an
# unaudited sync.
DECISIONS = RECORDS / "publish_decisions.jsonl"

# Append-only evidence. Every line of these must parse before a commit.
APPEND_ONLY = ("arms.jsonl", "prereg.jsonl", "champions.jsonl", "halt_audit.jsonl")

SECRET = re.compile(r"(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|"
                    r"github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16})")


@dataclass
class Decision:
    may_publish: bool
    reasons: list[str] = field(default_factory=list)   # why it refused
    notes: list[str] = field(default_factory=list)     # what it did
    staged: int = 0
    verdict: str = "?"
    invariants: str = "?"
    probes: str = "?"
    new_arms: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"may_publish": self.may_publish, "reasons": self.reasons,
                "notes": self.notes, "staged": self.staged, "verdict": self.verdict,
                "invariants": self.invariants, "probes": self.probes,
                "new_arms": self.new_arms, "when_h": time.strftime("%Y-%m-%d %H:%M:%S")}


def git(*args: str) -> tuple[int, str]:
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)


# ---------------------------------------------------------------- gate 1
def gate_science(d: Decision) -> None:
    """Refuse to publish from a wrong verdict, or with a halt standing."""
    import doctor
    import halt as haltmod

    rep = doctor.diagnose()
    d.verdict = rep.verdict
    lens = rep.auditors
    inv = lens.get("provenance", {}), lens.get("methodology", {})
    n_ok = sum(a.get("ok", 0) for a in lens.values())
    n_all = sum(a.get("n", 0) for a in lens.values())
    d.invariants = f"{n_ok}/{n_all} checks"
    d.probes = f"{lens.get('adversary', {}).get('ok', 0)}/{lens.get('adversary', {}).get('n', 0)} recomputed"

    h = haltmod.status()
    if rep.verdict == "wrong" or rep.criticals:
        d.reasons.append(
            f"doctor says {rep.verdict.upper()}: {rep.diagnosis[:140]}")
    if h.active:
        d.reasons.append(f"a halt is standing since {h.since_h}: {h.reason[:120]}")
    if d.reasons:
        # A gate that finds a violation must be able to veto on its own. Leaving
        # that to the caller means the correctness of publication depends on
        # remembering one line in `publish()` — and the day someone adds a second
        # caller, or reorders these, a `wrong` verdict gets committed.
        d.may_publish = False
    if not d.reasons:
        d.notes.append(f"doctor {rep.verdict}, {d.invariants}, adversary {d.probes}")


# ---------------------------------------------------------------- gate 2
def gate_records_parse(d: Decision) -> None:
    """Every line of every append-only record must parse.

    The daemon appends with `fsync_append` while this runs. A partial line here
    is not a cosmetic problem: every invariant and every adversarial probe reads
    these files, and a reader who clones the commit inherits the same truncated
    line. Better to publish nothing this tick than to publish a broken record.
    """
    for name in APPEND_ONLY:
        p = RECORDS / name
        if not p.is_file():
            continue
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                d.reasons.append(
                    f"{name}:{i} does not parse ({exc.msg}) — an arm is mid-append; "
                    f"refusing to publish a torn record")
                d.may_publish = False
                return
    d.notes.append(f"{len(APPEND_ONLY)} append-only record(s) parse cleanly")


# ---------------------------------------------------------------- gate 3
def gate_secrets(d: Decision) -> bool:
    """Scan what is about to be STAGED, not the source tree.

    `tests/test_no_cost.py` scans the working tree; this scans the index. They
    are different sets, and .gitignore is only a policy — a file that slips past
    it is invisible to the tree scan and one `git add -A` away from a push.
    """
    rc, out = git("diff", "--cached", "--name-only")
    if rc != 0:
        d.reasons.append(f"git diff --cached failed: {out.strip()[:120]}")
        d.may_publish = False
        return False
    # Re-read the index rather than trusting a name from another scope. The count
    # lives on `d` now, but the PATHS still have to come from somewhere, and
    # hoisting them into `publish()` left this function referring to a variable
    # that no longer existed here — a NameError on every healthy publish, which
    # is the worst possible moment for a publisher to fall over.
    staged = [f for f in out.split() if f.strip()]
    d.staged = len(staged)
    import detectors as _det
    for rel in staged:
        p = ROOT / rel
        if not p.is_file() or p.stat().st_size > 20_000_000:
            continue
        # Same exclusion list as every other scanner in the project. This was the
        # THIRD copy of the concept -- `detectors.py`, `invariants.py` and now
        # here -- and it immediately flagged this test file's planted fake token.
        # A credential pattern and a credential are different things, and a file
        # whose job is to prove the gate bites necessarily contains the former.
        if _det.is_detector(p.name):
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if SECRET.search(line):
                d.reasons.append(f"{rel}:{i} looks like a credential — refusing to stage it")
                d.may_publish = False
                return False
    d.notes.append(f"{d.staged} staged path(s), no credential literals")
    return True


# ---------------------------------------------------------------- evidence
def cut_evidence_bundle(arm: str) -> tuple[Path, str] | None:
    """Tar an arm's per-question rows with a sha256, for upload as a Release asset.

    The rows are what probe A1 recomputes every headline from. Not publishing them
    makes every number in the repository an unverifiable claim; publishing them in
    git makes every clone carry ~100 MB forever. A Release asset is immutable,
    versioned, and downloadable at a stable URL, which is exactly the property
    evidence needs and exactly what git history cannot provide.
    """
    d = RECORDS / arm
    if not d.is_dir():
        return None
    rows = next(iter(sorted(d.glob("*.items.jsonl"))), None)
    if rows is None:
        return None
    meta = {}
    for extra in ("rows.json", "result.json"):
        q = d / extra
        if q.is_file():
            try:
                meta[extra] = json.loads(q.read_text())
            except json.JSONDecodeError:
                meta[extra] = "unparseable"
    payload = json.dumps({
        "arm": arm,
        "items": rows.name,
        "items_sha256": _sha256(rows),
        "items_bytes": rows.stat().st_size,
        "note": "A1 recomputes this arm's per-target top-1 from these rows. "
                "Verify with: sha256sum and the check in tests/test_adversary.py.",
        "summary": meta,
    }, indent=2, default=str).encode()
    out = STATE / "evidence"
    out.mkdir(parents=True, exist_ok=True)
    tarpath = out / f"{arm}.evidence.tar.gz"
    with tarfile.open(tarpath, "w:gz") as tf:
        tf.add(rows, arcname=f"{arm}/{rows.name}")
        info = tarfile.TarInfo(f"{arm}/MANIFEST.json")
        info.size = len(payload)
        import io
        tf.addfile(info, io.BytesIO(payload))
    return tarpath, _sha256(tarpath)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- the message
def message(d: Decision, new_arms: list[str]) -> str:
    """The commit message IS the audit trail.

    `auto-sync` answers nothing. A reader of a self-improving system has every
    right to ask what the project believed and whether it was healthy on the day a
    number was published, and this is the only place that question is answerable
    after the fact.
    """
    head = (f"record: {len(new_arms)} arm(s) — doctor {d.verdict}, "
            f"{d.invariants}, adversary {d.probes}")
    if new_arms:
        head += "\n\narms:\n" + "\n".join(f"  {a}" for a in new_arms)
    return head + "\n\nPublished only after the critical agent cleared it; a halt " \
                  "or a wrong verdict blocks this commit.\n"


# ---------------------------------------------------------------- the run
def publish(dry_run: bool = False, push: bool = False) -> Decision:
    d = Decision(may_publish=True)
    git("add", "-A")
    rc, out = git("diff", "--cached", "--name-only")
    staged = [f for f in out.split() if f.strip()] if rc == 0 else []
    # Populate the count HERE, not inside `gate_secrets`. It used to be set there,
    # and `gate_secrets` is skipped whenever an earlier gate refuses — so every
    # refusal reported `staged = 0`, and the CLI's "nothing to publish" branch
    # swallowed it. The result was that a refusal exited 0, identically to a
    # successful no-op: the one signal meant to distinguish "held back" from
    # "nothing to do" was structurally incapable of firing.
    d.staged = len(staged)
    if not staged:
        d.notes.append("nothing staged; the working tree matches the last commit")
        d.may_publish = False
        d.reasons.append("(nothing to publish — this is not a refusal)")

    last = _last_published_arms()
    d.new_arms = _arms_now()
    fresh = [a for a in d.new_arms if a not in last]

    gate_science(d)
    gate_records_parse(d)
    if d.may_publish:
        gate_secrets(d)

    if d.reasons:
        d.may_publish = False
        record_decision(d)
        return d
    if dry_run:
        d.notes.append("dry run: nothing committed")
        record_decision(d)
        return d

    for arm in fresh:
        bundle = cut_evidence_bundle(arm)
        if bundle:
            path, sha = bundle
            d.notes.append(f"evidence bundle {path.name} sha256 {sha[:16]}…")

    # The root commit is not an incremental sync, and giving it a message that
    # says "record: 1 arm(s)" would misrepresent the repository's entire history
    # as an update to something that did not exist. So the first commit gets a
    # message that describes the project; every commit after it is governed by
    # `message()` and carries the audit fields.
    rc_has_head, _ = git("rev-parse", "--verify", "HEAD")
    if rc_has_head != 0:
        body = ROOT / "docs" / "ROOT_COMMIT.md"
        if body.is_file():
            # The root commit is the one exception: it is a real, substantive
            # publication, so its decision line belongs inside it. Every later run
            # rides its line along with whatever real change it finds.
            record_decision(d)
            d = git("commit", "-q", "-F", str(body))
            d.notes.append("root commit: message from docs/ROOT_COMMIT.md")
            return _finish(d, push)
        d.reasons.append("no HEAD and no docs/ROOT_COMMIT.md to describe the root "
                         "commit — refusing to invent one")
        d.may_publish = False
        record_decision(d)
        return d

    # Stage FIRST, then record. The staging snapshot is taken before this run's
    # decision line exists, so the log rides along with whatever REAL change the
    # run found and is committed by the NEXT run that has one.
    #
    # The alternative — recording before staging — makes the log self-sustaining:
    # every run appends a line, the tree is therefore never clean, so every run
    # commits, so every run appends. That is 48 commits a day, each one line, each
    # saying nothing, and it buries the commits that matter. An audit log must not
    # be able to manufacture the events it exists to record.
    rc, _ = git("add", "-A")
    record_decision(d)
    rc, out = git("commit", "-m", message(d, fresh))
    if rc != 0 and "nothing to commit" not in out:
        d.reasons.append(f"git commit failed: {out.strip()[:160]}")
        d.may_publish = False
        record_decision(d)
        return d
    d.notes.append(f"committed {d.staged} path(s)")

    return _finish(d, push)


def _finish(d: Decision, push: bool) -> Decision:
    if push:
        rc, out = git("push", "origin", "HEAD")
        if rc != 0:
            d.reasons.append(f"push FAILED after a successful commit: {out.strip()[:200]}")
            d.may_publish = False
        else:
            d.notes.append("pushed")
    return d


def _arms_now() -> list[str]:
    p = RECORDS / "arms.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line).get("arm", "?"))
            except json.JSONDecodeError:
                pass
    return out


def _last_published_arms() -> set[str]:
    rc, out = git("show", "HEAD:records/arms.jsonl")
    if rc != 0:
        return set()
    names = set()
    for line in out.splitlines():
        if line.strip():
            try:
                names.add(json.loads(line).get("arm", "?"))
            except json.JSONDecodeError:
                pass
    return names


def record_decision(d: Decision) -> None:
    """Append the decision to the committed log. Idempotent, append-only, small.

    One line per run. Written whether the run published or refused, because the
    refusals are the interesting half: they are the record of the critical agent
    stopping the pipeline from saying something it should not have said.
    """
    try:
        DECISIONS.parent.mkdir(parents=True, exist_ok=True)
        with DECISIONS.open("a") as fh:
            fh.write(json.dumps(d.to_json(), default=str) + "\n")
    except OSError as exc:
        # Never let the audit write stop the publish. A publisher that cannot
        # record its decision should still be safe to run; the log is a
        # convenience, the gate is the control.
        d.notes.append(f"could not append to {DECISIONS.name}: {exc}")


def render(d: Decision) -> str:
    L = []
    w = L.append
    tag = "PUBLISH" if d.may_publish else "REFUSE"
    w(f"  [{tag}]  doctor {d.verdict} · {d.invariants} · adversary {d.probes}")
    for r in d.reasons:
        w(f"    REFUSED: {r}")
    for n in d.notes:
        w(f"    {n}")
    if d.new_arms:
        w(f"    arms: {len(d.new_arms)} total, "
          f"{'new: ' + ', '.join(d.new_arms) if d.new_arms else 'none new'}")
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true",
                    help="for unattended runs: no stdout unless something happened")
    a = ap.parse_args()
    if not (ROOT / ".git").is_dir():
        print("  not a git repository — `git init` first")
        raise SystemExit(1)
    # `publish()` records the decision itself on every path, including the ones
    # that refuse — the refusals are the half worth keeping. Calling
    # `record_decision` again here would double every line.
    r = publish(dry_run=a.dry_run, push=a.push)
    if not (a.quiet and r.may_publish and not r.reasons):
        print(render(r))
    if a.json:
        print(json.dumps(r.to_json(), indent=2))
    # Three outcomes, three exit codes, so a launchd log or a monitor can tell
    # them apart without parsing prose:
    #
    #   0  published, or there was genuinely nothing to publish
    #   2  REFUSED — the gate stopped it and something WAS waiting
    #   3  error — the publisher itself could not decide
    #
    # The previous version collapsed "refused" into "nothing to do" whenever the
    # staged set happened to be empty, so a refusal during a quiet stretch exited
    # 0 and looked identical to a successful no-op. An exit code that cannot
    # distinguish the two is not a signal.
    if not r.staged:
        raise SystemExit(0)
    if r.reasons and not r.may_publish:
        raise SystemExit(2)
    if r.may_publish:
        raise SystemExit(0)
    raise SystemExit(3)



