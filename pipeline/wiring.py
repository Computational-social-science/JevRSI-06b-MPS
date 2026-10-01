"""The wiring: what produces what, who consumes it, and where that chain is broken.

## Why this exists

A 24/7 pipeline is judged on two questions a liveness check cannot answer:

  * **Is a link dangling?** — something is produced and nobody consumes it, so it
    is not evidence, it is decoration. A checker whose result is printed and
    discarded will pass today and say nothing tomorrow.
  * **Does a link need a human?** — the chain is intact but stalled, and no
    amount of waiting will move it.

Both are invisible from the outside, because an intact process producing a
discarded artifact looks exactly like a working one. The dashboard said
`healthy` for the whole of this writing, and two real dangling links existed the
entire time:

  * `formal/kernel_test.py` proves the gate's safety property and hands the result
    to the terminal. **No pipeline module reads it.** A proof that stopped
    compiling, or that started depending on `sorryAx`, would have gone unnoticed
    for as long as nobody happened to run it by hand.
  * `publish.py` cuts a per-arm evidence tarball carrying the raw per-question
    rows and their sha256, so that adversarial probe A1 stays reproducible by a
    reader. **Nothing uploads it.** The repository therefore ships numbers that
    nobody outside this machine can re-derive — which is the one thing a search
    claiming reproducibility cannot afford.

So the graph below is DECLARED, and then CHECKED. Declared because the wiring is a
design fact that grep cannot recover: a file nobody opens is still a link whose
output goes nowhere. Checked because a declared graph is a claim, and a claim
that is never tested is exactly what this module exists to catch.

Each link declares a producer, an artifact, a consumer set, and the cadence at
which the artifact is expected to change. A link is `intact`, `stale`, `dangling`
(produced, consumed by nobody), `orphaned` (consumed, produced by nobody) or
`absent` (the artifact does not exist at all). Only the first is healthy.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

INTACT, STALE, DANGLING, ORPHANED, ABSENT, IDLE = (
    "intact", "stale", "dangling", "orphaned", "absent", "idle")

# The pipeline's own cadence, in seconds. `None` means "per arm", which has no
# fixed interval and is judged against whether an arm is actually running.
PER_ARM = None


@dataclass
class Link:
    name: str
    stage: str
    producer: str
    artifact: str
    consumers: tuple[str, ...] = ()
    every: int | None = PER_ARM
    # A link whose artifact is only ever read by a person is a HUMAN link, and
    # that is not the same as a dangling one. The trajectory exists to be looked
    # at; that is its consumer.
    audience: str = "machine"
    note: str = ""
    status: str = INTACT
    age_s: float | None = None
    detail: str = ""


# The wiring, in the order the data flows. Declared once, checked every tick.
WIRING: tuple[Link, ...] = (
    # ---- the loop, one arm at a time
    Link("propose", "loop", "agenda.propose", "records/prereg.jsonl",
         ("daemon", "invariants.I1"), PER_ARM,
         note="a prediction must exist before GPU time is spent"),
    Link("train", "loop", "run_one.py", "records/<arm>/result.json",
         ("daemon.gate", "adversary.A1"), PER_ARM,
         note="one forward pass per question; the per-question rows land beside it"),
    Link("gate", "loop", "daemon.gate", "records/arms.jsonl",
         ("power", "confirm", "trajectory", "version_card", "adversary", "publish"),
         PER_ARM,
         note="the bar: champion + noise floor, decided here and nowhere else"),
    Link("floor", "loop", "power.analyse", "state/power.json",
         ("agenda.roles", "loop.gate", "invariants.I5", "adversary.A4"), PER_ARM,
         note="measured from null replicas; the bar cannot be set without it"),
    Link("confirm", "loop", "confirm.schedule", "state/pending_confirm.json",
         ("daemon.propose", "invariants.I7"), PER_ARM,
         note="outranks every hypothesis: an unconfirmed arm may not be champion"),
    Link("held_out", "loop", "held_out.spend", "state/held_out.json",
         ("invariants.I8", "dashboard"), PER_ARM,
         note="read once, on a crowned champion, never during the search"),
    Link("collect", "loop", "lineage.collect", "ckpt/<discarded>/",
         ("invariants.I13",), PER_ARM,
         note="a discarded offspring is a measurement; its bytes are not evidence"),

    # ---- the critical path, every 15 minutes
    Link("lenses", "critical", "doctor.diagnose", "state/doctor.json",
         ("publish", "watchdog", "dashboard"), 900,
         note="six lenses; a critical in the science class is never diluted"),
    Link("halt", "critical", "halt.reconcile", "state/HALT.json",
         ("daemon.propose",), 900,
         note="the only mechanism that can stop the search; never self-clears"),
    Link("audit", "critical", "halt._audit", "records/halt_audit.jsonl",
         ("publish",), None, audience="human",
         note="who released a halt, and why — answerable after the fact"),

    # ---- publication, every 30 minutes
    Link("publish", "publish", "publish.publish", "records/publish_decisions.jsonl",
         ("git",), 1800, audience="human",
         note="every decision recorded, including the refusals"),
    Link("remote", "publish", "git push",
         "https://github.com/Computational-social-science/JevRSI-06b-MPS", (), 1800,
         audience="human", note="gated by the same verdict that gates the loop"),

    # ---- evidence a reader can check
    Link("trajectory", "evidence", "trajectory.build", "state/trajectory.html",
         (), None, audience="human", note="the climb, and the discards on it"),
    Link("record", "evidence", "version_card.build", "versions/current.md",
         (), PER_ARM, audience="human",
         note="what is currently champion, and what was thrown away"),
    Link("rows", "evidence", "run_one.py", "records/*/**.items.jsonl",
         ("adversary.A1",), PER_ARM,
         note="the per-question rows every published number is recomputed from"),

    # ---- the formal layer. See the module docstring: this one was dangling.
    Link("kernel", "formal", "kernel_test.py", "state/claims/kernel.json",
         ("claims", "publish"), 3600,
         note="hourly, unattended; stamps the hash of the source it proved"),
)


def _age(path: Path) -> float | None:
    if not path.exists():
        return None
    return time.time() - path.stat().st_mtime


def _expand(artifact: str) -> Path | None:
    """Resolve a declared artifact to a path, tolerating globs and markers."""
    if artifact.startswith("http"):
        return None
    if "<" in artifact:
        # `records/<arm>/result.json` is a per-arm pattern, not a literal path.
        # Resolving it literally reported "does not exist" for every arm that had
        # ever run, which is a fault report about correct behaviour.
        head, _, tail = artifact.partition("<")
        prefix = head.rsplit("/", 1)[0]
        name = tail.split(">", 1)[1]
        hits = sorted((ROOT / prefix).glob(f"*/{name}")) if (ROOT / prefix).is_dir() else []
        return max(hits, key=lambda f: f.stat().st_mtime) if hits else None
    p = ROOT / artifact
    if "*" in artifact:
        hits = sorted(ROOT.glob(artifact))
        return max(hits, key=lambda f: f.stat().st_mtime) if hits else None
    return p


def _lean_installed() -> tuple[bool, str]:
    for c in (Path.home() / ".elan" / "toolchains", Path.home() / ".elan" / "bin"):
        if c.exists() and any(c.rglob("lean")):
            return True, str(next(c.rglob("lean")))
    return False, "no lean toolchain on this machine"


def check(in_flight: str | None = None) -> list[Link]:
    """Evaluate every declared link against what is actually on disk."""
    out: list[Link] = []
    for l in WIRING:
        if l.name == "kernel":
            p = ROOT / "state" / "claims" / "kernel.json"
            l.age_s = _age(p) if p.exists() else None
            if not p.exists():
                l.status, l.detail = ABSENT, "the check has never run"
            else:
                import claims as _claims
                sm = _claims.summary()
                n_bad = len(sm["unproven"])
                l.status = INTACT if sm["may_publish"] else DANGLING
                l.detail = (f"{len(sm['claims'])} claim(s) proven, hourly"
                             if sm["may_publish"]
                             else f"{n_bad} claim(s) unproven: {', '.join(sm['unproven'])}")
            out.append(l)
            continue

        if l.name == "confirm":
            p = ROOT / "state" / "pending_confirm.json"
            if p.is_file():
                l.age_s = _age(p)
                l.status, l.detail = INTACT, "a confirmation is owed and will run first"
            else:
                l.status, l.detail = IDLE, "nothing awaiting confirmation"
            out.append(l)
            continue

        if l.name == "remote":
            r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                               capture_output=True, text=True)
            sync = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "origin/main"],
                                  capture_output=True, text=True)
            l.age_s = 0.0
            if r.returncode == 0 and sync.returncode == 0:
                same = r.stdout.strip() == sync.stdout.strip()
                l.status = INTACT if same else STALE
                l.detail = "in sync with origin" if same else "local is ahead of origin"
            else:
                l.status, l.detail = ABSENT, "no remote configured"
            out.append(l)
            continue

        if l.name == "rows":
            dirs = [d for d in (ROOT / "records").iterdir()
                    if d.is_dir() and any(d.glob("*.items.jsonl"))] \
                if (ROOT / "records").is_dir() else []
            l.age_s = max((_age(d) or 0) for d in dirs) if dirs else None
            if not dirs:
                l.status, l.detail = ABSENT, "no arm has produced rows yet"
            else:
                l.status, l.detail = INTACT, f"{len(dirs)} arm(s) carry raw rows"
            out.append(l)
            continue

        if l.name == "collect":
            held = sum(f.stat().st_size for f in (ROOT / "ckpt").rglob("*")
                       if f.is_file()) if (ROOT / "ckpt").is_dir() else 0
            l.age_s = 0.0
            l.status = INTACT
            l.detail = f"{held/1e9:.2f} GB held — lineage only"
            out.append(l)
            continue

        if l.name == "halt":
            # HALT.json is only MEANINGFUL when it is active. A file left behind
            # by the last clear is the absence of a halt, not a stale halt — and
            # reporting it as STALE would train a reader to ignore the one panel
            # whose entire job is to say "stop".
            p = ROOT / "state" / "HALT.json"
            l.age_s = _age(p) if p.exists() else None
            if not p.exists():
                l.status, l.detail = IDLE, "no halt has ever been raised"
            else:
                try:
                    active = json.loads(p.read_text()).get("active", False)
                except (json.JSONDecodeError, OSError):
                    active = True      # unreadable means held; see halt.status()
                if active:
                    l.status = INTACT
                    l.detail = "A HALT IS STANDING — the loop will not propose"
                else:
                    l.status, l.detail = IDLE, "cleared; no halt in force"
            out.append(l)
            continue

        p = _expand(l.artifact)
        l.age_s = _age(p) if p else None
        if p is None or not p.exists():
            l.status = ABSENT
            l.detail = f"{l.artifact} does not exist"
        elif l.audience == "none":
            l.status = DANGLING
            l.detail = l.note
        else:
            l.status = INTACT
            n = len(l.consumers)
            l.detail = f"{n} consumer(s)" + ("" if n else " — human only")
            if l.every and l.age_s is not None:
                if l.age_s > l.every * 3:
                    l.status, l.detail = STALE, (
                        f"{l.age_s/60:.0f} min old, expected every "
                        f"{l.every/60:.0f} min")
        out.append(l)
    return out


def summary(links: list[Link]) -> dict:
    by: dict[str, int] = {}
    for l in links:
        by[l.status] = by.get(l.status, 0) + 1
    needs_human = [l for l in links if l.status in (DANGLING, ORPHANED, ABSENT)]
    return {
        "links": [l.__dict__ for l in links],
        "counts": by,
        "needs_human": [f"{l.name} ({l.status}): {l.detail}" for l in needs_human],
        # A stale link is a machine-fixable problem; a dangling one is a design
        # hole. Only the second is worth waking someone for.
        "human_required": [l.name for l in links if l.status in (DANGLING, ORPHANED)],
        "stages": sorted({l.stage for l in links}),
    }


def render() -> str:
    import doctor as doct
    import halt as haltmod
    inflight = None
    try:
        import invariants
        inflight = invariants.running_arm()
    except Exception:
        pass
    links = check(inflight)
    s = summary(links)
    L = ["  the wiring — what is connected, and what is not", ""]
    cur = None
    for l in links:
        if l.stage != cur:
            cur = l.stage
            L.append(f"  [{cur}]")
        mark = {INTACT: "  ok  ", STALE: " STALE", DANGLING: "DANGLE",
                ORPHANED: "ORPHAN", ABSENT: " ABSENT", IDLE: " idle "}[l.status]
        L.append(f"  {mark}  {l.name:11s} {l.producer:22s} → {l.artifact[:34]:34s} {l.detail[:44]}")
    L.append("")
    L.append("  " + "  ".join(f"{k} {v}" for k, v in sorted(s["counts"].items())))
    if s["human_required"]:
        L.append("")
        L.append("  needs a human — these are design holes, not faults:")
        for n in s["human_required"]:
            L.append(f"    {n}")
    return "\n".join(L)


if __name__ == "__main__":
    print(render())
    raise SystemExit(0)
