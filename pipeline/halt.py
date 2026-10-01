"""The halt: authority, not advice.

Everything else in this system observes. This is the part that acts.

A monitor that reports a scientific violation at 3am has failed at its one job,
because the loop it is watching is still running and still publishing. A
diagnosis nobody reads by morning has cost a night's compute and, worse, a
version card that says the search was healthy. The gap between *detecting* a
violated protocol and *preventing* numbers from being produced on the strength of
it is the whole difference between a dashboard and a critical agent.

So: a critical in the science class writes a HALT, and the daemon refuses to
propose a new arm while one stands.

Three properties make this safe to leave armed, and each was a deliberate choice
against the obvious alternative:

1. **A halt blocks NEW work; it does not kill the arm in flight.** An arm
   mid-checkpoint is the most expensive thing to throw away, and stopping the
   search does not require stopping it. The arm finishes, is recorded, and the
   next proposal is refused.

2. **A halt does not clear itself.** A flapping invariant must not be able to
   resume a search by being briefly green at 3:01. Clearing requires a human to
   write a reason, and the reason is appended to an audit log — so "why was this
   allowed to continue" is always answerable after the fact.

3. **Operational failures do not halt.** A dead dashboard is broken, not wrong.
   Halting a search over a monitor outage would trade a real result for a
   cosmetic one, which inverts every priority this system is built on. Only the
   science class halts.

`status()` is the daemon's gate. It is deliberately cheap and deliberately
conservative: unknown or malformed halt state means HALTED, because a halt file
that cannot be read is not evidence that the search is safe to continue.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
HALT = Path(os.environ["RSIJEV_HALT"]) if os.environ.get("RSIJEV_HALT") else STATE / "HALT.json"
AUDIT = ROOT / "records" / "halt_audit.jsonl"

# The class that halts. Derived from the doctor's own verdict, never re-decided
# here, so there is exactly one place in the system that knows what "wrong" means.
HALTING_VERDICTS = {"wrong"}


@dataclass
class Halt:
    active: bool
    reason: str = ""
    since: float = 0.0
    since_h: str = ""
    findings: list[str] | None = None
    unreadable: bool = False

    @property
    def marker(self) -> Path:
        """Where the daemon records that it has already explained this halt.

        Derived from the halt file's own directory rather than from a fixed
        STATE path, so redirecting the halt path redirects everything that goes
        with it. A marker that lives somewhere else is shared state between two
        things that are supposed to be independent — the live pipeline and
        whatever is testing it.
        """
        return HALT.parent / "halt_logged"

    def to_json(self) -> dict:
        return {"active": self.active, "reason": self.reason, "since": self.since,
                "since_h": self.since_h, "findings": self.findings or [],
                "unreadable": self.unreadable}


def status() -> Halt:
    """Read the halt. Unknown state is treated as halted, never as clear."""
    if not HALT.is_file():
        return Halt(active=False)
    try:
        d = json.loads(HALT.read_text())
    except (json.JSONDecodeError, OSError):
        # A halt file that cannot be parsed is not a cleared halt. Anything else
        # turns a corrupt file into an unattended search resuming itself.
        return Halt(active=True, unreadable=True,
                    reason="HALT.json exists but cannot be parsed — refusing to "
                           "resume a search on the strength of a file nobody can read",
                    since=HALT.stat().st_mtime if HALT.is_file() else 0.0,
                    since_h=time.strftime("%Y-%m-%d %H:%M:%S"))
    if not d.get("active"):
        return Halt(active=False)
    return Halt(active=True, reason=str(d.get("reason", "")),
                since=float(d.get("since", 0.0)),
                since_h=str(d.get("since_h", "")),
                findings=list(d.get("findings", [])))


def raise_halt(reason: str, findings: list[str] | None = None) -> Halt:
    """Arm the halt. Idempotent: a standing halt keeps its original timestamp."""
    cur = status()
    if cur.active and not cur.unreadable:
        return cur
    h = Halt(active=True, reason=reason, since=time.time(),
             since_h=time.strftime("%Y-%m-%d %H:%M:%S"), findings=findings or [])
    _write(h)
    _audit("halted", reason, findings or [])
    return h


def clear_halt(who: str, why: str) -> Halt:
    """Release the halt. Requires a human, a name and a reason, and is logged."""
    if not why.strip():
        raise ValueError("clearing a halt requires a reason")
    prev = status()
    _audit("cleared", why, [f"by {who}"])
    h = Halt(active=False)
    _write(h)
    h.active = prev.active  # keep the prior state visible in the return value
    return h


def _write(h: Halt) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = HALT.with_suffix(".tmp")
    tmp.write_text(json.dumps(h.to_json(), indent=2, default=str) + "\n")
    os.replace(tmp, HALT)          # atomic: never leave a half-written halt


def _audit(action: str, why: str, findings: list[str]) -> None:
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT.open("a") as fh:
        fh.write(json.dumps({"ts": time.time(),
                             "when_h": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "action": action, "why": why,
                             "findings": findings}, default=str) + "\n")


def reconcile(verdict: str, ok: bool, reason: str,
              findings: list[str] | None = None) -> Halt:
    """The doctor's one call. Halts on the science class, never on operations."""
    if verdict in HALTING_VERDICTS:
        return raise_halt(reason, findings)
    return status()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear", metavar="WHY", help="release the halt (requires a reason)")
    ap.add_argument("--who", default="operator", help="who is clearing it")
    a = ap.parse_args()
    if a.clear:
        clear_halt(a.who, a.clear)
        print(f"  halt cleared by {a.who}: {a.clear}")
    else:
        h = status()
        print(f"  {'HALTED' if h.active else 'running'}")
        if h.active:
            print(f"  since {h.since_h}  {h.reason}")
            for f in h.findings or []:
                print(f"    {f}")
    raise SystemExit(0)
