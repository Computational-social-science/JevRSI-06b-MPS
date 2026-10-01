#!/usr/bin/env python3
"""Does the publisher REFUSE when it should?

`pipeline/publish.py` is the one component whose failure mode is silent in the
worst possible way. A publisher that always commits does not look broken: the
repository grows, the sync appears to work, and the violation it was supposed to
catch is already public. And a publisher that always refuses is equally
undetectable — it looks diligent.

So each gate gets its violation planted, and each must refuse. The three that
matter most:

  * a `wrong` doctor verdict blocks the commit — the whole point of the design;
  * a torn record blocks the commit, because the daemon appends with
    `fsync_append` while this runs and a partial line in the evidence is worse
    than no evidence;
  * a credential in the INDEX blocks the commit, because the working-tree scan in
    `tests/test_no_cost.py` and the index are different sets, and .gitignore is
    a policy rather than an enforcement.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True)


def main() -> int:
    import publish

    print("0. the repository is set up the way the publisher assumes")
    check("there is a git repository", (ROOT / ".git").is_dir())
    git("add", "-A")
    staged = git("diff", "--cached", "--name-only").stdout.split()
    check("something is staged to publish", len(staged) > 0, f"{len(staged)} paths")
    for big in (".venv", "ckpt", "corpus", "logs"):
        check(f"{big} is NOT staged", not any(s.startswith(big) for s in staged))
    check("no per-question rows are staged",
          not any(s.endswith(".items.jsonl") for s in staged))

    print("\n1. gate 2 — a torn record blocks the commit")
    # Plant a half-written line, exactly what a commit taken mid-fsync_append sees.
    arms = ROOT / "records" / "arms.jsonl"
    original = arms.read_text()
    try:
        arms.write_text(original + '{"arm": "half_written", "verdict": "kept", "pool')
        d = publish.Decision(may_publish=True)
        publish.gate_records_parse(d)
        check("a truncated line is caught", not d.may_publish, "; ".join(d.reasons)[:70])
        check("... and the reason names the file and line",
              any("arms.jsonl" in r for r in d.reasons), str(d.reasons)[:90])
        check("... and says it is mid-append",
              any("mid-append" in r for r in d.reasons), str(d.reasons)[:90])
    finally:
        arms.write_text(original)

    d = publish.Decision(may_publish=True)
    publish.gate_records_parse(d)
    check("the intact record passes", d.may_publish, "; ".join(d.reasons)[:60])

    print("\n2. gate 3 — a credential in the INDEX blocks the commit")
    poison = ROOT / "config" / "_publish_canary.json"
    try:
        poison.write_text('{"note": "ghp_abcdefghijklmnopqrstuvwxyz0123456789"}')
        git("add", "-A")
        d = publish.Decision(may_publish=True)
        ok = publish.gate_secrets(d)
        check("a staged credential is caught", not ok and not d.may_publish,
              "; ".join(d.reasons)[:70])
        check("... and the path is named",
              any("_publish_canary" in r for r in d.reasons), str(d.reasons)[:80])
    finally:
        poison.unlink(missing_ok=True)
        git("add", "-A")
    d = publish.Decision(may_publish=True)
    publish.gate_secrets(d)
    check("the clean index passes", d.may_publish, "; ".join(d.reasons)[:60])

    print("\n3. gate 1 — a wrong doctor verdict blocks the commit  ← the whole point")
    # Plant a real violation rather than stubbing the doctor: drop the champion
    # record, which is invariant I3, a critical in the provenance lens.
    #
    # `gate_science` calls the REAL doctor, and the doctor raises a real halt. So
    # the halt state must be redirected for the duration, or this test writes
    # "planted for the test" into the live `state/HALT.json` and into the live
    # audit trail — which is exactly what happened: the unattended daemon read
    # that reason nine times and refused to propose. A test that reaches the live
    # control plane is a test that can stop the search.
    import halt as _haltmod
    _saved_halt = (_haltmod.STATE, _haltmod.HALT, _haltmod.AUDIT)
    _htmp = Path(tempfile.mkdtemp(prefix="pubhalt-"))
    _haltmod.STATE = _htmp
    _haltmod.HALT = _htmp / "HALT.json"
    _haltmod.AUDIT = _htmp / "halt_audit.jsonl"

    champ = ROOT / "state" / "champion.json"
    original = champ.read_text() if champ.is_file() else None
    arms_backup = arms.read_text()
    try:
        # I3: a champion with no record. Corrupt the champion's arm name so it
        # names an arm that does not exist in arms.jsonl.
        champ.write_text(json.dumps({"arm": "ghost_arm_that_never_ran",
                                     "pooled_top1": 0.9, "control_top1": 0.46}))
        d = publish.Decision(may_publish=True)
        publish.gate_science(d)
        blocked = (not d.may_publish)
        check("a critical invariant blocks publication", blocked,
              f"verdict={d.verdict} reasons={d.reasons[:1]}")
        check("... and the refusal names the doctor verdict",
              "WRONG" in " ".join(d.reasons).upper() or d.verdict == "wrong",
              d.verdict)
        check("... and it says what the diagnosis was",
              any(len(r) > 40 for r in d.reasons), str(d.reasons)[:110])
    finally:
        if original is not None:
            champ.write_text(original)
        else:
            champ.unlink(missing_ok=True)
        arms.write_text(arms_backup)

    # A HEALTHY doctor is not sufficient to publish: a standing halt blocks on its
    # own. Asserting otherwise would be asserting something untrue — the live
    # pipeline currently IS healthy AND halted, and the correct answer is refuse.
    d = publish.Decision(may_publish=True)
    publish.gate_science(d)
    check("a healthy doctor alone is NOT enough while a halt stands",
          (not d.may_publish) and d.verdict == "healthy",
          f"verdict={d.verdict} {d.reasons[:1]}")
    _haltmod.HALT = _htmp / "HALT.json"          # empty: no halt
    _htmp.joinpath("HALT.json").unlink(missing_ok=True)
    d = publish.Decision(may_publish=True)
    publish.gate_science(d)
    healthy_ok = d.may_publish
    _haltmod.STATE, _haltmod.HALT, _haltmod.AUDIT = _saved_halt
    shutil.rmtree(_htmp, ignore_errors=True)

    check("with no halt standing, a healthy pipeline may publish", healthy_ok,
          f"verdict={d.verdict} {d.reasons[:1]}")
    check("... and the decision records what it checked",
          d.verdict and d.invariants and d.probes, f"{d.verdict} {d.invariants}")

    print("\n4. a standing halt blocks publication")
    import halt as haltmod
    saved = (haltmod.STATE, haltmod.HALT, haltmod.AUDIT)
    tmp = Path(tempfile.mkdtemp(prefix="pubhalt-"))
    try:
        haltmod.STATE = tmp
        haltmod.HALT = tmp / "HALT.json"
        haltmod.AUDIT = tmp / "halt_audit.jsonl"   # audit too, or it pollutes the real trail
        haltmod.raise_halt("planted for the test", ["provenance/I3"])
        d = publish.Decision(may_publish=True)
        publish.gate_science(d)
        check("a halt blocks publication", not d.may_publish, str(d.reasons)[:70])
        check("... and says the halt is standing",
              any("halt is standing" in r for r in d.reasons), str(d.reasons)[:80])
    finally:
        haltmod.STATE, haltmod.HALT, haltmod.AUDIT = saved
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n5. the commit message is an audit trail, not 'auto-sync'")
    d = publish.Decision(may_publish=True, verdict="healthy", invariants="24/24",
                         probes="5/5")
    msg = publish.message(d, ["null1", "null2"])
    check("the message carries the doctor verdict", "healthy" in msg, msg[:60])
    check("... the check counts", "24/24" in msg and "5/5" in msg)
    check("... and the arms published", "null1" in msg and "null2" in msg)
    check("... and it is not 'auto-sync'", "auto-sync" not in msg.lower())

    print("\n6. the evidence bundle carries a sha256 so A1 stays reproducible")
    arms_dir = ROOT / "records"
    arm = next((d_.name for d_ in sorted(arms_dir.iterdir())
                if d_.is_dir() and list(d_.glob("*.items.jsonl"))), None)
    if arm is None:
        check("an arm with raw rows exists to bundle", False)
    else:
        got = publish.cut_evidence_bundle(arm)
        check("a bundle is produced", got is not None, arm)
        if got:
            import tarfile
            path, sha = got
            check("the bundle is a real tarball", tarfile.is_tarfile(path))
            with tarfile.open(path) as tf:
                names = tf.getnames()
                man = [n for n in names if n.endswith("MANIFEST.json")]
                check("it contains the raw rows",
                      any(n.endswith(".items.jsonl") for n in names), str(names)[:70])
                check("... and a manifest", bool(man))
                if man:
                    m = json.loads(tf.extractfile(man[0]).read())
                    check("... with a sha256 of the rows",
                          len(m.get("items_sha256", "")) == 64, m.get("items_sha256", "")[:20])
                    check("... naming the arm", m.get("arm") == arm, m.get("arm", "?"))
            check("the bundle's own sha256 is reported", len(sha) == 64, sha[:20])
            path.unlink(missing_ok=True)

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
