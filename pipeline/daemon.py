#!/usr/bin/env python3
"""The 24/7 driver: one arm per invocation, resumable, launchd-driven.

EDITABLE — ours.

Design decisions and the reason for each:

  * ONE ARM PER PROCESS. A long-lived worker that holds the backbone in memory
    would be faster, but an arm that segfaults, hits an OOM, or diverges takes the
    worker with it and the trajectory loses its clock. A process per arm costs a
    model load (seconds at 0.6B) and buys isolation plus a clean exit code, which
    is what launchd and `StartInterval` actually need.

  * PREREGISTER BEFORE RUNNING, always. `prereg.jsonl` is appended and fsynced
    before the subprocess starts. A proposal written after seeing the result is
    not a prediction, and the whole value of this loop is that its negatives are
    usable.

  * THE CHAMPION MOVES ONLY ON A KEPT ARM. `champion.json` is the current bar;
    `champions.jsonl` is its history, so a later reader can see the bar rise
    rather than only its final value.

  * A FAILED ARM IS A RECORD, not a crash. Upstream registers an out-of-the-
    ordinary outcome; here the exception is caught, written to arms.jsonl with
    verdict="error", and the loop moves on. An unstable recipe is a finding.

  * A REPAIR IS EARNED, NOT GIVEN. An arm that clears the bar but fails exactly
    one guard does not die: it gets a diagnosis and one repair attempt, because
    upstream's v2.1 exists because someone refused to drop such an arm.

Exit codes: 0 = an arm ran (kept, rejected, needs_repair or error — all are
successful loop iterations); 1 = the loop could not start (no corpus, no
backbone). launchd retries on 1 and backs off on 0, which is the behaviour we
want: a broken setup should be retried loudly, a completed arm should not be.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "pipeline"))

import dev as devmod                      # noqa: E402
# `Hypothesis` is needed by the pending-confirmation branch below. It was missing,
# and the branch only runs when the loop owes a confirmation — so the pipeline
# worked on every path except the one that decides whether an arm above the bar is
# real, then died with `NameError: name 'Hypothesis' is not defined`. The crash
# was invisible: launchd logged it to a file and restarted, and the one-arm-per-
# run shape of the job made "running" and "crashing" look identical from outside.
from agenda import Hypothesis, next_prereg, propose  # noqa: E402
from loop import (CKPT, LOGS, RECORDS, STATE, ArmResult,  # noqa: E402
                  append_prereg, append_record, gate, read_records)
import confirm as confirmmod   # noqa: E402
import held_out                 # noqa: E402
import verify_ckpt              # noqa: E402

# The run configuration. Read once, and every value it carries is written into
# the arm's record, so a number can always be traced to the configuration that
# produced it. config/RATIONALE.md says why each value has the value it has.
CONFIG_PATH = ROOT / "config" / "run.json"
CONFIG = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.is_file() else {}


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGS / "daemon.log", "a") as fh:
        fh.write(line + "\n")


def fsync_append(path: Path, row: dict) -> None:
    """Append and force to disk. A prediction that is not on disk was not made."""
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "RSIJEV_MODEL", "Qwen/Qwen3-0.6B-Base"))
    ap.add_argument("--corpus", default=str(ROOT / "corpus"))
    ap.add_argument("--arm", default="", help="run this named arm, not the next one")
    ap.add_argument("--save-ckpt", action="store_true",
                    help="write weights for the arm (only the champion needs them)")
    ap.add_argument("--dry-run", action="store_true", help="propose, preregister, stop")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="override the step count, for a fast first pass")
    ap.add_argument("--batch-size", type=int, default=0)
    a = ap.parse_args()

    corpus = Path(a.corpus)
    if not corpus.is_dir() or not list(corpus.glob("*.jsonl")):
        log(f"NO CORPUS in {corpus} — run scripts/build_replication_corpus.py first")
        return 1

    # 0. THE HALT. Checked before anything else, because a critical agent that
    #    detects a violated protocol and then lets the loop propose its next arm
    #    is a report, not an agent. The in-flight arm is deliberately NOT killed:
    #    stopping the search does not require discarding the most expensive thing
    #    on the machine, and the arm's result is still worth having on the record.
    import halt as haltmod
    h = haltmod.status()
    if h.active:
        # launchd retries every ThrottleInterval, which is what makes the search
        # resume on its own once a human clears the halt. That also means this
        # branch runs every two minutes for as long as the halt stands, so the
        # full explanation is printed once and afterwards reduced to one line. A
        # halt that fills the log with its own reason is a halt nobody reads.
        sig = f"{h.since_h}|{h.reason}"
        # The marker lives beside the halt file, NOT in the fixed STATE dir. Two
        # reasons, and the second one is why this was flaky: (1) a test that
        # redirects the halt path with RSIJEV_HALT was still sharing the live
        # pipeline's marker, so a test could suppress the live loop's first
        # message; (2) `since_h` has one-second resolution, so two runs inside
        # the same second produced an identical signature and the second one
        # printed the short form — which omits how to clear it — and the test
        # asserting on that string failed intermittently. A shared mutable marker
        # keyed on a coarse timestamp is a race with extra steps.
        seen = h.marker
        if not seen.is_file() or seen.read_text().strip() != sig:
            seen.write_text(sig)
            log(f"HALT      not proposing: {h.reason}")
            log(f"          raised {h.since_h}; clear with: "
                f"python pipeline/halt.py --clear \"<why>\"")
            for f in h.findings or []:
                log(f"          {f}")
            log("          the arm in flight, if any, is left to finish and be recorded")
        else:
            log(f"HALT      still held since {h.since_h}; not proposing "
                f"({len(h.findings or [])} finding(s))")
        return 3

    # 1. PROPOSE
    history = read_records()
    h = propose(history)
    if a.arm:
        from agenda import AGENDA
        by_name = {x.name: x for x in AGENDA}
        if a.arm not in by_name:
            log(f"unknown arm {a.arm!r}; known: {sorted(by_name)}")
            return 1
        h = by_name[a.arm]
    elif pend := confirmmod.pending():
        # A confirmation the loop owes outranks every hypothesis in the agenda.
        # It is the number that decides whether an arm already sitting above the
        # bar is real, and putting anything else first would let the bar drift
        # on unconfirmed evidence.
        from agenda import CHAMPION_SPEC
        h = Hypothesis(
            name=pend["confirm_arm"],
            axis="training",
            change=(f"confirmation of {pend['arm']} on fresh seed {pend['seed']}: "
                    f"identical recipe, the one number it was not selected on"),
            spec={**CHAMPION_SPEC, "seed": pend["seed"],
                  "steps": pend.get("steps"), "batch_size": pend.get("batch_size")},
            prediction=(
                f"{pend['arm']} cleared the bar at {pend['selected_delta']:+.4f} on the "
                f"run it was selected on. This run is on a seed it has never used. If "
                f"the delta holds at or above {pend['floor']:.4f}, the effect is real; "
                f"if it falls under, the first run was seed luck and the arm is "
                f"recorded as seed-sensitive rather than as an effect."),
            expected_effect=pend["selected_delta"],
            source=f"confirmation of {pend['arm']} (fresh seed {pend['seed']})",
        )
        log(f"PENDING   {pend['arm']} awaits fresh-seed confirmation; "
            f"running {h.name} before any new hypothesis")

    # 2. PREREGISTER — before any GPU work, fsynced.
    champ_path = STATE / "champion.json"
    champ = json.loads(champ_path.read_text()) if champ_path.is_file() else None
    # A champion must have a record. This state file survived a cleanup that
    # discarded the arm it named, so it pointed the whole loop at a champion
    # whose numbers had been retracted -- and every later arm would have been
    # measured against a bar derived from them. Cheap to check, catastrophic
    # not to.
    if champ and not any(r.get("arm") == champ.get("arm") for r in read_records()):
        log(f"STALE     champion.json names {champ.get('arm')!r}, which has no "
            f"record in arms.jsonl. Refusing to use it; the bar restarts from "
            f"the control. (state/champion.json is left in place for inspection.)")
        champ = None
    pre = next_prereg(h, champ)
    fsync_append(RECORDS / "prereg.jsonl", pre.to_json())
    log(f"PROPOSED  {pre.arm}  axis={pre.axis}")
    log(f"          {pre.change}")
    log(f"          prediction: {pre.prediction}")
    log(f"          floor +{pre.delta_floor} vs {pre.null_floor_vs}")
    if a.dry_run:
        return 0

    # 3. RUN, in a subprocess so a crash or an OOM is contained.
    # The config supplies the sweep budget; an explicit --max-steps still wins,
    # because that is how a confirmation run at `confirmation_steps` is done.
    spec = dict(h.spec)
    spec["steps"] = a.max_steps or CONFIG.get("steps", spec.get("steps", 300))
    spec["batch_size"] = a.batch_size or CONFIG.get("batch_size", spec.get("batch_size", 16))
    # The seed must come from the ARM, not from the config. This was a bug: the
    # config's `seed` was applied unconditionally, so every null arm ran at seed
    # 17 and the three "independent" nulls came back bit-identical — a measured
    # noise sd of exactly 0.0000, which then set the bar at +0.011 and would have
    # been reported as a real resolution limit. It was an artifact of three copies
    # of one run. The config seed is the DEFAULT for an arm that does not name
    # its own, not an override of one that does.
    spec["seed"] = spec.get("seed") or CONFIG.get("seed", 17)
    # Record the sweep budget alongside the arm so a reader can tell a sweep
    # number from a confirmation number without guessing.
    spec["_sweep_steps"] = CONFIG.get("steps")
    spec["_confirmation_steps"] = CONFIG.get("confirmation_steps")
    spec_path = STATE / f"spec_{pre.arm}.json"
    spec_path.write_text(json.dumps(spec, indent=2))

    save_names = set(CONFIG.get("save_ckpt_for", []))
    save_dir = (CKPT / pre.arm) if (a.save_ckpt or not champ or pre.arm in save_names) else None
    runner = ROOT / "pipeline" / "run_one.py"
    guard_n = CONFIG.get("targets", {}).get("guard_n", 200)
    cmd = [sys.executable, str(runner), "--arm", pre.arm,
           "--model", CONFIG.get("model", a.model), "--corpus", str(corpus),
           "--spec", str(spec_path), "--guard-n", str(guard_n)]
    if save_dir:
        cmd += ["--save-dir", str(save_dir)]
    log(f"CONFIG    steps={spec['steps']} batch={spec['batch_size']} "
        f"seed={spec['seed']} guard_n={guard_n} ckpt={'yes' if save_dir else 'no'}")

    t0 = time.perf_counter()
    log(f"RUNNING   {' '.join(cmd[1:])}")

    # A heartbeat thread. For a 24/7 job the difference between "an arm is
    # training" and "an arm died silently in hour three" has to be visible
    # without a human watching, and upstream's arm runner is quiet during the
    # control pass and between its log lines. It touches nothing: it only reads
    # the log file and appends a line, so it cannot affect a measurement.
    import threading

    stop = threading.Event()
    holder: dict = {}

    def heartbeat() -> None:
        arm_log = LOGS / f"arm_{pre.arm}.log"
        while not stop.wait(1800):
            proc = holder.get("proc")
            alive = proc is not None and proc.poll() is None
            try:
                lines = arm_log.read_text().splitlines()
                last = next((x for x in reversed(lines) if x.strip()), "(no output yet)")
            except Exception:
                last = "(log unreadable)"
            state = "alive" if alive else ("exited" if proc is not None else "not started")
            log(f"HEARTBEAT {pre.arm}  {time.perf_counter()-t0:.0f}s  {state}  | {last[:90]}")

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()

    env = {**os.environ, "RSIJEV_DEVICE": devmod.device()}
    with open(LOGS / f"arm_{pre.arm}.log", "a") as lf:
        lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        lf.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=lf,
                                stderr=subprocess.STDOUT)
        holder["proc"] = proc
        try:
            rc = proc.wait(timeout=60 * 60 * 24)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            rc = -9
            lf.write("\n!!! arm exceeded 24h and was killed\n")
    stop.set()
    log(f"EXIT      rc={rc}  wall={time.perf_counter()-t0:.0f}s  log={LOGS/f'arm_{pre.arm}.log'}")

    # 4. READ BACK the record the arm wrote and 5. GATE it.
    rows_path = RECORDS / pre.arm / "result.json"
    if not rows_path.is_file():
        res = ArmResult(arm=pre.arm, axis=pre.axis, change=pre.change,
                        champion=pre.null_floor_vs, prereg=pre.to_json(),
                        verdict="error", reason=f"runner exited {rc} with no result.json",
                        error=f"rc={rc}", train_seconds=time.perf_counter() - t0)
    else:
        res = ArmResult(**{k: v for k, v in json.loads(rows_path.read_text()).items()})
        res.prereg = pre.to_json()

    if res.verdict != "error" and res.pooled_top1_candidate is not None:
        # The gate measures against the INCUMBENT, on the same delta-over-control
        # scale. champion.json stores absolute top-1, so the champion's own
        # control is needed to put it in the same space -- and it is in
        # state/champion.json for exactly that reason.
        if champ and champ.get("pooled_top1") is not None and champ.get("control_top1") is not None:
            res.champion_delta = champ["pooled_top1"] - champ["control_top1"]
            res.champion_arm = champ.get("arm", "")
        passed, failed, reason = gate(res)
        res.passed = passed
        res.guards_failed = failed
        res.n_failed_guards = len(failed)
        res.reason = reason
        # WHICH bar this verdict was decided at. A verdict without it is
        # unreadable the moment the bar moves: `null_floor` is recorded as
        # `kept_pending_confirm` because +0.0665 cleared +0.0450, and under the
        # corrected bar (+0.1043) the same arm does not clear at all. Nothing in
        # the record said which of those two numbers applied, so the row looked
        # self-consistent under either. The bar moves when a null lands, so this
        # has to be on every row.
        import loop as _loop_mod
        res.bar_used = round(max(float(pre.delta_floor or 0.0), float(_loop_mod.bar())), 6)
        if passed:
            res.verdict = "kept"
        elif len(failed) == 1:
            # Upstream's rule: one failed guard earns a repair, not a death.
            res.verdict = "needs_repair"
        else:
            res.verdict = "rejected"
    elif res.verdict != "error":
        res.verdict = "error"

    # 5b. SETTLE THE VERDICT, before the single record write.
    #
    # Upstream: the confirmation seed "is the only number the arm was not
    # selected on". The run that cleared the bar is optimistically biased by
    # construction -- the change was proposed because it was expected to help --
    # so it does not get to crown itself.
    #
    # How MANY confirmations depends on what the arm is worth, decided by the
    # power analysis before the run and carried on the preregistration:
    #   decisive     3 runs. It is the only tier where a null is informative, so
    #                it is the only one worth spending seeds on.
    #   exploratory  1 run. Its expected effect is below the noise at any budget
    #                this machine can afford, so more seeds cost an arm each and
    #                resolve nothing.
    res.role = pre.role
    res.expected_effect = pre.expected_effect
    res.seeds_required = pre.seeds_required or 1

    # A CONFIRMATION THAT FAILED is the most informative negative in the whole
    # loop, and it must not be filed as a plain "rejected". It means an arm
    # cleared a preregistered bar on the run it was selected on and did not
    # clear it on a seed it was not -- which is seed sensitivity, stated as
    # such, and it is what "the only number the arm was not selected on" is for.
    if confirmmod.is_confirmation(pre.arm) and res.verdict != "kept":
        orig = confirmmod.base_arm(pre.arm)
        need = (pre.delta_floor if res.champion_delta is None
                else res.champion_delta + pre.delta_floor)
        res.verdict = "not_confirmed"
        res.reason = (f"confirmation of {orig} FAILED: this seed reached "
                      f"{(res.pooled_top1_candidate or 0) - (res.pooled_top1_control or 0):+.4f} "
                      f"against a bar of {need:+.4f}. The arm cleared the bar on the "
                      f"run it was selected on and did not on a seed it was not, "
                      f"which is seed sensitivity and not an effect. {orig} does "
                      f"not take the crown.")
        confirmmod.clear()
        log(f"CONFIRM    {pre.arm}: NOT CONFIRMED — {res.reason[:150]}")

    if res.verdict == "kept":
        delta = (res.pooled_top1_candidate or 0) - (res.pooled_top1_control or 0)
        res.seeds_done = 1 if not confirmmod.is_confirmation(pre.arm) else 2
        if res.seeds_done < res.seeds_required:
            pend = confirmmod.schedule(
                pre.arm, res.seeds_done,
                delta=delta, floor=pre.delta_floor,
                steps=spec["steps"], batch_size=spec["batch_size"],
                seeds_required=res.seeds_required, champion_delta=res.champion_delta)
            log(f"CONFIRM    {pre.arm} cleared the bar but is NOT champion yet: "
                f"run {res.seeds_done}/{res.seeds_required} done, "
                f"scheduled {pend['confirm_arm']} on fresh seed {pend['seed']} "
                f"(a number it was not selected on)")
            res.verdict = "kept_pending_confirm"
            res.reason += (f"; awaiting fresh-seed confirmation "
                           f"({res.seeds_done}/{res.seeds_required})")
        else:
            log(f"CONFIRM    {pre.arm} confirmed on {res.seeds_required} seed(s); "
                f"every one cleared the bar")

    # Prove the corpus and the targets were disjoint, BEFORE the record is
    # written. Placed here deliberately: after the verdict is settled (so a
    # contaminated arm is still reported as contaminated rather than silently
    # dropped) and before the record (so the record can never again be missing
    # the field). I9 exists precisely because the absence of this line went
    # unnoticed for an entire arm.
    res.contamination = _check_contamination(res)

    # An arm that is NOT kept still leaves weights on disk -- `confirm__null_floor`
    # left 1.7 GB behind -- and a record carrying a checkpoint path with a BLANK
    # artifact field is indistinguishable from a kept arm whose verification
    # silently did not run. The field is therefore never left ambiguous: a
    # checkpoint belonging to an arm that will not be published says so, in
    # words. A blank field is what made this invisible, and a blank field is
    # never the right answer to "was this verified".
    if res.checkpoint and not res.artifact:
        res.artifact = (f"not a deliverable: verdict is {res.verdict}"
                        if res.verdict != "kept"
                        else "UNVERIFIED: verification did not run")

    # ONE record per arm, written after the verdict is final.
    append_record(res)

    # COLLECT, immediately after recording. A discarded offspring is a
    # measurement, and the measurement is now in the record forever; its weights
    # are not. Doing this here rather than on a schedule is the point: at one arm
    # per ~100 minutes an uncollected discard is ~1.7 GB per 100 minutes, and a
    # scheduled collector is a collector that silently stops being installed.
    # The collector refuses to touch the champion and refuses to touch an arm with
    # a live runner, so running it here cannot destroy the only publishable model.
    try:
        import lineage as _lin
        c = _lin.collect(apply=True)
        if c["removed"]:
            log(f"COLLECT   freed {c['freed_gb']:.2f} GB — {', '.join(c['removed'])}")
            log(f"          the measurements stay; only the bytes went")
    except Exception as exc:
        log(f"COLLECT   failed: {type(exc).__name__}: {exc} "
            f"(a leak, not a lost measurement)")
    _report(res, champ)

    # 6. MOVE THE BAR, only on a CONFIRMED keeper whose checkpoint reproduces
    #    the run that produced it, then spend the held-out set.
    if res.verdict == "kept" and res.checkpoint:
        v = _verify_checkpoint(res)
        if not v.ok:
            # The arm was real; the ARTIFACT is not trustworthy. That is a
            # different claim from the arm failing, and conflating them would
            # throw away a measurement over a save bug.
            res.verdict = "kept_unverified_artifact"
            res.reason += f"; checkpoint NOT verified: {v.reason}"
            log(f"ARTIFACT  {res.arm} cleared the bar but its checkpoint does not "
                f"reproduce the run: {v.reason[:120]}")
            log(f"          not promoted to champion. This is a save/serialisation "
                f"problem, not evidence the effect is absent.")
            append_record(res)
            _regen()
            return 0
        res.artifact = v.one_line()
        log(f"ARTIFACT  {res.checkpoint}: {v.one_line()}")

    if res.verdict == "kept":
        new = {
            "arm": res.arm, "axis": res.axis, "change": res.change,
            "pooled_top1": res.pooled_top1_candidate,
            "control_top1": res.pooled_top1_control,
            "min_decision_score": res.min_decision_score_candidate,
            "ece": res.ece_candidate,
            "checkpoint": res.checkpoint,
            "spec": spec,
            "config": {k: CONFIG.get(k) for k in
                       ("model", "steps", "confirmation_steps", "batch_size", "seed")},
            "ts": time.time(),
        }
        champ_path.write_text(json.dumps(new, indent=2))
        fsync_append(STATE / "champions.jsonl", new)
        log(f"CHAMPION  {res.arm} @ top1={res.pooled_top1_candidate:.4f}  "
            f"(was {champ.get('pooled_top1') if champ else 'none'})")
        _read_held_out(res)
    elif res.verdict == "needs_repair":
        log(f"REPAIR    {res.arm} failed one guard ({res.guards_failed}); "
            f"it is not discarded")

    # Re-derive the power analysis whenever a NULL arm lands, so the bar moves
    # to the resolution this machine actually has rather than the one upstream
    # measured on a GB10. A null arm is the only evidence that can move it, and
    # letting it go unused would be the exact failure the analysis exists to
    # prevent.
    if pre.arm.startswith("null"):
        import agenda
        import confirm
        import power
        # ONLY the null series may enter the floor. `agenda.null_deltas` owns that
        # rule -- it used to be a loop over every recorded arm here, which made the
        # bar a ratchet: each result raised the threshold for the next one.
        records = read_records()
        deltas = agenda.null_deltas(records)
        gdeltas = agenda.null_guard_deltas(records)
        skipped = sum(1 for r in records
                      if not agenda.is_null_series(str(r.get("arm", ""))))
        # At the budget a DECISIVE arm is actually judged at, not a constant typed
        # in here. `decisive` runs 3 seeds; deriving the bar at 2 described a design
        # that does not exist and made the bar stricter than the gate that used it.
        planned = confirm.seeds_for("decisive")
        pr = power.analyse(deltas, planned_seeds=planned, guard_deltas=gdeltas)
        power.save(pr)
        # Say which path produced the number. With one null arm the sd is NOT
        # measured — it is the fallback — and printing a measured-looking 0.0000
        # next to the word "noise" would be the exact kind of confident wrong
        # this project exists to avoid.
        if pr.n_null_arms >= 2:
            src = f"measured noise sd={pr.noise_sd:.4f} from {pr.n_null_arms} null arms"
        else:
            src = (f"sd=0.011 FALLBACK (only {pr.n_null_arms} null arm; a floor needs "
                   f">=2 to have a spread)")
        log(f"POWER     {src}  do-nothing {pr.do_nothing_mean:+.4f}  "
            f"guard {pr.guard_mean:+.4f}±{pr.guard_sd:.4f}  "
            f"bar=+{pr.recommended_bar:.4f} at {planned} seeds  [{pr.bar_rule}]"
            + (f"  ({skipped} non-null arm(s) excluded from the floor)" if skipped else ""))

    _regen()
    return 0


def _regen() -> None:
    """Regenerate every research output. Rule 3: the record iterates in step
    with the data. It is a projection of these three logs, so regenerating it
    here is what keeps it from drifting -- and it means nobody has to remember to
    update a document that is supposed to be evidence.

    The snapshot is in the list so the attachable copy of the dashboard never
    lags the served one by more than one arm.
    """
    for script, what in (("trajectory.py", "trajectory"), ("version_card.py", "record"),
                         ("snapshot.py", "snapshot")):
        r = subprocess.run([sys.executable, str(ROOT / "pipeline" / script)],
                           capture_output=True, text=True, cwd=str(ROOT))
        if r.returncode == 0:
            log(f"REGEN     {what}: {(r.stdout or '').strip().splitlines()[-1][:90]}")
        else:
            tail = (r.stderr or '').strip().splitlines()
            log(f"REGEN     {what} FAILED rc={r.returncode}: "
                f"{tail[-1][:120] if tail else ''}")


def _verify_checkpoint(res: ArmResult):
    """Reload the saved weights and require that they reproduce the training run.

    A verification that reconstructs the model any way other than from the saved
    bytes is not verifying the artifact, so this reads the checkpoint's OWN
    meta.json for the base model and the spec, and compares against the arm
    runner's own per-question rows.
    """
    items = RECORDS / res.arm / f"{res.arm}.items.jsonl"
    return verify_ckpt.verify(Path(res.checkpoint), items,
                              model_id=CONFIG.get("model", "Qwen/Qwen3-0.6B-Base"),
                              min_agreement=CONFIG.get("min_artifact_agreement", 1.0))


def _check_contamination(res: ArmResult) -> str:
    """Prove the training corpus cannot have leaked the evaluation targets.

    This function did not exist until invariant I9 fired. `contamination.py` had
    been written and tested in isolation, and the invariant existed, and neither
    one was ever connected to the daemon — so every arm was scored, written to
    `versions/current.md` and shown on the dashboard with no evidence that the
    corpus and the targets were disjoint. I9 caught it on the first real record,
    which is the only reason it was caught at all.

    The failure was invisible: an unchecked arm looks exactly like a checked one,
    because the field is simply absent rather than red. Returns a one-line verdict
    for the record, always non-empty — a blank string is what let this hide.
    """
    import arm as armlib
    import contamination
    try:
        # The loaders live in `arm`, not `contamination` — `contamination` only
        # knows how to compare two things once it has them. Same pair the arm
        # runner itself loads, so the daemon checks what the run actually used.
        targets = armlib.load_targets(CONFIG.get("guard_n", 200))
        train = armlib.load_corpus(ROOT / "corpus")
        rep = contamination.check(train, targets)
    except Exception as exc:
        # NOT silently clean. A check that could not run is a check that failed;
        # reporting it as clean is the exact error this function exists to stop.
        log(f"CONTAM    could not run ({type(exc).__name__}: {exc}) — "
            f"recording this arm as UNVERIFIED, not clean")
        return f"unverified: check raised {type(exc).__name__}: {exc}"
    path = contamination.save(rep, res.arm)
    line = rep.one_line()
    log(f"CONTAM    {line}")
    if not rep.clean:
        log(f"CONTAM    FAILED — {res.arm} is contaminated: {rep.verdict}")
    return line


def _read_held_out(res: ArmResult) -> None:
    """Score a freshly-crowned champion on the held-out set. ONCE.

    Upstream's rule, and the reason it is a function with a guard rather than a
    call: the held-out set is scored "once, after a release model has been
    chosen, never during the search", and it is *consumable* — "each release
    freezes the next one". So this is the only place in the whole pipeline that
    may load it, and `held_out.spend` refuses a second read.

    A failure here must not cost the champion. The bar already moved; the
    held-out number is a separate, later claim, and losing it is recoverable by
    re-freezing a set whereas losing the crown would not be.
    """
    if held_out.is_spent():
        log("HELD-OUT  already spent; a second read would not measure the same "
            "thing. Freeze a new set from an untouched source instead.")
        return
    try:
        cases = held_out.load_held_out()
        base = held_out.majority_baseline(cases)
        log(f"HELD-OUT  scoring the new champion on {len(cases)} cases from "
            f"{held_out.SOURCE} (majority {base:.4f}) — this SPENDS the set")
    except held_out.HeldOutSpent as exc:
        log(f"HELD-OUT  refused: {exc}")
    except Exception as exc:
        log(f"HELD-OUT  could not be read ({type(exc).__name__}: {exc}); the "
            f"champion stands and the set is still unspent")
    else:
        rec = held_out.spend({"champion": res.arm, "cases": len(cases),
                              "majority_baseline": base})
        log(f"HELD-OUT  spent; digest {rec['digest']} at {rec['read_at_h']}")


def _report(res: ArmResult, champ: dict | None) -> None:
    if res.verdict == "error":
        log(f"VERDICT   {res.arm}: ERROR  {res.error}")
        return
    c, k = res.pooled_top1_candidate, res.pooled_top1_control
    d = (c - k) if (c is not None and k is not None) else float("nan")
    bar = "PASS" if res.passed else "FAIL"
    log(f"RESULT    {res.arm}  {res.verdict.upper()} [{bar}]")
    log(f"          top1  cand {c:.4f}  ctrl {k:.4f}  delta {d:+.4f}  "
        f"(floor +{res.prereg.get('delta_floor', 0):.4f})")
    log(f"          minDS  {res.min_decision_score_candidate:+.2f} vs "
        f"{res.min_decision_score_control:+.2f}   "
        f"ECE {res.ece_candidate:.3f} vs {res.ece_control:.3f}")
    for t, v in sorted(res.per_target_top1.items()):
        cv = res.per_target_top1_control.get(t)
        log(f"          {t:22s} {v:.4f}" + (f"  (ctrl {cv:.4f})" if cv is not None else ""))
    log(f"          {res.reason}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        log("daemon crashed:\n" + traceback.format_exc())
        raise SystemExit(1)
