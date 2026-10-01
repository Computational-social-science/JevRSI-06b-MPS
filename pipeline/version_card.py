#!/usr/bin/env python3
"""The research record, regenerated from the logs after every arm.

EDITABLE — ours. Rule 3 of this project is "keep the research output iterating
in step", so the record is not a document anyone maintains: it is a projection
of `prereg.jsonl`, `arms.jsonl` and `champions.jsonl`, rewritten on every arm.
It cannot drift from the data because it is derived from the data.

The structure follows upstream's release-card discipline, because that
discipline is the part worth copying:

  * one card per version, all of them kept, none overwritten;
  * what was tried, what shipped, what was killed, and by which measurement;
  * the caveats attached to every figure.

It adds the two things upstream's cards carry and a reproduction needs and this
project's setting makes load-bearing:

  * the POWER section, so every null is labelled "no effect we could see" or
    "no effect" by the resolution this machine actually had;
  * the ENVIRONMENT section, so no number is readable without the kernel stack
    and device it was produced on — the rule upstream states and the reason an
    MPS number is never pooled with a CUDA one.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

OUT = ROOT / "versions"
CEILINGS = {
    "majority_baseline": 0.520,
    "latent_factor_bound": 0.704,
    "teacher_self_agreement": 0.735,
    "quirk_line": 0.750,
}
# Upstream's published numbers, for scale. Different backbone, different
# hardware, so these are CONTEXT and never a comparison our deltas are made
# against — the control inside each arm is.
UPSTREAM = {
    "note": "Qwen3.5-2B tower on an NVIDIA GB10. Context only.",
    "jev_typed_decisions": 0.727,
    "rsijev_v3.0_2b_typed_decisions": 0.791,
    "rsijev_v1.0_2b_pooled": 0.662,
    "rsijev_v1.0_0.8b_suite": None,
}


def load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _cfg_min_artifact_agreement() -> float:
    """The artifact floor, read from the run config so the record cannot quote a
    different number from the one the gate used."""
    import json
    p = ROOT / "config" / "run.json"
    if p.is_file():
        try:
            return float(json.loads(p.read_text()).get("min_artifact_agreement", 1.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return 1.0


def build() -> str:
    import power
    import held_out
    from loop import RECORDS, STATE, read_prereg, read_records

    prereg = {p["arm"]: p for p in read_prereg()}
    arms = read_records()
    champs = load(STATE / "champions.jsonl")
    pw = power.load()
    cost_p = ROOT / "state" / "cost.json"
    cost = json.loads(cost_p.read_text()) if cost_p.is_file() else {}
    cfg_p = ROOT / "config" / "run.json"
    cfg = json.loads(cfg_p.read_text()) if cfg_p.is_file() else {}
    corpus_m = ROOT / "corpus" / "manifest.json"
    manifest = json.loads(corpus_m.read_text()) if corpus_m.is_file() else {}

    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    kept = [a for a in arms if a.get("verdict") == "kept"]
    rejected = [a for a in arms if a.get("verdict") == "rejected"]
    repair = [a for a in arms if a.get("verdict") == "needs_repair"]
    errors = [a for a in arms if a.get("verdict") == "error"]

    L: list[str] = []
    w = L.append
    w("# RSI-Jev on Qwen3-0.6B — research record")
    w("")
    w(f"*Generated {now} from `records/prereg.jsonl`, `records/arms.jsonl` and*")
    w("*`state/champions.jsonl`. Not maintained by hand: it cannot drift from the*")
    w("*data, because it is a projection of the data.*")
    w("")
    w("## 1. What this is")
    w("")
    w("A replication of [RSI-Jev](https://github.com/Shanghua-Gao/RSI-Jev)'s *method* "
      "rather than its checkpoints: a")
    w("self-improving loop that proposes a hypothesis, **registers the prediction "
      "before spending compute**, runs")
    w("the experiment, and retires its own champion when the measurement says to. "
      "The backbone is")
    w(f"`{cfg.get('model', '?')}` on local Apple MPS, against upstream's "
      f"`Qwen3.5-2B` on an NVIDIA GB10.")
    w("")
    w("Upstream code is vendored **verbatim** and unmodified except for one "
      "performance line in")
    w("`rsijev/evaluate.py`, proven bit-identical by `tests/test_eval_parity.py`. "
      "So a difference")
    w("between our numbers and theirs cannot be attributed to a different "
      "implementation — only to a")
    w("different backbone, a different kernel stack, or a real effect.")
    w("")
    w("## 2. The question")
    w("")
    w("> Are RSI-Jev's conclusions about *what changes a decision model* general "
      "laws, or artefacts of one")
    w("> particular backbone and machine?")
    w("")
    w("Upstream reports 204 arms across four releases. Its own most useful output "
      "is a **negative**:")
    w("in its first 24-arm run, nothing cleared the bar, and the reason it "
      "concluded is quoted in §5.")
    w("")

    w("### What this search can and cannot conclude")
    w("")
    w("Stated before any result, because it governs how every result below may "
      "be read. Upstream reports effect *E* at 2B; we test *E* at 0.6B.")
    w("")
    w("| | |")
    w("|---|---|")
    w("| **E confirmed here** | the effect survived a 3.3x scale reduction and a "
      "different kernel stack — **strong evidence it is a law**, not an artefact "
      "of one setup |")
    w("| **E null here** | **ambiguous**: consistent both with *E was never real* "
      "and with *E is real and smaller at 0.6B*. This design cannot distinguish "
      "them. |")
    w("")
    w("> **This search can confirm positives. It cannot refute them.** Every null "
      "below is a null *at this scale*. A null here is not evidence against "
      "upstream's finding; it is evidence that the finding did not reproduce here "
      "— a much weaker sentence, and the only one this design is entitled to.")
    w("")
    w("A second limit, on the power itself: the control is the frozen model's "
      "logprob readout, identical for every arm, so the nulls' spread measures "
      "**training** variance with the evaluation's variance cancelled. It is a "
      "*paired* sd and is therefore smaller than upstream's unpaired per-seed "
      "0.011 — so the fallback in force until the nulls finish **overstates** our "
      "noise and sets the bar too high. That errs safe, and the record says so "
      "rather than quietly benefiting.")
    w("")
    w("## 3. Environment")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| backbone | `{cfg.get('model', '?')}`, {cfg.get('steps', '?')} steps "
      f"(sweep) / {cfg.get('confirmation_steps', '?')} (confirmation) |")
    w(f"| device | `{cost.get('device', '?')}`, torch {cost.get('torch', '?')}, fp32 |")
    w("| kernel stamp | `torch-reference/torch-*/mps` — **not** comparable with "
      "a CUDA or fla run |")
    w(f"| eval target | typed-decisions test, 400 cases / 2,000 questions, soft "
      f"gold |")
    w(f"| guard | MMLU-Pro, {cfg.get('targets', {}).get('guard_n', '?')} of 1,000 "
      f"rows (general-knowledge veto) |")
    if manifest:
        w("| corpus | " + ", ".join(
            f"`{k}` {v['cases']} cases sha `{v['sha256'][:12]}`"
            for k, v in manifest.items()) + " |")
    w("")
    w("Numbers only compare within one kernel stack. This is upstream's rule, "
      "applied rather than")
    w("assumed: every record here carries the device in its kernel stamp, and no "
      "MPS number is")
    w("pooled with a CUDA one.")
    w("")

    w("## 4. Power — what this setup could have seen")
    w("")
    if pw and pw.n_null_arms >= 2:
        w(f"Measured from **{pw.n_null_arms} null arms** (arms identical to the "
          f"control, so their spread is")
        w(f"the noise floor): sd = **{pw.noise_sd:.4f}**.")
    else:
        w(f"**{pw.n_null_arms if pw else 0} null arms so far.** Until there are at "
          f"least two, the noise floor is")
        w("not measured and the fallback is upstream's per-seed sd of 0.011, taken "
          "at the conservative")
        w("end. This is stated rather than hidden because it is a weaker claim.")
    w("")
    if pw:
        w("| seeds | minimum detectable effect |")
        w("|---|---|")
        for n, v in sorted(pw.mde.items()):
            if v and v < 1e6:
                w(f"| {n} | {v:.4f} |")
        w("")
        w(f"**Bar in force: +{pw.recommended_bar:.4f}.**")
        w("")
        vis = [k for k, ok in pw.detectable.items() if ok]
        invis = [k for k, ok in pw.detectable.items() if not ok]
        w(f"Of the effects upstream reports, this design could detect: "
          f"{', '.join(vis) if vis else '**none**'}.")
        w("")
        w(f"It could **not** detect: {', '.join(invis) if invis else 'none'}.")
        w("")
        w("This is the single most important number in this document. A null "
          "verdict on an arm whose")
        w("expected effect is below the MDE means *no effect we could see*, not "
          "*no effect*, and the")
        w("trajectory labels those arms `underpowered` rather than calling them "
          "published negatives.")
    w("")

    w("## 5. Why the bar is where it is")
    w("")
    w("Upstream diagnosed its own first run's failure, and the diagnosis is the "
      "reason this")
    w("pipeline measures its noise floor *before* proposing any hypothesis:")
    w("")
    w("> **The bar was above the resolution.** A difference of two 3-seed means "
      "has an sd of about")
    w("> 0.010, so a realistic single-axis effect of +0.005 is invisible against "
      "a +0.020 threshold. The")
    w("> run was not powered to find what it was looking for.")
    w("")
    w("A search whose threshold sits above its own resolution cannot distinguish "
      "\"this lever does")
    w("nothing\" from \"this lever does something smaller than I could see\" — and "
      "reports the first.")
    w("Upstream spent 24 arms learning that. The first arm here is a null arm for "
      "exactly this reason.")
    w("")

    # ---- the agenda, ordered by what the design can resolve
    import agenda as _ag
    roles: dict[str, list] = {}
    for h in _ag.AGENDA:
        roles.setdefault(h.role, []).append(h)
    mde = pw.recommended_bar if pw else None
    w("### The agenda, ordered by what this design can resolve")
    w("")
    if mde:
        w(f"Minimum detectable effect at the planned budget: **{mde:.4f}**. An arm's "
          f"*role* is decided from its expected effect against that number, **before** "
          f"it runs — never from what it produced. That is what stops a null from being "
          f"read as a refutation when the design could not have seen one.")
    else:
        w("The minimum detectable effect is not yet measured, so every arm is currently "
          "classified against the fallback.")
    w("")
    w("| role | meaning | seed budget | arms |")
    w("|---|---|---|---|")
    w(f"| **calibration** | replicas of the champion recipe; their SPREAD is the noise "
      f"floor the bar is set at | 1 | {', '.join('`'+h.name+'`' for h in roles.get('calibration', []))} |")
    w(f"| **decisive** | expected effect above the detectable minimum — a null here is "
      f"informative | 3 | {', '.join('`'+h.name+'`' for h in roles.get('decisive', []))} |")
    w(f"| **exploratory** | expected effect below it — can only ever confirm, never "
      f"refute | 1 | {len(roles.get('exploratory', []))} arms |")
    w("")
    w("The order is derived from the power analysis, not hand-written, so a change in "
      "the measured floor re-orders the queue instead of quietly making it wrong. The "
      "seed budget follows the role: a `decisive` arm is the only one where more seeds "
      "change what is knowable, so it is the only one they are spent on.")
    w("")
    w("## 6. Results")
    w("")
    if not arms:
        w("*No arm has completed yet.*")
    else:
        w(f"{len(arms)} arms: **{len(kept)} kept**, {len(rejected)} published "
          f"negatives, {len(repair)} needing repair, {len(errors)} errors.")
        w("")
        w("| arm | role | Δ vs control | expected | bar | contamination | verdict |")
        w("|---|---|---|---|---|---|---|")
        for a in arms:
            c, k = a.get("pooled_top1_candidate"), a.get("pooled_top1_control")
            d = f"{c - k:+.4f}" if c is not None and k is not None else "—"
            pr = prereg.get(a.get("arm"), {})
            exp = pr.get("expected_effect")
            exp_s = f"{exp:+.4f}" if isinstance(exp, (int, float)) else "—"
            und = ""
            if pw and isinstance(exp, (int, float)) and exp and a.get("verdict") != "kept" \
                    and abs(exp) < pw.recommended_bar:
                und = " *(underpowered)*"
            ct = a.get("contamination") or ""
            ct_s = ("clean" if ct.startswith("clean") else
                    "**NOT CHECKED**" if not ct else ct[:38])
            seeds = (f" ({a.get('seeds_done', 1)}/{a.get('seeds_required', 1)} seeds)"
                     if a.get("seeds_required", 1) > 1 else "")
            bu = a.get("bar_used")
            bu_s = f"+{float(bu):.4f}" if isinstance(bu, (int, float)) else "?"
            w(f"| `{a.get('arm')}` | {a.get('role') or '—'}{seeds} | {d} | {exp_s} | "
              f"{bu_s} | {ct_s} | {a.get('verdict')}{und} |")
        # A verdict is unreadable without the bar that produced it, and the bar
        # moves every time a null lands. `null_floor` is kept_pending_confirm
        # because +0.0665 cleared +0.0450; under the bar now in force it does not
        # clear at all. Both statements are true of the same number, so the record
        # has to say which threshold applied rather than leaving the reader to
        # assume the current one.
        stale = [a for a in arms if isinstance(a.get("bar_used"), (int, float))
                 and pw and abs(float(a["bar_used"]) - pw.recommended_bar) > 1e-6]
        unrec = [a for a in arms if a.get("bar_used") is None]
        if pw and (stale or unrec):
            w("")
            if stale:
                w(f"**Gated at a superseded bar.** The bar now in force is "
                  f"+{pw.recommended_bar:.4f} "
                  f"({pw.bar_rule or 'rule not recorded'}); these rows were decided at "
                  + ", ".join(f"`{a['arm']}` +{float(a['bar_used']):.4f}" for a in stale)
                  + ". Their verdicts stand as recorded — the measurement did not "
                    "change — but they are not comparable to rows gated at the "
                    "current bar, and the honest reading of a keeper among them is "
                    "'cleared a bar that doing nothing also cleared'.")
            if unrec:
                w(f"**{len(unrec)} row(s) predate the `bar` column** ("
                  + ", ".join(f"`{a['arm']}`" for a in unrec)
                  + "): the record cannot say which bar decided them. Treat those "
                    "verdicts as unverified against any bar.")
    w("")

    if kept:
        w("### Champion")
        w("")
        w("| arm | pooled top-1 | control | minDS | ECE | checkpoint |")
        w("|---|---|---|---|---|---|")
        for ch in champs:
            w(f"| `{ch.get('arm')}` | {ch.get('pooled_top1')} | "
              f"{ch.get('control_top1')} | {ch.get('min_decision_score')} | "
              f"{ch.get('ece')} | `{ch.get('checkpoint') or '—'}` |")
        w("")

    w("## 7. Reference points")
    w("")
    w("Read these as ceilings, not targets. The comparison that counts is inside "
      "each arm, between its")
    w("candidate and the control measured in the same call on the same weights.")
    w("")
    w("| | value | what it is |")
    w("|---|---|---|")
    for k, v in CEILINGS.items():
        w(f"| {k.replace('_', ' ')} | {v} | typed-decisions test |")
    w(f"| Jev | {UPSTREAM['jev_typed_decisions']} | the reference System One model |")
    w(f"| RSI-Jev v3.0 2B | {UPSTREAM['rsijev_v3.0_2b_typed_decisions']} | "
      f"upstream, typed-decisions |")
    w(f"| RSI-Jev v1.0 2B | {UPSTREAM['rsijev_v1.0_2b_pooled']} | upstream, "
      f"pooled top-1 over 2,000 decisions |")
    w("")
    w(f"_{UPSTREAM['note']} A 0.6B tower is ~3.3x smaller, so these are context, "
      f"not a target: upstream's own "
      f"0.8B and 2B differ by 0.048._")
    w("")

    w("## 8. Discipline")
    w("")
    ho = held_out.status()
    w(f"- **Held-out set**: `{ho.get('source')}` pinned at `{ho.get('revision')}`, "
      f"frozen {ho.get('frozen_at_h', '?')}, "
      + ("**read and spent**" if ho.get("spent") else "**not yet read**")
      + ". Upstream scores it \"once, after a release model has been chosen, never "
        "during the search\", and it is consumable — so `pipeline/held_out.py` refuses "
        "a second read and the champion does not take the crown until a fresh seed has "
        "confirmed it.")
    w("- **Fresh-seed confirmation**: an arm that clears the bar is recorded as "
      "`kept_pending_confirm` and does **not** become champion until the same recipe "
      "clears it on a seed it was not selected on. Upstream: the confirmation seed "
      "\"is the only number the arm was not selected on\".")
    w("- **Contamination is verified per arm, not asserted at build time**: "
      "`pipeline/contamination.py` re-checks the *loaded* corpus against the *loaded* "
      "targets before any compute — exact state hashes and 12-word phrase overlap, "
      "upstream's own rule. An arm whose corpus is not clean is refused rather than "
      "scored. A shared question *key* is deliberately NOT a criterion: the synthetic "
      "corpus is the same task family as the benchmark by design, and a key naming the "
      "same kind of question about a different document is the task being the task.")
    w(f"- **Artifacts are verified before publication**: a champion's checkpoint is "
      f"reloaded from disk and must reproduce its training run's per-question "
      f"predictions (floor {_cfg_min_artifact_agreement():.4f}). Upstream: a checkpoint "
      f"is \"published only if it reproduces its training run's per-question predictions "
      f"exactly\", and a near-agreement is reported rather than rounded into a pass. "
      f"An arm that clears the bar but whose artifact fails is recorded as "
      f"`kept_unverified_artifact` — a save problem, not evidence the effect is absent.")
    w("")

    w("## 9. Cost")
    w("")
    if cost:
        ac = cost.get("arm_cost", {}).get(str(cfg.get("steps", 300)), {})
        w(f"- compute: local MPS, **{ac.get('total_h', '?')} h per arm** at "
          f"{cfg.get('steps')} steps "
          f"({cost['eval']['seconds']}s x2 eval + {cost['train']['seconds_per_step']}s/step)")
        w("- **money: 0.00** — no rented GPU, no inference endpoint, no paid API")
        w("- network: HuggingFace Hub, anonymous, for the open-weights backbone only")
    w("- enforced by `tests/test_no_cost.py`, which fails the build on any "
      "metered-provider")
    w("  reference or credential literal in our code")
    w("")

    w("## 10. Discarded measurements")
    w("")
    disc = ROOT / "versions" / "DISCARDED.md"
    if disc.is_file():
        n = disc.read_text().count("\n## D")
        w(f"**{n} measurement(s) were taken, found invalid, and thrown away.** They are "
          f"written up in [`DISCARDED.md`](DISCARDED.md), because upstream's rule is that "
          f"the record is the evidence — \"failures ship, including the ones that killed "
          f"our own champion\" — and a failure that leaves no trace is indistinguishable "
          f"from one that never happened.")
        w("")
        w("The most instructive: three null arms that turned out to be three copies of a "
          "single run, because a config value had overridden the per-arm seed. They "
          "returned a **measured noise sd of exactly 0.0000** and set the bar at +0.0110. "
          "A noise-free measurement system would have been a remarkable finding; it was a "
          "bug, and the giveaway was that the sd was *exactly* zero.")
    else:
        w("*No measurement has been discarded yet.*")
    w("")

    w("## 11. Reproduction")
    w("")
    w("```bash")
    w(".venv/bin/python tests/test_gates.py        # the bar, the proposer")
    w(".venv/bin/python tests/test_eval_parity.py  # the one patched line is inert")
    w(".venv/bin/python tests/test_no_cost.py      # rule 2, enforced")
    w(".venv/bin/python pipeline/measure_cost.py   # what an arm costs here")
    w(".venv/bin/python pipeline/daemon.py         # one arm, resumable")
    w(".venv/bin/python pipeline/trajectory.py     # this record + the chart")
    w("```")
    w("")
    w("Artifacts: `corpus/manifest.json` (sha256 per training file), "
      "`records/arms.jsonl` (one row per")
    w("arm), `records/prereg.jsonl` (predictions, fsynced before each run), "
      "`state/champions.jsonl`.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "current.md"))
    a = ap.parse_args()
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    md = build()
    p.write_text(md)
    print(f"wrote {p} ({len(md.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
