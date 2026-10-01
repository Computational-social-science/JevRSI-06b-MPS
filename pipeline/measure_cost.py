#!/usr/bin/env python3
"""Measure what an arm actually costs on THIS hardware, then size the agenda.

EDITABLE — ours, and the step before an unattended run.

An arm's cost is two numbers, and they are measured rather than guessed because
every scheduling decision downstream depends on them:

  EVAL_SECONDS   one pass over the evaluation set. It runs TWICE per arm (the
                 control and the candidate) and the control is identical work,
                 so this is a fixed per-arm tax paid before and after training.
  STEP_SECONDS   one optimiser step at the arm's batch size, with the tower
                 training. This is the only number that scales with `steps`.

Knowing both lets the agenda be sized to a budget instead of to a wish. An arm
that costs 14 hours is not run at 1500 steps unattended; it is run at the
largest step count that fits the budget, and the record says which — because a
truncated arm is a different experiment from a shortened one, and a reader has
to be able to tell them apart.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-cases", type=int, default=400)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--train-steps", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "state" / "cost.json"))
    a = ap.parse_args()

    import torch
    import dev as devmod
    from rsijev.contract import load_cases
    from rsijev.targets import load_typed_decisions
    from rsijev.encode import EncodeConfig, iter_questions
    from rsijev.evaluate import predict
    from rsijev.arch import ArchConfig, DecisionModel, LogprobReadout
    from rsijev.fit import FitConfig, fit

    d = devmod.device()
    print(f"device {d}  torch {torch.__version__}")
    lm, tok = devmod.load_base_model("Qwen/Qwen3-0.6B-Base", d)

    # --- EVAL cost, on the real target at the real batch size.
    td = load_typed_decisions("test")[:a.eval_cases]
    nq = sum(len(c.questions) for c in td)
    enc = EncodeConfig(option_order="canonical")
    model = DecisionModel(getattr(lm, "model", lm), 1024,
                          ArchConfig(readout="option_xattn")).to(d)
    model.scorer.to(torch.float32)
    model.eval()
    predict(model, tok, td[:32], enc, max_options=80, device=d,
            batch_size=a.eval_batch)          # warm up
    t0 = time.perf_counter()
    predict(model, tok, td, enc, max_options=80, device=d, batch_size=a.eval_batch)
    eval_s = time.perf_counter() - t0
    print(f"\nEVAL   {len(td)} cases / {nq} questions in {eval_s:.1f}s "
          f"= {nq/eval_s:.1f} q/s")
    per_q = eval_s / nq

    # --- TRAIN cost, with the tower training (the champion spec's freeze_base=False).
    corpus = load_cases(str(ROOT / "corpus" / "synth.jsonl"))[:400]
    import copy
    tower = copy.deepcopy(getattr(lm, "model", lm)).to(torch.float32)
    for p in tower.parameters():
        p.requires_grad_(True)
    tower.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    tm = DecisionModel(tower, 1024, ArchConfig(readout="option_xattn",
                                               freeze_base=False)).to(d)
    tm.scorer.to(torch.float32)
    for p in tm.tower.get_input_embeddings().parameters():
        p.requires_grad_(False)
    t0 = time.perf_counter()
    fit(tm, tok, corpus, enc,
        FitConfig(objective="soft_ce", steps=a.train_steps, batch_size=a.batch_size,
                  lr_head=1e-4, lr_base=5e-6, autocast_bf16=False),
        max_options=80, seed=17, device=d)
    step_s = (time.perf_counter() - t0) / a.train_steps
    print(f"TRAIN  {a.train_steps} steps at batch {a.batch_size} = {step_s:.2f}s/step")

    # --- What the arms will cost, and what fits a budget.
    # The multiplier is the part that is easy to get wrong and expensive to get
    # wrong. An arm does not evaluate once. `run_arm` scores:
    #   * an `in_distribution` holdout — 1 case in 10 of the training corpus,
    #     ~2100 questions, which `run_arm` itself adds and which exists to
    #     separate "the pipeline is broken" from "the head fits its training
    #     sources but does not transfer";
    #   * the primary target, 2000 questions;
    #   * the MMLU-Pro guard, 200 rows;
    # and it does all three under BOTH option orders (the champion spec sets
    # eval_option_orders = [canonical, reversed]) and for BOTH roles (the
    # control and the candidate). So one arm costs
    #     3 targets x 2 orders x 2 roles
    # passes — about 17,000 question-evaluations, ~30 minutes — before a single
    # optimiser step. Measured the naive way this under-counts the fixed tax by
    # roughly 3x, which is enough to promise an overnight finish that does not
    # happen. It is the reason the first measurement of this pipeline was wrong.
    n_orders = 2
    n_roles = 2
    holdout_q = 3 * (len(load_cases(str(ROOT / "corpus" / "synth.jsonl"))) // 10 + 1)
    guard_rows = 200
    passes = n_orders * n_roles
    eval_tax = per_q * (nq + guard_rows + holdout_q) * passes
    print(f"\neval tax per arm = {per_q:.4f}s/question x "
          f"({nq} primary + {holdout_q} in-distribution + {guard_rows} guard) "
          f"x {passes} passes = {eval_tax:.0f}s")

    def arm_cost(steps: int) -> dict:
        tr = step_s * steps
        return {"steps": steps, "eval_s": round(eval_tax), "train_s": round(tr),
                "total_s": round(eval_tax + tr), "total_h": round((eval_tax + tr) / 3600, 2)}

    table = {s: arm_cost(s) for s in (50, 100, 150, 300, 600, 1500)}
    print("arm cost by step count:")
    for s, c in table.items():
        print(f"  {s:5d} steps -> {c['total_h']:7.2f} h  "
              f"({c['eval_s']}s eval + {c['train_s']}s train)")

    out = {
        "device": d, "torch": torch.__version__,
        "eval": {"cases": len(td), "questions": nq, "seconds": round(eval_s, 1),
                 "q_per_s": round(nq / eval_s, 1), "batch": a.eval_batch,
                 "orders": n_orders, "roles": n_roles,
                 "in_distribution_questions": holdout_q, "guard_rows": guard_rows,
                 "per_arm_tax_s": round(eval_tax),
                 "note": ("one pass measured here; an arm pays it for 3 targets "
                          "x 2 option orders x 2 roles, which is ~3x the "
                          "single-pass number and dominates any short arm")},
        "train": {"seconds_per_step": round(step_s, 2), "batch": a.batch_size,
                  "measured_steps": a.train_steps,
                  "note": "tower training, gradient checkpointing on, fp32 on MPS"},
        "arm_cost": table,
        "measured_at": time.time(),
    }
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
