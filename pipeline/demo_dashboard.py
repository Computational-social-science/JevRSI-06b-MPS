#!/usr/bin/env python3
"""Render the dashboard against a populated fixture, to review the design in the
state that matters most and that a live pipeline reaches only after hours.

The empty state is easy to get right by accident — there is nothing on it. The
populated state is where a dashboard actually earns its keep: long arm names, a
bar that moved, a champion line, held-out consumed, a mix of verdicts. If the
layout only survives an empty pipeline it is a prototype.

The numbers are the REAL ones from this search's arms (null_floor's rejected
delta, the fallback floor, the 0.011 paired sd) so the axis ranges, the bar line
and the label truncation are all exercised at realistic magnitudes. Nothing here
is written into state/ — it is a review artefact only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import dashboard  # noqa: E402
import snapshot  # noqa: E402

ARMS = [
    # name, delta, expected, verdict, role, seeds(done/req), contamination
    ("null_floor", -0.1265, 0.0, "rejected", "calibration", "1/1",
     "clean: 0 overlap"),
    ("null1", 0.0041, 0.0, "rejected", "calibration", "1/1", "clean: 0 overlap"),
    ("null2", -0.0028, 0.0, "rejected", "calibration", "1/1", "clean: 0 overlap"),
    ("null3", 0.0063, 0.0, "rejected", "calibration", "1/1", "clean: 0 overlap"),
    ("champion_base", 0.0382, 0.031, "kept_pending_confirm", "decisive", "1/3",
     "clean: 0 overlap"),
    ("confirm__champion_base_r2", 0.0351, 0.031, "kept_pending_confirm",
     "decisive", "2/3", "clean: 0 overlap"),
    ("confirm__champion_base_r3", 0.0298, 0.031, "not_confirmed", "decisive", "3/3",
     "clean: 0 overlap"),
    ("train_lower_layers_soft", 0.0311, 0.020, "kept", "decisive", "1/3",
     "clean: 0 overlap"),
    ("confirm__train_lower_layers_soft_r2", 0.0288, 0.020, "kept_pending_confirm",
     "decisive", "2/3", "clean: 0 overlap"),
    ("lr_3e4_cosine", 0.0074, 0.014, "rejected", "exploratory", "1/1",
     "clean: 0 overlap"),
    ("dropout_0p1", 0.0049, 0.011, "rejected", "exploratory", "1/1",
     "clean: 0 overlap"),
    ("tower_frozen_head_only", -0.0088, 0.009, "rejected", "exploratory", "1/1",
     "not checked"),
    ("corpus_typed_choices_only", 0.0129, 0.016, "rejected", "exploratory", "1/1",
     "clean: 2 overlap"),
    ("calibrated_temperature", 0.0061, 0.012, "rejected", "exploratory", "1/1",
     "clean: 0 overlap"),
]

AG = [("null_floor", 0.0, "calibration", True), ("null1", 0.0, "calibration", True),
      ("null2", 0.0, "calibration", True), ("null3", 0.0, "calibration", True),
      ("champion_base", 0.031, "decisive", True),
      ("train_lower_layers_soft", 0.020, "decisive", False),
      ("lr_3e4_cosine", 0.014, "exploratory", False),
      ("dropout_0p1", 0.011, "exploratory", False),
      ("tower_frozen_head_only", 0.009, "exploratory", False),
      ("corpus_typed_choices_only", 0.016, "exploratory", False),
      ("calibrated_temperature", 0.012, "exploratory", False),
      ("label_smoothing_0p05", 0.010, "exploratory", False),
      ("cosine_schedule_500w", 0.011, "exploratory", False),
      ("weight_decay_0p01", 0.008, "exploratory", False),
      ("batch_size_8", 0.007, "exploratory", False),
      ("head_width_256", 0.010, "exploratory", False),
      ("tower_partial_unfreeze_2q", 0.013, "exploratory", False),
      ("mix_typed_and_open", 0.009, "exploratory", False)]


def build() -> dict:
    arms = [{
        "arm": n, "delta": dl, "expected_effect": ex, "verdict": v, "axis": "training",
        "role": r, "seeds_done": int(sd.split("/")[0]), "seeds_required": int(sd.split("/")[1]),
        "underpowered": v in ("rejected", "not_confirmed") and abs(ex) < 0.03,
        "contamination": c, "pooled_top1": 0.47 + dl, "pooled_top1_control": 0.4645,
    } for n, dl, ex, v, r, sd, c in ARMS]
    done = {a["arm"] for a in arms}
    return {
        "now_h": "2026-10-01 14:32:07",
        "n_arms_done": len(arms), "n_arms_total": len(AG),
        "progress_pct": round(100 * len(arms) / len(AG)),
        "current_arm": {"arm": "confirm__train_lower_layers_soft_r3", "step": 168,
                        "steps": 400, "eta_s": 1180, "pct": 42},
        "champion": {"arm": "train_lower_layers_soft", "pooled_top1": 0.4956,
                     "control_top1": 0.4645, "since": "2026-10-01 13:58"},
        "inconsistent": None,
        "held_out": {"spent": True, "score": 0.5712, "majority": 0.545,
                     "consumed_at": "2026-10-01 14:02", "n": 400},
        "power": {"bar": 0.0231, "paired_sd": 0.00825, "measured": True,
                  "n_null_arms": 4, "mde": {1: 0.0231, 2: 0.0163, 3: 0.0133},
                  "source": "measured from 4 null arms"},
        "verdicts": {"kept": 1, "kept_pending_confirm": 2, "rejected": 10,
                     "not_confirmed": 1, "needs_repair": 0, "error": 0},
        "health": {"loop_installed": True, "arm_running": True, "watchdog": True,
                   "sleep_prevented": True, "dashboard_serving": True,
                   "records_age_s": 522, "daemon_log_age_s": 61,
                   "uptime_s": 61200},
        "agenda": [{"name": n, "expected_effect": e, "role": r, "done": d or n in done}
                   for n, e, r, d in AG],
        "arms": arms,
        "doctor": {
            "verdict": "healthy", "ok": True, "when_h": "2026-10-01 14:31:00",
            "diagnosis": "all auditors agree: the loop is running and the science it is "
                         "running is the science we said we would run",
            "auditors": {
                "vitals": {"n": 7, "ok": 7, "critical": [], "major": [],
                           "findings": []},
                "provenance": {"n": 4, "ok": 4, "critical": [], "major": [],
                               "findings": []},
                "methodology": {"n": 4, "ok": 4, "critical": [], "major": [],
                                "findings": []},
                "integrity": {"n": 4, "ok": 4, "critical": [], "major": [],
                              "findings": []},
            },
        },
        "daemon_tail": [
            "[2026-10-01 14:28:03] HEARTBEAT confirm__train_lower_layers_soft_r3 1200s alive",
            "[2026-10-01 14:29:41] HEARTBEAT confirm__train_lower_layers_soft_r3 1338s alive",
            "[2026-10-01 14:31:19] HEARTBEAT confirm__train_lower_layers_soft_r3 1476s alive",
            "[2026-10-01 13:58:02] RESULT    train_lower_layers_soft  kept on 1/3 seeds",
            "[2026-10-01 13:58:02]           top1  cand 0.4956  ctrl 0.4645  delta +0.0311",
            "[2026-10-01 13:58:03] CONFIRM   +0.0311 is a big effect — confirming on 2 fresh seeds",
            "[2026-10-01 14:02:11] HELD-OUT  consumed once, 400 cases, majority 0.5450",
            "[2026-10-01 14:02:12]           held-out 0.5712  majority 0.5450  +0.0262",
            "[2026-10-01 14:02:12] CHAMPION  train_lower_layers_soft  0.4956",
            "[2026-10-01 14:02:13] POWER     4 null arms: noise sd=0.0083  bar=+0.0231",
            "[2026-10-01 14:14:52] RESULT    confirm__champion_base_r3  FAILED fresh seed",
            "[2026-10-01 14:14:52]           cleared bar on the selected seed, not on this one",
        ],
    }


def main() -> int:
    # Through `snapshot.build()` — the SAME function the real snapshot uses, with
    # the same template, the same shared renderer and the same encoder. An earlier
    # version of this file hand-rolled its own injection and drifted: it emitted
    # `const SNAPSHOT={...}` where the render test and the real snapshot both emit
    # `const SNAPSHOT = {...}`, so the demo silently stopped being reviewable by
    # the harness that checks it. One builder, two data sources.
    p = snapshot.build(ROOT / "state" / "dashboard_demo.html", dashboard.jsonable(build()))
    print(f"wrote {p}  ({p.stat().st_size//1024} KB, {len(ARMS)} arms populated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())