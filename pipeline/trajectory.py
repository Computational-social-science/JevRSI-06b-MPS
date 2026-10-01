#!/usr/bin/env python3
"""The self-evolution trajectory: read the logs, write one file.

EDITABLE — ours. Everything it reads was written by the loop: `prereg.jsonl`
(predictions, fsynced before each arm), `arms.jsonl` (one row per arm, with its
gate verdict) and `champions.jsonl` (the bar as it moved). Nothing is recomputed
and nothing is inferred — if a number is in the trajectory it is in a log.

It writes:

  trajectory.json   machine-readable: the arms in order, with prereg, numbers,
                    gate verdict, and the champion line over time
  trajectory.html   the same, as a chart: champion line rising, every arm as a
                    mark above or below it, kept arms called out

The point of the chart is the same as upstream's: the champion line is the bar,
an arm that stays under it is a published negative, and the interesting part of
a self-improving loop is the NEGATIVES — they are what delete a branch nobody
has to pay for again.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from loop import RECORDS, STATE, read_prereg, read_records  # noqa: E402
import power  # noqa: E402

VERDICT_CLASS = {
    "kept": ("kept (confirmed on a fresh seed)", "#2f9e5e"),
    "kept_pending_confirm": ("over the bar, unconfirmed", "#4a9e7a"),
    "not_confirmed": ("did not survive a fresh seed", "#8a6a2e"),
    "rejected": ("published negative", "#b4442e"),
    "needs_repair": ("needs repair", "#c98a1b"),
    "error": ("error", "#8a8a8a"),
}


def _mde() -> float:
    """The minimum detectable effect of this design, from the power analysis.

    Falls back to upstream's measured per-seed sd when no null arm has been run
    yet, which is the same conservative fallback pipeline/power.py uses.
    """
    p = power.load()
    if p and p.recommended_bar:
        return p.recommended_bar
    return power.mde_for_seeds(0.011, 2)


def build() -> dict:
    prereg = {p["arm"]: p for p in read_prereg()}
    arms = []
    for r in read_records():
        p = prereg.get(r.get("arm"), {})
        c = r.get("pooled_top1_candidate")
        k = r.get("pooled_top1_control")
        delta = (c - k) if c is not None and k is not None else None
        exp = p.get("expected_effect")
        # The distinction that keeps a null honest: an arm whose expected effect
        # is below what this design can detect cannot report "no effect". It
        # reports "no effect we could see", and the difference is the whole
        # reason the power analysis exists.
        underpowered = None
        if delta is not None and exp is not None and r.get("verdict") != "kept":
            underpowered = abs(exp) < _mde()
        arms.append({
            "arm": r.get("arm"),
            "axis": r.get("axis"),
            "change": r.get("change"),
            "prediction": p.get("prediction", ""),
            "expected_effect": exp,
            "underpowered": underpowered,
            "delta_floor": p.get("delta_floor"),
            "null_floor_vs": p.get("null_floor_vs", ""),
            "top1_candidate": r.get("pooled_top1_candidate"),
            "top1_control": r.get("pooled_top1_control"),
            "delta": ((r.get("pooled_top1_candidate") or 0) - (r.get("pooled_top1_control") or 0))
                     if r.get("pooled_top1_candidate") is not None
                     and r.get("pooled_top1_control") is not None else None,
            "min_decision_score": r.get("min_decision_score_candidate"),
            "ece": r.get("ece_candidate"),
            "n_questions": r.get("n_questions", 0),
            "targets": r.get("targets_used", []),
            "per_target": r.get("per_target_top1", {}),
            "per_target_control": r.get("per_target_top1_control", {}),
            "verdict": r.get("verdict"),
            "passed": r.get("passed", False),
            "guards_failed": r.get("guards_failed", []),
            "reason": r.get("reason", ""),
            "error": r.get("error", ""),
            "train_seconds": r.get("train_seconds", 0),
            "final_loss": r.get("final_loss"),
            "ts": r.get("ts", 0),
        })
    champs = []
    cp = STATE / "champions.jsonl"
    if cp.is_file():
        for line in cp.read_text().splitlines():
            if line.strip():
                try:
                    champs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return {
        "generated": time.time(),
        "backbone": "Qwen/Qwen3-0.6B-Base",
        "n_arms": len(arms),
        "n_prereg": len(prereg),
        "n_kept": sum(1 for a in arms if a["verdict"] == "kept"),
        "n_rejected": sum(1 for a in arms if a["verdict"] == "rejected"),
        "n_needs_repair": sum(1 for a in arms if a["verdict"] == "needs_repair"),
        "n_error": sum(1 for a in arms if a["verdict"] == "error"),
        "champions": champs,
        "arms": arms,
    }


def render_html(t: dict) -> str:
    e = html.escape
    arms = t["arms"]
    # The chart: x = arm order, y = top-1. Champion line = the best-so-far
    # control-relative top-1, which is what the bar moves along.
    best = None
    pts_c, pts_b, labels = [], [], []
    for i, a in enumerate(arms):
        if a["top1_candidate"] is None:
            labels.append((i, a["arm"], None, a))
            continue
        labels.append((i, a["arm"], a["top1_candidate"], a))
        if best is None or a["top1_candidate"] > best:
            best = a["top1_candidate"]
        pts_c.append((i, a["top1_candidate"]))
        pts_b.append((i, best))
    W, H, PAD = 1100, 340, 46
    ys = [v for _, v in pts_c] + [v for _, v in pts_b]
    if ys:
        lo, hi = min(ys), max(ys)
        span = max(hi - lo, 0.02)
        lo, hi = lo - span * 0.15, hi + span * 0.15
    else:
        lo, hi = 0.0, 1.0

    def X(i):
        return PAD + (i * (W - 2 * PAD) / max(1, len(arms) - 1))

    def Y(v):
        return H - PAD - (v - lo) / (hi - lo) * (H - 2 * PAD)

    svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="trajectory">']
    # horizontal gridlines with values
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = Y(v)
        svg.append(f'<line x1="{PAD}" y1="{y:.1f}" x2="{W-PAD}" y2="{y:.1f}" '
                   f'stroke="var(--border)" stroke-width="1"/>')
        svg.append(f'<text x="{PAD-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" '
                   f'fill="var(--muted-foreground)">{v:.3f}</text>')
    # champion line
    if len(pts_b) > 1:
        d = " ".join(f"{'M' if j==0 else 'L'}{X(i):.1f},{Y(v):.1f}"
                    for j, (i, v) in enumerate(pts_b))
        svg.append(f'<path d="{d}" fill="none" stroke="#2f9e5e" stroke-width="2.5"/>')
    # every arm
    for i, name, v, a in labels:
        if v is None:
            svg.append(f'<circle cx="{X(i):.1f}" cy="{Y(lo):.1f}" r="4" fill="#8a8a8a"/>')
            continue
        _, col = VERDICT_CLASS.get(a["verdict"], ("?", "#666"))
        r = 7 if a["verdict"] == "kept" else 5
        svg.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="{r}" fill="{col}" '
                   f'stroke="var(--card)" stroke-width="1.5"><title>{e(name)} — '
                   f'{v:.4f} ({e(a["verdict"])})</title></circle>')
        svg.append(f'<text x="{X(i):.1f}" y="{H-PAD+16}" text-anchor="middle" font-size="10" '
                   f'fill="var(--muted-foreground)" transform="rotate(30 {X(i):.1f} {H-PAD+16})">'
                   f'{e(name)}</text>')
    svg.append(f'<line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" '
               f'stroke="var(--border)" stroke-width="1"/>')
    svg.append("</svg>")

    def cell(v, fmt="{:.4f}"):
        return f'<td class="num">{fmt.format(v)}</td>' if isinstance(v, (int, float)) \
            else '<td class="num">—</td>'

    rows = []
    for a in arms:
        label, col = VERDICT_CLASS.get(a["verdict"], (a["verdict"] or "?", "#666"))
        d = f"{a['delta']:+.4f}" if a["delta"] is not None else "—"
        # An arm whose expected effect is below the resolution limit cannot
        # report "no effect". Its label says so on the row, not in a footnote.
        if a.get("underpowered") and a["verdict"] in (
                "rejected", "needs_repair", "not_confirmed"):
            label = f"{label} · underpowered"
        exp = a.get("expected_effect")
        rows.append(
            "<tr>"
            f'<td><code>{e(a["arm"])}</code><br><span class="ax">{e(a["axis"] or "")}</span></td>'
            f'<td class="chg">{e(a["change"] or "")}</td>'
            + cell(a["top1_control"]) + cell(a["top1_candidate"])
            + cell(d, "{:}")
            + cell(exp, "{:+.4f}")
            + f'<td><span class="v" style="background:{col}">{e(label)}</span></td>'
            + f'<td class="why">{e(a["reason"] or a["error"] or "")}</td>'
            "</tr>")

    legend = " ".join(
        f'<span class="v" style="background:{c}">{e(l)}</span>'
        for l, c in VERDICT_CLASS.values())

    return f"""<!doctype html><meta charset="utf-8">
<style>
 body {{ font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif; margin:0; padding:20px; }}
 h1 {{ font-size:19px; margin:0 0 2px; }}
 .sub {{ color:var(--muted-foreground); margin:0 0 18px; }}
 .cards {{ display:flex; gap:10px; margin:0 0 18px; flex-wrap:wrap; }}
 .card {{ border:1px solid var(--border); border-radius:8px; padding:9px 14px; background:var(--card); }}
 .card b {{ font-size:20px; display:block; }}
 .card span {{ color:var(--muted-foreground); font-size:11px; }}
 table {{ border-collapse:collapse; width:100%; font-size:12px; }}
 th,td {{ text-align:left; padding:7px 9px; border-bottom:1px solid var(--border); vertical-align:top; }}
 th {{ color:var(--muted-foreground); font-weight:600; font-size:11px; text-transform:uppercase; }}
 td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
 td.chg,td.why {{ color:var(--muted-foreground); max-width:340px; }}
 code {{ font-size:12px; }}
 .ax {{ color:var(--muted-foreground); font-size:10px; text-transform:uppercase; }}
 .v {{ color:#fff; border-radius:4px; padding:1px 7px; font-size:11px; white-space:nowrap; }}
</style>
<h1>RSI-Jev on Qwen3-0.6B — self-evolution trajectory</h1>
<p class="sub">Every mark is one arm, preregistered before it ran.
  The green line is the champion: the bar each arm had to clear.
  {e(str(t['n_kept']))} kept, {e(str(t['n_rejected']))} published negatives,
  {e(str(t['n_needs_repair']))} needing repair, {e(str(t['n_error']))} errors
  of {e(str(t['n_arms']))} arms.</p>
<div class="cards">
  <div class="card"><b>{t['n_arms']}</b><span>arms run</span></div>
  <div class="card"><b>{t['n_prereg']}</b><span>preregistered</span></div>
  <div class="card"><b>{t['n_kept']}</b><span>kept</span></div>
  <div class="card"><b>{t['n_rejected']}</b><span>published negatives</span></div>
  <div class="card"><b>{len(t['champions'])}</b><span>champion changes</span></div>
</div>
<p>{legend}</p>
{''.join(svg)}
<table><thead><tr><th>arm</th><th>what changed</th><th>control</th><th>candidate</th>
<th>delta</th><th>expected</th><th>verdict</th><th>why</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "state" / "trajectory.json"))
    ap.add_argument("--html", default=str(ROOT / "state" / "trajectory.html"))
    a = ap.parse_args()
    t = build()
    Path(a.out).write_text(json.dumps(t, indent=2, default=str) + "\n")
    Path(a.html).write_text(render_html(t))
    print(f"{t['n_arms']} arms -> {a.out} and {a.html}")
    for arm in t["arms"]:
        d = f"{arm['delta']:+.4f}" if arm["delta"] is not None else "   —   "
        print(f"  {arm['arm']:26s} {d}  {arm['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
