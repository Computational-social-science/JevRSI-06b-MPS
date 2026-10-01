"""The trajectory: a CLIMB, not a scatter of attempts.

## The correction that shaped this figure

The first version of this page drew the obvious thing — a point per arm, coloured
by outcome, which is what a progress figure for an experiment log looks like. It
was wrong about what this project is doing, and the user was right to correct it.

autoresearch edits a file and asks "is this number better than the one I had?"; its
curve is a band of attempts, and a flat band is a normal result. This pipeline
does something else. A **keep can only move the incumbent UP** — that is not a
convention, it is `clearsBar_implies_strictly_better`, proved in `formal/Gate.lean`
for an arbitrary ordered group: an arm that clears the bar is strictly better than
the champion, because the bar is `champion + floor` and `floor > 0`. So the
trajectory is not a band of attempts. **It is a climb**, and it cannot descend.

That is the figure's whole claim, and it is worth drawing explicitly rather than
leaving the reader to infer it from a scatter:

  * the staircase IS the champion, rising only where an offspring survived;
  * a flat stretch is the search stuck, which is information, not an absence;
  * discards are drawn at the height they REACHED, so "we got this close and it
    was not enough" is legible;
  * a crash is at the floor, because scoring it zero would corrupt the axis.

The three-way keep/discard/crash distinction is borrowed in idea from karpathy's
`autoresearch` and credited in `lineage.py`, which also states exactly what was
NOT taken. Nothing here is adapted from that project — the renderer, the schema
and the encoding are this pipeline's own, because the questions are different.

Three things this figure has to make true at a glance, and one thing it must not
do:

  1. A KEEP is a surviving offspring. Its weights are the deliverable and the
     incumbent moves.
  2. A DISCARD is a measurement. The arm ran, produced a valid number, and the
     number said "not better than the parent". It is drawn as a point, not as an
     absence — the gap between "we tried it and it lost" and "we never tried it"
     is the whole content of an honest search record.
  3. A CRASH is the absence of a measurement, and is drawn as such, at the floor,
     because pretending it scored zero would corrupt the axis.
  4. It must NOT smooth. A running-best line is the single most flattering way to
     draw a search, and it hides every discard by construction — the discards are
     the results.

The lineage is a single line of descent, so a discarded offspring has no
descendants; confirmation runs are folded into the offspring they test rather
than drawn as competitors, because a confirmation is a second MEASUREMENT of one
candidate, not a new candidate.
"""
from __future__ import annotations

import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import lineage as lin  # noqa: E402

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Trajectory</title>
<style>
  :root{
    --bg:#ffffff; --fg:#16191d; --mut:#5c6570; --bd:#d9dde3; --grid:#eceef1;
    --keep:#0f7a4d; --disc:#5c6570; --crash:#a32b1c; --bar:#a32b1c;
  }
  @media (prefers-color-scheme: dark){
    :root{ --bg:#121417; --fg:#e8eaed; --mut:#8b95a1; --bd:#2a2e34; --grid:#22262b;
           --keep:#5cc48c; --disc:#8b95a1; --crash:#f08a76; --bar:#f08a76; }
  }
  *{box-sizing:border-box}
  body{margin:0;padding:28px;background:var(--bg);color:var(--fg);
       font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
       -webkit-font-smoothing:antialiased; font-variant-numeric:tabular-nums}
  h1{font-size:18px;font-weight:600;margin:0 0 3px;letter-spacing:-.01em}
  h2{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;
     color:var(--mut);margin:26px 0 10px}
  .sub{font-size:12px;color:var(--mut);margin-bottom:22px;max-width:78ch}
  .tally{display:flex;gap:1px;background:var(--bd);border:1px solid var(--bd);
         border-radius:6px;overflow:hidden;max-width:560px;margin:0 0 20px}
  .tally div{background:var(--bg);padding:10px 14px;flex:1}
  .tally .n{font-size:21px;font-weight:600;line-height:1.1}
  .tally .k{font-size:10px;text-transform:uppercase;letter-spacing:.06em;
            color:var(--mut);font-weight:600;margin-top:3px}
  .n.keep{color:var(--keep)} .n.disc{color:var(--disc)} .n.crash{color:var(--crash)}
  svg{display:block;width:100%;height:auto}
  .axis{stroke:var(--bd);stroke-width:1}
  .grid{stroke:var(--grid);stroke-width:1}
  text{font-family:ui-sans-serif,system-ui,sans-serif}
  .tick{font-size:10px;fill:var(--mut)}
  .lab{font-size:10.5px;fill:var(--fg);font-weight:600}
  .pt{stroke:var(--bg);stroke-width:1.5}
  .legend{display:flex;gap:18px;flex-wrap:wrap;font-size:11px;color:var(--mut);
          align-items:center;margin-top:12px}
  .legend i{display:inline-block;width:10px;height:10px;border-radius:50%;
            margin-right:6px;vertical-align:-1px}
  table{width:100%;border-collapse:collapse;font-size:12px;max-width:920px}
  th{font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);
     text-align:left;font-weight:600;padding:4px 8px 6px;border-bottom:1px solid var(--bd)}
  td{padding:6px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
  td.num{text-align:right;white-space:nowrap}
  code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px}
  .note{font-size:11px;color:var(--mut);margin-top:10px;max-width:82ch;line-height:1.6}
  .empty{padding:40px 20px;text-align:center;border:1px dashed var(--bd);
         border-radius:6px;color:var(--mut)}
</style>
<h1>The climb</h1>
<div class="sub">The champion over the arm sequence. The staircase rises only where an
offspring <b>survived</b>, and it cannot fall — an arm that clears the bar is strictly
better than the champion by construction, so a search that loses ground has a bug rather
than a bad week. Every arm is drawn too: a <b>discard</b> is a measurement, not an absence
(it ran, and the number said it was not better than its parent), and a <b>crash</b> is
the absence of a measurement, drawn at the floor because scoring it zero would corrupt
the axis.</div>
<div id="app"></div>
<script>
const DATA = __DATA__;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const sgn = v => (typeof v === "number" && isFinite(v)) ? (v>=0?"+":"")+v.toFixed(4) : "—";

function figure(){
  const s = DATA.series;
  const scored = s.filter(r => typeof r.delta === "number" && r.delta !== null);
  if (!scored.length) return `<div class="empty"><b>No arm has been scored yet.</b>
    <div class="note" style="margin:8px auto 0">The four calibration replicas run first —
    their spread is the noise floor every later arm is judged against, so there is nothing
    to plot until it exists.</div></div>`;

  const W = 980, H = 340, P = {l:56, r:20, t:16, b:74};
  const real = scored.map(r => r.delta);
  let lo = Math.min.apply(null, real.concat([0]));
  let hi = Math.max.apply(null, real.concat([0]));
  const pad = Math.max((hi - lo) * 0.18, 0.004);
  lo -= pad; hi += pad;
  const floor = lo - pad * 1.4;

  const all = s.map((r, i) => ({ r: r, i: i }));
  const n = all.length;
  const X = i => P.l + (n < 2 ? 0 : i * (W - P.l - P.r) / (n - 1));
  const Y = v => (H - P.b) - ((v - floor) / (hi - floor)) * (H - P.t - P.b);

  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Trajectory: keeps, discards and crashes by arm">`;
  for (let k = 0; k <= 4; k++){
    const v = floor + (hi - floor) * k / 4, y = Y(v);
    g += `<line class="grid" x1="${P.l}" y1="${y.toFixed(1)}" x2="${W-P.r}" y2="${y.toFixed(1)}"/>`;
    g += `<text class="tick" x="${P.l-8}" y="${(y+3.4).toFixed(1)}" text-anchor="end">${v.toFixed(3)}</text>`;
  }
  if (DATA.bar != null){
    const by = Y(DATA.bar);
    g += `<line x1="${P.l}" y1="${by.toFixed(1)}" x2="${W-P.r}" y2="${by.toFixed(1)}"
           stroke="var(--bar)" stroke-width="1.5" stroke-dasharray="5 3"/>`;
    g += `<text class="tick" x="${P.l+5}" y="${(by-5).toFixed(1)}" style="fill:var(--bar)">bar +${DATA.bar.toFixed(4)}</text>`;
  }
  if (lo <= 0 && hi >= 0){
    const z = Y(0);
    g += `<line x1="${P.l}" y1="${z.toFixed(1)}" x2="${W-P.r}" y2="${z.toFixed(1)}"
           stroke="var(--mut)" stroke-width="1" stroke-dasharray="2 3" opacity=".5"/>`;
    g += `<text class="tick" x="${W-P.r}" y="${(z-5).toFixed(1)}" text-anchor="end">no change vs control</text>`;
  }

  // THE CLIMB. A staircase through the champion, rising ONLY at a keep, and
  // therefore incapable of descending. That is not a drawing convention: it is
  // `clearsBar_implies_strictly_better` from formal/Gate.lean, proved for an
  // arbitrary ordered group. An arm that clears the bar is strictly better than
  // the champion, because the bar is `champion + floor` with `floor > 0`.
  //
  // Drawn as a staircase rather than a smooth best-so-far line, because the flat
  // stretches ARE the result: a plateau is the search stuck at that level for
  // that many arms, and a smoothed line would hide exactly that.
  let best = null;
  const steps = [];      // [x, y, arm] at each change of level
  let level = null, levelArm = null, flatFrom = 0;
  all.forEach(o => {
    const d = o.r.delta;
    if (o.r.keep && typeof d === "number" && (best === null || d > best)) {
      if (level !== null) steps.push([X(flatFrom), level, levelArm, o.i - flatFrom]);
      best = d; level = d; levelArm = o.r.arm; flatFrom = o.i;
    }
  });
  if (level !== null) steps.push([X(flatFrom), level, levelArm, all.length - flatFrom]);
  else if (steps.length === 0 && scored.length) {
    // No survivor yet: the climb has not started, and saying so is the point.
    best = Math.max.apply(null, scored.map(r => r.delta));
  }

  if (steps.length){
    // Filled band under the climb, so the accumulated height reads as a hill.
    let d = `M ${X(0).toFixed(1)} ${(H-P.b).toFixed(1)}`;
    steps.forEach((st, k) => {
      d += ` L ${st[0].toFixed(1)} ${Y(st[1]).toFixed(1)}`;
      d += ` L ${X(all.length-1).toFixed(1)} ${Y(st[1]).toFixed(1)}`;
      if (k < steps.length-1) d += ` L ${X(all.length-1).toFixed(1)} ${Y(steps[k+1][1]).toFixed(1)}`;
    });
    d += ` L ${X(all.length-1).toFixed(1)} ${(H-P.b).toFixed(1)} Z`;
    g += `<path d="${d}" fill="var(--keep)" opacity="0.10"/>`;
    let line = `M ${steps[0][0].toFixed(1)} ${Y(steps[0][1]).toFixed(1)}`;
    steps.forEach((st, k) => {
      line += ` L ${X(all.length-1).toFixed(1)} ${Y(st[1]).toFixed(1)}`;
      if (k < steps.length-1) line += ` L ${X(all.length-1).toFixed(1)} ${Y(steps[k+1][1]).toFixed(1)}`;
    });
    g += `<path d="${line}" fill="none" stroke="var(--keep)" stroke-width="2.5"
           stroke-linejoin="round"/>`;
    // Label each rise with the arm that caused it, and each plateau with how
    // long it lasted — the plateau length is the most informative number here.
    steps.forEach(st => {
      if (st[3] > 1) g += `<text class="tick" x="${(X(all.length-1)).toFixed(1)}"
          y="${(Y(st[1])-7).toFixed(1)}" text-anchor="end">flat for ${st[3]} arm${st[3]===1?"":"s"}</text>`;
    });
  }

  all.forEach(o => {
    const r = o.r;
    const crash = r.crash || typeof r.delta !== "number";
    const y = crash ? Y(floor) : Y(r.delta);
    const col = r.keep ? "var(--keep)" : (crash ? "var(--crash)" : "var(--disc)");
    g += `<circle class="pt" cx="${X(o.i).toFixed(1)}" cy="${y.toFixed(1)}" r="${r.keep ? 6.5 : 5}" fill="${col}">` +
         `<title>${esc(r.arm)} — ${crash ? "crash: no measurement" : sgn(r.delta)} (${r.keep ? "keep" : "discard"})</title></circle>`;
    const lbl = r.arm.length > 16 ? r.arm.slice(0, 15) + "…" : r.arm;
    g += `<text class="tick" x="${X(o.i).toFixed(1)}" y="${H-P.b+14}" text-anchor="end" ` +
         `transform="rotate(-45 ${X(o.i).toFixed(1)} ${H-P.b+14})">${esc(lbl)}</text>`;
  });

  g += `<line class="axis" x1="${P.l}" y1="${H-P.b}" x2="${W-P.r}" y2="${H-P.b}"/>`;
  g += `<text class="lab" x="${P.l}" y="${H-22}">arms, in the order they ran</text>`;
  if (n) g += `<text class="lab" x="${W-P.r}" y="${H-22}" text-anchor="end">${n} arm${n===1?"":"s"}</text>`;
  g += `<text class="lab" x="4" y="${P.t+2}">Δ top-1 vs the control</text></svg>`;

  return g + `<div class="legend">
      <span><i style="background:var(--keep)"></i>keep — a surviving offspring; the climb rises</span>
      <span><i style="background:var(--disc)"></i>discard — ran, measured, did not beat the parent</span>
      <span><i style="background:var(--crash)"></i>crash — no measurement</span>
      <span><i style="background:var(--keep);border-radius:0;height:3px;width:18px;vertical-align:3px"></i>the climb: the champion, and it cannot descend</span>
    </div>
    <p class="note">The climb cannot go down, and that is a proved property rather than a
    drawing choice: an arm that clears the bar is strictly better than the champion, because the
    bar is <code>champion + floor</code> with <code>floor &gt; 0</code>
    (<code>formal/Gate.lean</code>, <code>clearsBar_implies_strictly_better</code>, proved for an
    arbitrary ordered group). A flat stretch is therefore the search genuinely stuck at that level,
    which is why each plateau is labelled with its length.</p>`;
}

function rows(){
  return DATA.series.map((r, i) => {
    const conf = (r.confirmations || []).map(c =>
      `<div class="note" style="margin:2px 0 0">${c.keep ? "confirmed" : "failed"} on a fresh seed: ${esc(c.arm)} ${sgn(c.delta)}</div>`).join("");
    return `<tr>
      <td class="num">${i+1}</td>
      <td><code>${esc(r.arm)}</code>${conf}</td>
      <td>${r.role ? `<span class="note" style="margin:0">${esc(r.role)}</span>` : ""}</td>
      <td class="num">${(r.crash || typeof r.delta !== "number") ? "—" : sgn(r.delta)}</td>
      <td>${r.keep ? '<b style="color:var(--keep)">keep</b>'
                 : (r.crash ? '<span style="color:var(--crash)">crash</span>'
                            : '<span style="color:var(--disc)">discard</span>')}</td>
    </tr>`;
  }).join("");
}

// The headline numbers are about the CLIMB, not the attempt count. "How high did
// it get, and how much did it cost" is the question; "how many arms ran" is not.
const climbed = DATA.series.filter(r => r.keep);
const gain = (DATA.start != null && DATA.top != null) ? DATA.top - DATA.start : null;
$("#app").innerHTML =
  `<div class="tally">
     <div><div class="n keep">${DATA.n_keep}</div><div class="k">survivors</div></div>
     <div><div class="n">${gain == null ? "—" : (gain>=0?"+":"") + gain.toFixed(4)}</div><div class="k">climb gained</div></div>
     <div><div class="n">${DATA.n_arms ? (DATA.n_keep/DATA.n_arms*100).toFixed(0) : "0"}<span style="font-size:12px">%</span></div><div class="k">arms that survived</div></div>
     <div><div class="n disc">${DATA.n_discard}</div><div class="k">discards</div></div>
     <div><div class="n crash">${DATA.n_crash}</div><div class="k">crashes</div></div>
     <div><div class="n">${DATA.ckpt_gb.toFixed(2)}<span style="font-size:12px"> GB</span></div><div class="k">weights held</div></div>
   </div>
   <h2>Climb</h2>${figure()}
   <h2>Every arm</h2>
   <table><thead><tr><th class="num">#</th><th>arm</th><th>role</th>
     <th class="num">Δ vs control</th><th>outcome</th></tr></thead><tbody>${rows()}</tbody></table>
   <p class="note">Weights are lineage, not evidence. An arm keeps its checkpoint only while it is
   the surviving offspring; a discarded arm's measurement stays in the record permanently and its
   bytes are collected, so the loop cannot leak ~1.7 GB per arm for models that can never be
   published. Generated ${esc(DATA.generated_h)}.</p>`;
</script>
"""


def build(out: Path) -> Path:
    s = lin.series()
    held = 0
    if (ROOT / "ckpt").is_dir():
        held = sum(f.stat().st_size for f in (ROOT / "ckpt").rglob("*") if f.is_file())
    bar = None
    try:
        bar = json.loads((ROOT / "state" / "power.json").read_text()).get("recommended_bar")
    except (OSError, json.JSONDecodeError):
        pass
    # `start` is the first measured arm and `top` the best ever reached, so the
    # headline gain is the height of the climb rather than a sum of deltas — a
    # search that went up 0.03 then down 0.01 and up 0.04 has climbed 0.04.
    deltas = [r["delta"] for r in s["series"]
              if isinstance(r.get("delta"), (int, float))]
    data = {"series": s["series"], "n_keep": s["n_keep"],
            "n_discard": s["n_discard"], "n_crash": s["n_crash"],
            "n_arms": len(s["series"]),
            "start": deltas[0] if deltas else None,
            "top": max(deltas) if deltas else None,
            "ckpt_gb": held / 1e9, "bar": bar,
            "generated_h": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.write_text(PAGE.replace("__DATA__", json.dumps(data, allow_nan=False, default=str)))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "state" / "trajectory.html"))
    a = ap.parse_args()
    p = build(Path(a.out))
    s = lin.series()
    print(f"wrote {p}  ({p.stat().st_size//1024} KB, "
          f"keep {s['n_keep']} / discard {s['n_discard']} / crash {s['n_crash']})")
