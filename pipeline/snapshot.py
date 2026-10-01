#!/usr/bin/env python3
"""Render the dashboard to a single self-contained HTML file.

EDITABLE — ours.

Why this exists when there is already a server: the same page has to work in
three places, and they do not have the same network reachability.

  * served at http://127.0.0.1:8788 — the loopback case, live;
  * opened as a `file://` snapshot — anywhere else, and in a preview frame that
    cannot reach the host's loopback at all;
  * attached to a message — a file, not a port.

So the page is built once and used in all three: the live page fetches
`api/status` on a timer, and a snapshot carries the same JSON inline and skips
the fetch. One renderer, so a snapshot cannot show something the live page would
not.

Every snapshot is stamped with when the data was read, because a dashboard that
looks live but is a week old is worse than one that says it is a week old.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import dashboard  # noqa: E402

SNAP = """<!doctype html>
<meta charset="utf-8">
<title>RSI-Jev · Qwen3-0.6B (snapshot)</title>
<style>%STYLE%</style>
<div id="app"><p class="mut">loading snapshot…</p></div>
<script>
const SNAPSHOT = %DATA%;
const LIVE = false;
%s
</script>
"""

# The fetch/tick tail of the live page, minus the network. Kept here rather than
# duplicated so the snapshot and the live page cannot drift apart.
OFFLINE_JS = """
function tick(){
  if(!SNAPSHOT){ location.reload(); return; }
  render(SNAPSHOT);
}
tick();
"""


def build(out: Path, data: dict | None = None) -> Path:
    page = (ROOT / "pipeline" / "dashboard.html").read_text()
    style = page.split("<style>", 1)[1].split("</style>", 1)[0]

    # Everything up to and including the live tick, so the renderers are shared.
    live_js = page.split("<script>", 1)[1].split("</script>", 1)[0]
    live_js = live_js.split("async function tick()")[0].rstrip()

    d = data if data is not None else dashboard.collect()
    d = dict(d)
    d["now_h"] = time.strftime("%Y-%m-%d %H:%M:%S") + "  (snapshot)"
    d["snapshot_taken"] = time.time()

    html = SNAP.replace("%STYLE%", style)
    # The SAME encoder the live API uses. Two encoders is how a snapshot and a
    # served page start disagreeing, and the shared renderer is pointless if the
    # data beside it is written two different ways.
    html = html.replace("%DATA%", dashboard.dumps(d))
    html = html.replace("%s", live_js + "\n" + OFFLINE_JS)
    out.write_text(html)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "state" / "dashboard_snapshot.html"))
    a = ap.parse_args()
    p = build(Path(a.out))
    import dashboard as _d
    n = _d.collect()["n_arms_done"]
    print(f"wrote {p}  ({p.stat().st_size/1024:.0f} KB, {n} arms at snapshot time)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
