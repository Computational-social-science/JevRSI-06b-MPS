#!/usr/bin/env python3
"""A live dashboard for the loop. stdlib only, on purpose.

EDITABLE — ours.

Why `http.server` and not Flask: this project spends nothing (rule 2) and the
dashboard is part of the project's own evidence, so it should have no
dependencies to install, pin, patch or keep current. `http.server` plus
`ThreadingHTTPServer` is enough for a page that reads JSON once every few
seconds, and the absence of a framework is one fewer thing that can be a
dependency or a CVE.

What "live" means here, precisely, because a dashboard that lies is worse than
none:

  * **No caching anywhere.** Every request re-reads the logs from disk. There is
    no in-memory copy that can go stale, so what the page shows is what is on
    disk at that moment.
  * **Staleness is shown, not hidden.** The page carries the mtime of the arm log
    and the age of the last record. A loop that died two hours ago and one that
    is mid-arm look different, because the page says which.
  * **The heartbeat is a first-class field.** Upstream's arm runner is silent
    during the control pass — legitimately, for tens of minutes — so silence is
    NOT evidence of death and the page must not imply it is. It shows the last
    heartbeat line verbatim and the elapsed time, and leaves the judgement to
    the deadline the watchdog enforces.
  * **A failed arm is a normal state, not an error state.** The loop's exit
    codes say an error is a successful iteration. The page colours it as a
    result, not a fault.

Endpoints:
  /              the page
  /api/status    everything, as JSON, re-read per request
  /api/trajectory the arm log
  /healthz       200 if the process is up
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

LOGS = ROOT / "logs"
RECORDS = ROOT / "records"
STATE = ROOT / "state"
PORT = int(os.environ.get("RSIJEV_DASH_PORT", "8788"))
LABEL = "com.research.rsijev.dash"


# ----------------------------------------------------------------- collectors
def _lines(p: Path, n: int = 0) -> list[str]:
    try:
        out = [x for x in p.read_text(errors="ignore").splitlines() if x.strip()]
    except Exception:
        return []
    return out[-n:] if n else out


def _jsonl(p: Path) -> list[dict]:
    out = []
    for line in _lines(p):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _age(p: Path) -> float | None:
    try:
        return round(time.time() - p.stat().st_mtime, 1)
    except Exception:
        return None


def _running(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern],
                          capture_output=True).returncode == 0


def _has_launchd() -> bool:
    return "com.research.rsijev" in subprocess.run(
        ["launchctl", "list"], capture_output=True, text=True).stdout


def _sleep_prevented() -> bool:
    return re.search(r"PreventSystemSleep\s+1",
                     subprocess.run(["pmset", "-g", "assertions"],
                                    capture_output=True, text=True).stdout) is not None


def collect() -> dict:
    """Everything the page needs, re-read from disk on every call."""
    import agenda
    import confirm as confirmmod
    import held_out
    import power
    from loop import read_prereg, read_records

    arms_raw = read_records()
    prereg = {p["arm"]: p for p in read_prereg()}
    champs = _jsonl(STATE / "champions.jsonl")
    champ = None
    if (STATE / "champion.json").is_file():
        champ = json.loads((STATE / "champion.json").read_text())
    pw = power.load()
    pend = confirmmod.pending()
    ho = held_out.status()

    # The arm currently in flight, from the most recent arm log.
    arm_logs = sorted(LOGS.glob("arm_*.log"), key=lambda p: p.stat().st_mtime,
                      reverse=True)
    current = None
    if arm_logs:
        lg = arm_logs[0]
        m = re.search(r"=== ARM (\S+) \((\w+)\)", lg.read_text(errors="ignore")[-4000:])
        if m:
            current = {"arm": m.group(1), "axis": m.group(2),
                       "log": lg.name, "idle_s": _age(lg)}

    # The multi-agent diagnosis. Read from disk each poll; if it has never run,
    # say so rather than implying the pipeline is checked when it is not.
    doc = None
    _dp = ROOT / "state" / "doctor.json"
    if _dp.is_file():
        try:
            doc = json.loads(_dp.read_text())
        except json.JSONDecodeError:
            doc = {"verdict": "unreadable", "ok": False,
                   "diagnosis": "doctor.json could not be parsed"}

    # Where this run downloads from. A number fetched from the origin and a
    # number fetched from the mirror are the same model, but a reader cannot
    # verify that from the score — so the endpoint travels with the run.
    try:
        import dev as _dev
        sources = _dev.describe_sources()
    except Exception as exc:
        sources = {"hf_endpoint": f"unknown ({type(exc).__name__})"}

    # The climb, and what the weights cost. `lineage` is the module that
    # distinguishes a surviving offspring from a discarded one, so the dashboard
    # asks it rather than re-deriving survival from verdict strings — two
    # versions of the same classification is two things to disagree.
    try:
        import lineage as _lin
        lineage = _lin.series()
        _c = _lin.collect(apply=False)
        lineage["ckpt_gb"] = _c["freed_bytes"] / 1e9 + sum(
            a.ckpt_bytes for a in _lin.lineage()) / 1e9
        lineage["collectable_gb"] = _c["freed_gb"]
    except Exception as exc:
        lineage = {"series": [], "n_keep": 0, "n_discard": 0, "n_crash": 0,
                   "error": f"{type(exc).__name__}: {exc}"}

    arms = []
    for r in arms_raw:
        p = prereg.get(r.get("arm"), {})
        c, k = r.get("pooled_top1_candidate"), r.get("pooled_top1_control")
        delta = (c - k) if c is not None and k is not None else None
        exp = p.get("expected_effect")
        under = None
        if (pw and delta is not None and r.get("verdict") != "kept"
                and isinstance(exp, (int, float)) and exp):
            under = abs(exp) < pw.recommended_bar
        arms.append({
            "arm": r.get("arm"), "axis": r.get("axis"), "change": r.get("change"),
            "prediction": p.get("prediction", ""), "expected_effect": exp,
            "underpowered": under, "delta": delta,
            "top1_candidate": c, "top1_control": k,
            "min_decision_score": r.get("min_decision_score_candidate"),
            "ece": r.get("ece_candidate"),
            "n_questions": r.get("n_questions", 0),
            "targets": r.get("targets_used", []),
            "per_target": r.get("per_target_top1", {}),
            "verdict": r.get("verdict"), "reason": r.get("reason", ""),
            "role": r.get("role", ""), "seeds_done": r.get("seeds_done", 1),
            "seeds_required": r.get("seeds_required", 1),
            "champion_delta": r.get("champion_delta"),
            "champion_arm": r.get("champion_arm", ""),
            "error": r.get("error", ""), "train_seconds": r.get("train_seconds", 0),
            "ts": r.get("ts", 0),
        })

    progress = 0.0
    if arms:
        progress = 100.0 * sum(1 for a in arms if a["verdict"] not in (None, "error")) \
            / len(agenda.AGENDA)

    # A champion with no record is a state-file inconsistency, not a result.
    # The page says so rather than showing a number that cannot be traced to an
    # arm -- a dashboard that renders a retracted measurement as live is worse
    # than one that shows nothing.
    inconsistent = None
    if champ and not any(a["arm"] == champ.get("arm") for a in arms):
        inconsistent = (f"champion.json names {champ.get('arm')!r} but that arm has no "
                        f"record in arms.jsonl; the bar is being measured against a "
                        f"retracted or unrecorded number")

    return {
        "now": time.time(),
        "now_h": time.strftime("%Y-%m-%d %H:%M:%S"),
        "health": {
            "loop_installed": _has_launchd(),
            "arm_running": _running("run_one.py"),
            "watchdog": _running("watchdog.py"),
            "caffeinate": _running("caffeinate"),
            "sleep_prevented": _sleep_prevented(),
            "records_age_s": _age(RECORDS / "arms.jsonl"),
            "daemon_log_age_s": _age(LOGS / "daemon.log"),
        },
        "current_arm": current,
        "doctor": doc,
        "sources": sources,
        "lineage": lineage,
        "progress_pct": round(progress, 1),
        "n_arms_done": len(arms),
        "n_arms_total": len(agenda.AGENDA),
        "n_prereg": len(prereg),
        "verdicts": {v: sum(1 for a in arms if a["verdict"] == v)
                     for v in ("kept", "kept_pending_confirm", "not_confirmed",
                               "rejected", "needs_repair", "error")},
        "power": ({
            "n_null_arms": pw.n_null_arms,
            "paired_sd": pw.paired_sd,
            "noise_sd": pw.noise_sd,
            "measured": pw.n_null_arms >= 2,
            "bar": pw.recommended_bar,
            # What doing nothing buys, and which rule turned that into the bar.
            # Both are on the panel because the bar is the number a reader is most
            # likely to accept without checking, and it is the number that was
            # wrong on 2026-10-01: +0.0450 while every null arm cleared it.
            "do_nothing_mean": pw.do_nothing_mean,
            "bar_rule": pw.bar_rule,
            "bar_rationale": pw.bar_rationale,
            "null_deltas": pw.null_deltas,
            "planned_seeds": pw.planned_seeds,
            "mde": pw.mde,
            "detectable": pw.detectable,
        } if pw else None),
        "pending_confirm": pend,
        "held_out": ho,
        "champion": champ,
        "inconsistent": inconsistent,
        "champions": champs,
        "arms": arms,
        "agenda": [{"name": h.name, "axis": h.axis, "change": h.change,
                    "expected_effect": h.expected_effect, "role": h.role,
                    "done": h.name in {a["arm"] for a in arms}}
                   for h in agenda.AGENDA],
        "daemon_tail": _lines(LOGS / "daemon.log", 14),
        "watchdog_tail": _lines(LOGS / "watchdog.log", 1),
        "arm_tail": _lines(LOGS / (arm_logs[0].name if arm_logs else Path("x")), 8)
        if arm_logs else [],
    }


# -------------------------------------------------------------------- serving
PAGE = Path(__file__).resolve().parent / "dashboard.html"

MIME = {".html": "text/html; charset=utf-8", ".json": "application/json; charset=utf-8"}


def jsonable(obj):
    """Make a structure safe for `json.dumps`, and safe for a real parser.

    Python's json module writes bare `Infinity` and `NaN` for non-finite floats,
    which is NOT valid JSON -- `JSON.parse` throws on it, and so does every
    conformant client. This bit the snapshot first (`const SNAPSHOT = {...}`)
    and it would have bitten the live `/api/status` identically, because
    `default=str` only rescues types json cannot serialise at all, not a float
    that happens to be infinite.

    And the value is not meaningless: `mde_for_seeds` returns `inf` for a
    one-seed design, which means "not measurable at any budget" -- that is
    ABSENT, not a number, so `null` is the honest encoding.

    `allow_nan=False` is left on deliberately: if a non-finite value ever
    survives this walk, the writer raises instead of emitting a document no
    parser can read.
    """
    import math
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def dumps(obj) -> str:
    return json.dumps(jsonable(obj), allow_nan=False, default=str)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):        # silence per-request logging
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), MIME[".html"])
            elif self.path.startswith("/api/status"):
                self._send(200, dumps(collect()).encode(), MIME[".json"])
            elif self.path == "/healthz":
                self._send(200, b'{"ok":true}', MIME[".json"])
            else:
                self._send(404, b'{"error":"not found"}', MIME[".json"])
        except Exception as exc:      # a broken dashboard must say so, not hang
            self._send(500, dumps(
                {"error": f"{type(exc).__name__}: {exc}"}).encode(), MIME[".json"])


def main() -> int:
    if not PAGE.is_file():
        print(f"missing {PAGE}", file=sys.stderr)
        return 1
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    srv.daemon_threads = True
    print(f"dashboard  http://127.0.0.1:{PORT}", flush=True)
    print(f"loop       {sum(1 for _ in (LOGS / 'daemon.log').open()) if (LOGS/'daemon.log').is_file() else 0}"
          f" daemon log lines", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
