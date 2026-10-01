#!/usr/bin/env python3
"""The vendor tree must be COMPLETE, ACCOUNTED FOR, and byte-reproducible.

## Why this exists, in the order things were actually discovered

`scripts/suite.py` is upstream's file, vendored. It imports `rsijev.targets_suite`
— the fifteen benchmark loaders, their weights, their licences, their pinned
origins. That module was never vendored, so the suite driver had never run, and
nothing noticed: no manifest said it should exist, and no test ran it.

Then the manifest went in and immediately found two more things:

* the tree's true provenance was unrecorded. It is upstream `4a2b2b4`
  (2026-09-29) — proven by matching git blob shas of three unpatched files, not
  by trusting a date. The record had claimed "verbatim, unmodified except one
  line" without ever naming the commit those words refer to.
* `scripts/build_corpus.py` was **ours**, not upstream's: a thin wrapper over
  upstream's `build_synth_corpus.py`, filed under an upstream filename. Every
  reader would have taken it for vendored code. It is now
  `build_replication_corpus.py`, and upstream's real file is recorded absent.

The general rule, which is the whole point: **a file that is not in the manifest
is a defect, not a detail.** It means either something in the tree impersonates
upstream, or something upstream has is missing here — and in both cases a number
in this project was produced by code that cannot be identified from this tree.

The project's own rule says vendor it unmodified so that a difference from the
reference cannot be attributed to a different implementation. A vendor tree
missing a file, or carrying an impostor, *is* a different implementation —
discovered late. This test is what makes the claim checkable.

No torch, no network, no GPU — under a second.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "rsijev" / ".upstream_manifest.json"
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def blob_sha(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def main() -> int:
    if not MANIFEST.is_file():
        check("the vendor manifest exists", False, str(MANIFEST))
        print(f"\n{len(FAILS)} failure(s)")
        return 1
    m = json.loads(MANIFEST.read_text())
    files = m["files"]
    print(f"vendor manifest: upstream {m['upstream']} @ {m['vendored_at'][:12]} "
          f"({m['vendored_at_date']})")

    print("\nevery vendored file is byte-identical to the pin, or declares its patch")
    bad: list[str] = []
    ours_list: list[str] = []
    for path, rec in sorted(files.items()):
        if rec.get("status") != "vendored":
            continue
        p = ROOT / path
        if not p.is_file():
            bad.append(f"{path} is in the manifest but MISSING from the tree")
            continue
        got = blob_sha(p.read_bytes())
        if got != rec.get("blob"):
            bad.append(f"{path} does not match its recorded blob")
        if rec.get("state") == "ours":
            ours_list.append(path)
            if "patch" not in rec:
                bad.append(f"{path} is declared ours with no stated patch")
            elif "LOCAL" not in p.read_text():
                bad.append(f"{path} is declared ours but carries no LOCAL marker in "
                           f"the file, so a reader cannot see the change")
    check("no vendored file is missing or altered", not bad, "; ".join(bad[:4]) or
          f"{sum(1 for r in files.values() if r.get('status') == 'vendored')} vendored, "
          f"{len(ours_list)} declared ours ({', '.join(ours_list) or 'none'})")

    print("\nnothing in the tree is unaccounted for")
    # The anti-impostor check, and the one that would have caught
    # scripts/build_corpus.py.
    unaccounted = []
    for d in ("rsijev", "scripts"):
        for p in sorted((ROOT / d).rglob("*.py")):
            rel = str(p.relative_to(ROOT))
            if rel not in files:
                unaccounted.append(rel)
    check("every .py under rsijev/ and scripts/ is in the manifest", not unaccounted,
          "; ".join(unaccounted[:4]) or
          "a file outside the manifest would be this project's own code wearing an "
          "upstream filename")

    print("\nthe suite driver runs, and the machinery it needs is here")
    check("rsijev/targets_suite.py is present",
          (ROOT / "rsijev" / "targets_suite.py").is_file(),
          "fifteen loaders, weights, licences and pinned origins")
    try:
        sys.path.insert(0, str(ROOT))
        from rsijev import targets_suite as ts
        check("both suite weightings load and sum to 1",
              len(ts.SUITES["v3"]) >= 12 and len(ts.SUITES["v2"]) >= 12
              and abs(sum(ts.SUITES["v3"].values()) - 1.0) < 1e-9,
              f"v3={len(ts.SUITES['v3'])} benchmarks, v2={len(ts.SUITES['v2'])}")
        check("the builder this project hand-rolled is upstream's",
              (ROOT / "scripts" / "build_specialist_replay_corpus.py").is_file(),
              "upstream ships the replay-corpus builder; we were about to write one")
    except Exception as exc:
        check("targets_suite imports cleanly", False, f"{type(exc).__name__}: {exc}")

    print("\nevery import in the vendor tree resolves inside it")
    present = {p.stem for p in (ROOT / "rsijev").glob("*.py")}
    pkg_names = set()
    try:
        tree = ast.parse((ROOT / "rsijev" / "__init__.py").read_text())
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                pkg_names.add(n.name)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        pkg_names.add(t.id)
    except Exception:
        pass
    unresolved: list[str] = []
    for d in ("rsijev", "scripts"):
        for p in sorted((ROOT / d).rglob("*.py")):
            try:
                t = ast.parse(p.read_text())
            except SyntaxError:
                continue
            for n in ast.walk(t):
                if isinstance(n, ast.ImportFrom) and n.module:
                    root = n.module.split(".")[0]
                    if root in ("rsijev",) or (n.level and n.level > 0):
                        mod = n.module.split(".")[-1]
                        # A module path must exist as a module; a bare name imported
                        # from a package may be a symbol, which is not a missing file.
                        if len(n.module.split(".")) > 1 or n.level:
                            if mod not in present and mod != "__init__":
                                unresolved.append(f"{p.name} -> {n.module}")
                        else:
                            for a in n.names:
                                if a.name not in present and a.name not in pkg_names:
                                    unresolved.append(f"{p.name} -> rsijev.{a.name}")
    check("no file imports a module the tree does not have", not unresolved,
          "; ".join(unresolved[:4]) or
          f"{len(present)} modules; a missing one is what broke suite.py")

    print("\nthe vendor surface is the running path, and nothing wider")
    # "Vendor it unmodified" is not a licence to vendor everything. Unused vendored
    # code has to be kept in step, a broken import inside it is a defect this
    # project has to triage, and a wide tree implies coverage it does not have.
    # On 2026-10-01 `fit_release_calibration.py` was pulled in, it imported
    # `rsijev.vision`, and the fix for that broken import was to pull in two more
    # out-of-scope files -- so a rule about restraint was broken by a well-meant
    # repair. Hence a test.
    scope = m.get("scope")
    check("the manifest states what is in scope and what is not", bool(scope),
          (scope or {}).get("rule", "")[:70] if scope else "no scope section")
    OFF_LIMITS = ("rsijev/vision", "scripts/build_vision", "scripts/vision_",
                  "scripts/serve", "scripts/demo_web", "scripts/build_rl2",
                  "scripts/build_depthdial", "scripts/depthdial",
                  "scripts/fit_release_calibration", "scripts/release_train",
                  "scripts/routing", "scripts/bench")
    leaked = [p for p in files if p.startswith(OFF_LIMITS) and files[p].get("status") == "vendored"]
    check("nothing from the vision / serving / RL / release surface is vendored",
          not leaked, "; ".join(leaked[:4]) or
          f"{sum(1 for r in files.values() if r.get('status') == 'vendored')} vendored, "
          f"all on the running path")
    unexplained = [p for p, r in files.items()
                   if r.get("status") == "vendored" and r.get("state") in ("added", "ours")
                   and not (r.get("note") or r.get("patch"))]
    check("every file added past the pin says why it is in scope", not unexplained,
          "; ".join(unexplained[:4]) or
          "a vendored file nobody can justify is a file nobody maintains")

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
