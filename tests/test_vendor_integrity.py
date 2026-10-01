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

    print("\nour two numbers are the SAME MEASUREMENTS upstream reports")
    # The point of the partial suite is comparability, and comparability is a claim
    # that can rot silently: if our own loader drifted from upstream's suite loader,
    # every number this project has ever compared with a release table would be a
    # comparison between two different measurements, and nothing would say so.
    #
    # So it is asserted. The two benchmarks that need no local checkout are the
    # primary target and the general-knowledge guard -- between them, the whole
    # trade-off this design currently cannot see.
    try:
        sys.path.insert(0, str(ROOT))
        from rsijev import targets_suite as ts
        from rsijev.targets import load_typed_decisions
        got = ts.load_suite(["typed_decisions_test", "mmlu_pro_1k"],
                            decontam=False, suite="v3")
        ours = load_typed_decisions("test")
        n_ours = sum(len(getattr(c, "questions", []) or []) for c in ours)
        n_suite = sum(len(getattr(c, "questions", []) or []) for c in got["typed_decisions_test"])
        check("our primary target scores the same question count upstream's suite does",
              n_ours == n_suite == 2000, f"ours {n_ours}, upstream suite {n_suite}")
        check("... and the same number of cases",
              len(ours) == len(got["typed_decisions_test"]) == 400,
              f"ours {len(ours)}, upstream suite {len(got['typed_decisions_test'])}")
        w = ts.SUITES["v3"]
        cov = w["typed_decisions_test"] + w["mmlu_pro_1k"]
        guard_n = json.loads((ROOT / "config" / "run.json").read_text()) \
            .get("targets", {}).get("guard_n", "?")
        check("the two loadable benchmarks carry the trade-off, in upstream's weights",
              abs(cov - 0.281) < 1e-9,
              f"typed_decisions {w['typed_decisions_test']} + mmlu_pro {w['mmlu_pro_1k']} "
              f"= {cov:.3f} of the v3 suite — the primary target AND the guard")
        check("the guard is a SUBSAMPLE of the same benchmark, and says so",
              len(got["mmlu_pro_1k"]) == 1000,
              f"upstream scores 1,000 MMLU-Pro rows; this project scores {guard_n} of "
              f"them as a guard, so the two are the same benchmark at different "
              f"sample sizes")
    except Exception as exc:
        check("the partial suite loads", False, f"{type(exc).__name__}: {exc}")

    print("\nno source file points at a script that is not there")
    # Dangling path references are the failure mode of every rename, and they are
    # silent until something runs the line that contains them. This one bit: renaming
    # scripts/build_corpus.py to build_replication_corpus.py was done with a string
    # replace, which caught `"scripts/build_corpus.py"` written as a literal and
    # missed `"scripts" / "build_corpus.py"` written as a path join -- so
    # tests/test_sources.py raised FileNotFoundError every run for two hours, and a
    # test suite run that happened before the rename reported it green.
    #
    # A test that cannot find a file it was told to read is not a failing assertion,
    # it is a crash, and a crash in a suite is a line of output nobody reads. So the
    # references are checked here instead, where the check can be a check.
    import ast as _ast
    import re as _re

    # Vendored files are upstream's text, not ours: the manifest governs them, and a
    # path that does not resolve inside a file we are forbidden to edit is upstream's
    # business, not a defect here.
    vendored = set()
    _man = ROOT / "rsijev" / ".upstream_manifest.json"
    if _man.is_file():
        vendored = {p for p, r in json.loads(_man.read_text())["files"].items()
                    if r.get("status") == "vendored"}

    def _code_strings(path: Path) -> list[str]:
        """String CONSTANTS that are not docstrings -- i.e. strings code can use.

        Scanning raw text finds prose: the comment in this very file that names
        `scripts/build_corpus.py` to explain why it was renamed, and upstream's own
        messages inside vendored files. A check that cries wolf on a healthy tree is
        the fastest way to teach a reader to ignore the panel, so the scan is over
        the AST and docstrings are excluded -- a docstring cannot be a path.
        """
        out = []
        try:
            tree = _ast.parse(path.read_text())
        except (OSError, SyntaxError):
            return out
        docstrings = set()
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.Module, _ast.FunctionDef, _ast.AsyncFunctionDef,
                                 _ast.ClassDef)):
                d = _ast.get_docstring(node, clean=False)
                if d is not None:
                    docstrings.add(d)
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Constant) and isinstance(node.value, str):
                if node.value in docstrings:
                    continue
                out.append(node.value)
        return out

    # The path pattern, kept out of the f-string below so the quoting stays sane:
    # a character class ending in a quote character terminates a triple-quoted string.
    _PATH = _re.compile(r"(?:pipeline|scripts|rsijev)/[A-Za-z0-9_./-]+\.(?:py|sh|json|lean)")
    _MODULE = ("contract", "targets", "targets_suite", "metrics", "evaluate",
               "encode", "arch", "data", "train", "calibrate", "fit", "rl2")
    dangling: list[str] = []
    for d in ("pipeline", "scripts", "tests"):
        for p in sorted((ROOT / d).rglob("*")):
            if p.suffix not in (".py", ".sh") or not p.is_file():
                continue
            if str(p.relative_to(ROOT)) in vendored:
                continue
            texts = _code_strings(p) if p.suffix == ".py" else [p.read_text()]
            for s in texts:
                for m in _PATH.finditer(s):
                    rel = m.group(0)
                    if rel.startswith("rsijev/") and rel[len("rsijev/"):-3] in _MODULE:
                        continue          # a MODULE path, imported rather than opened
                    if not (ROOT / rel).exists():
                        dangling.append(f"{p.name} -> {rel}")
    check("no source file references a path that does not exist", not dangling,
          "; ".join(sorted(set(dangling))[:4]) or
          "a rename that misses a path-join form leaves a test crashing on every run")

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
