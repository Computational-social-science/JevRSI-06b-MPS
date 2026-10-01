#!/usr/bin/env python3
"""Two standing rules, enforced rather than remembered.

1. **Research process and results documentation is in English, comments
   included.** Not a style preference: this repository is read by people who do
   not read Chinese, and a protocol that is only legible to its author is not a
   protocol. It is also the rule that decays fastest, because a comment written
   in the wrong language is invisible to the person who wrote it and invisible to
   every reviewer who skips it.

   Vendored upstream code under `rsijev/` is exempt, and must stay exempt.
   Upstream is vendored verbatim so the adaptation can be diffed against the
   thing it adapts; "helpfully" translating a vendored file would break the one
   property that makes vendoring worth doing. The exemption is by path, and this
   test asserts the exemption is still narrow.

2. **Paths in code are relative, and the repository follows GitHub
   conventions.** A hardcoded `/Users/<someone>/...` makes the repository
   unrunnable everywhere else, and it is a real failure rather than a theoretical
   one: the sibling JevRSI project records exactly this — a document survived a
   pure-text purity scan while a file hardcoded a machine path, and nothing
   stopped it. A guard that exists and has never fired is a comment, so both
   halves here are negative-controlled: each plants a violation and requires the
   check to catch it.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Upstream, vendored verbatim. Not ours to restyle.
VENDORED = ("rsijev/",)

# Directories whose contents are the project's own words. `rsijev/` is excluded.
OURS = ("pipeline", "tests", "scripts", "formal", "docs", "config")

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def tracked() -> list[str]:
    r = subprocess.run(["git", "-C", str(ROOT), "ls-files"],
                       capture_output=True, text=True)
    return [f for f in r.stdout.splitlines() if f.strip()]


def is_vendored(rel: str) -> bool:
    return rel.startswith(VENDORED)


def self_excluded(rel: str) -> bool:
    """Is this file allowed to contain what it searches for?

    From the shared list, not a local one. This is the FIFTH time in this project
    that a checker has fired on its own detection literals — after `replicate` in
    a comment, the snapshot's bare `Infinity`, I12 on `invariants.py`, I12 on
    `publish.py`, and now this file's own planted fixtures. Five times is not bad
    luck; it is a structural property of writing a checker, and the fix is one
    shared declaration rather than a fifth local exemption. See
    `pipeline/detectors.py`.
    """
    sys.path.insert(0, str(ROOT / "pipeline"))
    import detectors
    return detectors.is_detector(Path(rel).name)


def cjk(s: str) -> bool:
    """CJK ideographs, kana, and hangul — the ranges a Chinese/Japanese/Korean
    document would use. Punctuation is excluded on purpose: a full-width comma
    in an otherwise English file is not what this rule is about."""
    for ch in s:
        o = ord(ch)
        if (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
                or 0x3040 <= o <= 0x30FF or 0xAC00 <= o <= 0xD7AF):
            return True
    return False


def main() -> int:
    files = tracked()
    check("the repository has tracked files to check", bool(files), f"{len(files)} files")

    print("\n1. documentation and comments are in English")
    offenders: list[str] = []
    for rel in files:
        if (is_vendored(rel) or self_excluded(rel)
                or rel.startswith("state/") or rel.startswith("records/")):
            continue
        if Path(rel).suffix not in (".py", ".md", ".sh", ".lean", ".json", ".txt", ".html"):
            continue
        try:
            for i, line in enumerate(Path(ROOT / rel).read_text(errors="ignore").splitlines(), 1):
                if cjk(line):
                    offenders.append(f"{rel}:{i}")
                    break
        except OSError:
            continue
    check("no CJK text in our own files", not offenders, str(offenders[:4]))
    check("the vendored exemption is still narrow",
          VENDORED == ("rsijev/",), str(VENDORED))

    # Negative control: the detector must actually detect. Otherwise "no CJK
    # found" is indistinguishable from "the check does not work".
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "probe.py"
        f.write_text('"""这是一个中文文档字符串。"""\n')
        found = any(cjk(l) for l in f.read_text().splitlines())
    check("the CJK detector fires on real CJK text", found)

    print("\n2. paths in code are relative")
    # A machine-specific path, not a URL and not a glob.
    ABS = re.compile(r"""(?<![\w.])/(?:Users|home)/[A-Za-z0-9._-]+/""")
    hits: list[str] = []
    for rel in files:
        if (is_vendored(rel) or self_excluded(rel)
                or rel.startswith("state/") or rel.startswith("records/")):
            continue
        if Path(rel).suffix not in (".py", ".sh", ".lean", ".json", ".html"):
            continue
        try:
            for i, line in enumerate(Path(ROOT / rel).read_text(errors="ignore").splitlines(), 1):
                # `~/` is home-RELATIVE and portable; `/Users/` and `/home/`
                # name one machine. Docs may mention a path while explaining why
                # there are none, so prose in docs/ is exempt but code is not.
                if ABS.search(line) and not (rel.startswith("docs/") and line.lstrip().startswith(("#", "*", "-"))):
                    hits.append(f"{rel}:{i}")
        except OSError:
            continue
    check("no machine-specific absolute path in code or config", not hits,
          str(hits[:4]))

    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "probe.py"
        f.write_text('ROOT = "/Users/someone/research/x"\n')
        found = bool(ABS.search(f.read_text()))
        f2 = Path(td) / "ok.py"
        f2.write_text('ROOT = Path(__file__).resolve().parent.parent\nH = Path.home()\n')
        clean = not ABS.search(f2.read_text())
    check("the absolute-path detector fires on a machine path", found)
    check("... and accepts the relative and home-relative idioms", clean,
          "Path(__file__) and Path.home() are the portable forms")

    print("\n3. GitHub conventions the reproducibility claim depends on")
    have = set(files)
    for f in ("LICENSE", "README.md", "requirements.txt", ".gitignore", ".python-version"):
        check(f"{f} is present", f in have)
    req = (ROOT / "requirements.txt")
    if req.is_file():
        text = req.read_text()
        pins = [l for l in text.splitlines()
                if l.strip() and not l.strip().startswith("#")]
        check("every dependency is pinned with ==, not a range",
              bool(pins) and all("==" in p for p in pins),
              f"{len(pins)} pins, {sum('==' not in p for p in pins)} unpinned")
        check("the pins match what this search actually ran on",
              "torch==" in text and "transformers==" in text)
    lic = (ROOT / "LICENSE")
    if lic.is_file():
        check("the licence is MIT, as the README claims",
              "MIT License" in lic.read_text()[:200])
        check("it carries a copyright line",
              "Copyright (c)" in lic.read_text())

    print("\n4. the .gitignore keeps the large and the unpublishable out")
    ig = (ROOT / ".gitignore")
    if ig.is_file():
        text = ig.read_text()
        for pat, why in ((".venv/", "dependencies"), ("ckpt/", "weights"),
                         ("corpus/", "regenerable data"),
                         ("records/*/*.items.jsonl", "per-question rows")):
            check(f"excludes {pat}  ({why})", pat in text)

    print("\n5. the vendored tree is still verbatim")
    # "Unchanged SINCE IT WAS VENDORED", not "never touched". The commit that
    # added the vendored tree necessarily touches it, so the first version of this
    # check reported a violation on a repository whose vendoring was perfect.
    root = subprocess.run(["git", "-C", str(ROOT), "rev-list", "--max-parents=0", "HEAD"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    root = root[0] if root else None
    if not root:
        check("the vendored tree is unchanged since vendoring", False, "no root commit")
    else:
        r = subprocess.run(
            ["git", "-C", str(ROOT), "log", "--oneline", f"{root}..HEAD", "--", "rsijev/"],
            capture_output=True, text=True)
        after = [l for l in r.stdout.splitlines() if l.strip()]
        check("no commit since vendoring has touched the vendored tree", not after,
              f"{len(after)} commit(s) — vendoring is only worth doing if the "
              f"vendored code is left alone")
        # And confirm the exemption did not quietly widen: the vendored files
        # must still be non-empty, because an empty directory passes "unchanged".
        n = len([f for f in files if is_vendored(f)])
        check("the vendored tree is actually present", n >= 5, f"{n} vendored files")

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
