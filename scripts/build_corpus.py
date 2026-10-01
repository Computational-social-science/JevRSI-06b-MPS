#!/usr/bin/env python3
"""Build the training corpus. EDITABLE (data axis).

Thin wrapper over upstream's `scripts/build_synth_corpus.py`, which is the code
that produced v1.0's corpus: it converts `n4ze3m/typed-decisions-synth` into
Cases, takes gold from the TEACHER's soft distribution rather than the argmax
label, drops any state that collides with typed-decisions' own states, and
quarantines the nine keyword-ADJACENT domains into a separate file so an arm can
leave them out and show the result does not depend on near-workflow data.

Upstream's own note, which is the reason the corpus is built at all rather than
scraped: "Gold is the TEACHER's soft distribution (DeepSeek, sampled x3 and
averaged), the same construction as typed-decisions' own gold, not the argmax
label." A decision model fitted to a hard label throws away the teacher's
uncertainty, and on typed-decisions 61% of the split is below 0.67 max-prob and
carries 75% of the errors.

We add two things:

  * `--max-cases`, so a 24/7 loop on a laptop can start on a subset and grow the
    corpus later. A subset is a weaker claim and the record says which was used.
  * a manifest with sha256 per file, so a corpus can be reproduced and a result
    can name the bytes it was trained on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "corpus"))
    ap.add_argument("--max-cases", type=int, default=0,
                    help="0 = all. Truncates synth.jsonl for a fast first loop.")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if (out / "synth.jsonl").is_file():
        print(f"corpus already present in {out}; rebuilding is opt-in with --force")
        return 0

    print("building synth corpus via upstream scripts/build_synth_corpus.py ...")
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_synth_corpus.py"), "--out", str(out)],
        cwd=str(ROOT))
    if r.returncode != 0:
        print(f"upstream builder failed with {r.returncode}", file=sys.stderr)
        return r.returncode

    if a.max_cases:
        p = out / "synth.jsonl"
        lines = p.read_text().splitlines()
        keep = lines[:a.max_cases]
        p.write_text("\n".join(keep) + "\n")
        print(f"truncated synth.jsonl to {len(keep)} cases (--max-cases)")

    manifest = {}
    for f in sorted(out.glob("*.jsonl")):
        n = sum(1 for line in f.read_text().splitlines() if line.strip())
        manifest[f.name] = {"cases": n, "sha256": sha256(f), "bytes": f.stat().st_size}
        print(f"  {f.name}: {n} cases  {f.stat().st_size/1e6:.1f} MB  sha {manifest[f.name]['sha256'][:16]}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
