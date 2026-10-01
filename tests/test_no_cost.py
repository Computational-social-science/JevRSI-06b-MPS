#!/usr/bin/env python3
"""Rule 2, enforced: no paid API, no credential, no rented compute.

The rule is "avoid paid APIs and unnecessary cost". A promise in a document is
not enforcement, so this test makes it executable: it fails the build if our
code ever grows a reference to a metered endpoint or a credential, or if the
backbone stops being a free local model.

What counts as a violation:

  * a key-shaped literal (`sk-...`, `hf_...` with a real token length) — a
    committed credential is a cost AND a security problem, and it must not be
    possible to add one silently;
  * a reference to a metered inference provider -- OpenAI, Anthropic,
    OpenRouter, Together, Replicate, Groq, Cohere, or a Bedrock/Vertex/Azure
    OpenAI endpoint -- as a host, an SDK, or a credential-shaped token.
  * a non-HuggingFace network URL in our runtime code, which is where an
    accidental dependency on a paid service would enter;
  * a backbone that is not a known-free local model.

What is deliberately NOT a violation: `OPENAI_API_KEY` appearing in a COMMENT as
an example of what not to do, or the word "api" in a local module's name. The
test scans code, not prose, and it explains every hit rather than failing
silently.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OURS = ("pipeline", "scripts", "tests", "config")

# Free, local, open-weights backbones this project is allowed to run on.
FREE_BACKBONES = {
    "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base", "Qwen/Qwen3.5-0.8B-Base",
}

# `replicate` is deliberately NOT matched bare: "a null arm is a replicate of
# the recipe" is ordinary English, and a guard that fires on the word turns a
# scientist's vocabulary into a build failure. The paid service is Replicate THE
# COMPANY, which shows up as a host or an API token, so that is what is matched.
# The first version of this guard did match it bare and failed on a comment.
METERED = re.compile(
    r"\b(openai|anthropic|openrouter|cohere|perplexity|groq)\b"
    r"|\b(together|replicate)[._-]?(ai|api|com)\b"
    r"|api\.openai\.com|api\.anthropic\.com|openrouter\.ai/api"
    r"|bedrock-runtime|aiplatform\.googleapis"
    r"|\b[A-Z_]*(OPENAI|ANTHROPIC|OPENROUTER|REPLICATE|TOGETHER|GROQ|COHERE)"
    r"[A-Z_]*(KEY|TOKEN|SECRET)\b", re.I)
# A key-shaped literal, not the NAME of a key-shaped thing.
KEY_LITERAL = re.compile(r"sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{24,}|AKIA[0-9A-Z]{16}")
# Any non-HuggingFace http(s) URL in runtime code.
URL = re.compile(r"https?://([A-Za-z0-9.-]+)")
# The domestic mirror is the project's declared default (see `dev.hf_endpoint`),
# so it belongs here. Leaving it out would make the cost guard fire on the very
# download the standing rule requires — a guard that punishes compliance is
# worse than a guard that is merely absent, because it teaches the reader to
# disable it.
ALLOWED_HOSTS = {"huggingface.co", "hf.co", "hf-mirror.com",
                 "127.0.0.1", "localhost"}

# A URL that something would actually FETCH, not one that appears in prose. The
# distinction matters: the version card cites github.com in a markdown link and
# that is a reference in a document, not a dependency. Only a URL passed to a
# network call is a dependency, so that is the only thing banned.
NET_CALL = re.compile(
    r"(urlopen|requests\.(get|post|put|head)|httpx\.(get|post|Client)|"
    r"socket\.(connect|create_connection)|hf_hub_download|snapshot_download|"
    r"load_dataset|urlretrieve|curl\b|wget\b)")
URL_IN_CALL = re.compile(r"https?://([A-Za-z0-9.-]+)")

# Files allowed to contain the patterns they look for. NOT a second copy: the
# canonical list lives in `pipeline/detectors.py`, because two copies is one too
# many and they demonstrably drift. See that module for the four separate times
# this problem appeared and what each one cost.
sys.path.insert(0, str(ROOT / "pipeline"))
import detectors as _det  # noqa: E402

DETECTORS = set(_det.DETECTORS) | {Path(__file__).name}

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    # Skip build and dependency directories. `.gitignore` keeps them out of the
    # REPOSITORY, but this walks the FILESYSTEM, and `formal/.lake/` now holds
    # 7.6 GB of vendored Mathlib — so the guard was reading someone else's
    # source, which made it slow, made its findings depend on whether lake had
    # finished extracting, and would eventually flag Mathlib for a metered host.
    SKIP_DIRS = {".lake", ".git", "__pycache__", "node_modules", ".venv", "build"}
    files = [p for d in OURS for p in (ROOT / d).rglob("*")
             if p.is_file() and p.suffix in (".py", ".json", ".sh")
             and not (SKIP_DIRS & set(p.relative_to(ROOT).parts))]
    print(f"scanning {len(files)} files under {', '.join(OURS)}")

    hits_metered: list[str] = []
    hits_key: list[str] = []
    hits_url: list[str] = []
    for p in files:
        rel = p.relative_to(ROOT)
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            # A commented-out line cannot execute, and the test file itself
            # names these providers in order to ban them.
            if line.lstrip().startswith("#"):
                continue
            if p.name in DETECTORS:
                continue
            if METERED.search(line):
                hits_metered.append(f"{rel}:{i}: {line.strip()[:80]}")
            if KEY_LITERAL.search(line):
                hits_key.append(f"{rel}:{i}: KEY LITERAL")
            # Only a URL on a line that FETCHES is a dependency. A URL inside a
            # markdown citation, a docstring or a comment is a reference, and
            # banning those would forbid the record from naming its own source.
            if NET_CALL.search(line):
                for host in URL_IN_CALL.findall(line):
                    if host not in ALLOWED_HOSTS:
                        hits_url.append(f"{rel}:{i}: {host} (fetched)")

    check("no metered inference provider in our code", not hits_metered,
          "; ".join(hits_metered[:3]))
    check("no credential literal committed anywhere", not hits_key,
          "; ".join(hits_key[:3]))
    check("no network host other than HuggingFace (incl. the domestic mirror) "
          "and localhost", not hits_url,
          "; ".join(hits_url[:3]))

    cfg = ROOT / "config" / "run.json"
    if cfg.is_file():
        import json
        model = json.loads(cfg.read_text()).get("model", "")
        check("the configured backbone is free and local",
              model in FREE_BACKBONES, model or "(unset)")
    else:
        check("config/run.json exists", False)

    # No token in the environment is *required*: everything here works
    # anonymously, so a token must never be necessary.
    env_keys = [k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")
                if k in __import__("os").environ]
    check("no provider credential is even present in this environment",
          not env_keys, ", ".join(env_keys))

    print("\ncost ledger for one arm, measured on this machine:")
    cost = ROOT / "state" / "cost.json"
    if cost.is_file():
        import json
        c = json.loads(cost.read_text())
        ac = c["arm_cost"]["300"]
        print(f"  compute : local MPS, {ac['total_h']:.2f} h per arm at 300 steps")
        print(f"  money   : 0.00  (no rented GPU, no inference endpoint, no paid API)")
        print(f"  network : HuggingFace Hub, anonymous, for the open-weights backbone")
    else:
        print("  (state/cost.json absent — run pipeline/measure_cost.py)")

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
