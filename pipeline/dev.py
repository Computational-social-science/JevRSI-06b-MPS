"""Device selection. EDITABLE (our adaptation, not upstream).

Upstream hardcodes `dev = "cuda"` in its entry scripts (`release_train.py`,
`suite.py`, `load_release.py`) and takes `device` as a parameter everywhere in
the library, so the whole library is already device-agnostic. This module is the
one place that decides which device a run gets — the entire adaptation.

Precedence, and why:

  1. `RSIJEV_DEVICE` env var — an explicit override always wins, so a launchd
     job and an interactive shell can disagree about the device without a code
     edit.
  2. CUDA when available. The published numbers were produced there and every
     record carries a `linear_attn_kernel` stamp saying so.
  3. MPS on Apple Silicon. Numerically NOT the same stack as CUDA — different
     kernels, different reduction order — so a record is stamped with the device
     and must not be pooled with a CUDA one inside one comparison. That is
     upstream's own rule, applied honestly rather than by hoping MPS "counts".
  4. CPU. Always available.

Dtype follows from the device: CUDA gets bfloat16, the published stack; MPS and
CPU get float32. bf16-on-MPS is supported but has a different reduction order
from bf16-on-CUDA, so enabling it would create a third numeric stack rather than
a second. At 0.6B the cost of fp32 on a 64 GB machine is small.

Everything else upstream already does — the kernel stamp is `run_arm_lib`'s own
`LINEAR_ATTN_KERNEL`, the text hidden size is `run_arm_lib.text_hidden_size`, and
the decoder-layer list is `arch._decoder_layers`. None of it is reimplemented
here.
"""
from __future__ import annotations

import json
import os

import torch

# --- where this project downloads from ---------------------------------------
# A standing project rule: pull HuggingFace artefacts through the domestic
# mirror, not through huggingface.co directly.
#
# This lives here rather than in each entry point because `dev` is imported by
# every one of them — `arm`, `daemon`, `run_one`, `verify_ckpt`, the tests — so
# this is the single place where the fact is stated and the single place it can
# be read from. Setting it in the launchd plists as well is belt and braces, not
# the mechanism: launchd does not inherit a shell's environment, so a rule
# applied only in the plist would silently not apply to a hand-run arm, a test,
# or a future entry point somebody writes. One definition, honoured everywhere.
#
# An explicit `HF_ENDPOINT` in the environment still wins, so a one-off run
# against the origin remains possible without editing a file — and the record
# says which endpoint was used, so a number fetched from one is never silently
# pooled with a number fetched from the other.
_MIRROR = "https://hf-mirror.com"


def hf_endpoint() -> str:
    """The endpoint in force, and apply the project default if none is set."""
    cur = os.environ.get("HF_ENDPOINT", "").strip()
    if not cur:
        os.environ["HF_ENDPOINT"] = _MIRROR
        cur = _MIRROR
    # Read by huggingface_hub at import time in some versions, so it is set as
    # early as this module is imported rather than at first use.
    return cur


def describe_sources() -> dict:
    """What this run will download from. Recorded, so a reader can tell."""
    cfg = {}
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "config", "run.json")
    if os.path.isfile(p):
        try:
            cfg = json.load(open(p))
        except (OSError, json.JSONDecodeError):
            cfg = {}
    return {
        "hf_endpoint": hf_endpoint(),
        "hf_endpoint_default": _MIRROR,
        "hf_endpoint_is_default": os.environ.get("HF_ENDPOINT", "") == _MIRROR,
        "corpus_builder": cfg.get("corpus_builder", "scripts/build_corpus.py"),
        "model": cfg.get("model", ""),
    }


hf_endpoint()   # applied on import, deliberately


def device() -> str:
    """'cuda' | 'mps' | 'cpu'."""
    forced = os.environ.get("RSIJEV_DEVICE", "").strip()
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def autocast_enabled(device_str: str) -> bool:
    """bf16 autocast is worth it on CUDA and MPS; on CPU it is not."""
    return device_str in ("cuda", "mps")


def kernel_stamp() -> str:
    """Device + torch, appended to upstream's own `LINEAR_ATTN_KERNEL`.

    Upstream stamps the linear-attention kernel and the torch version, which
    separates CUDA from CPU for it. It has no MPS (Apple Silicon) runs to
    separate, so the device goes here. Appending rather than replacing keeps
    upstream's stamp intact and comparable.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    try:
        from run_arm_lib import LINEAR_ATTN_KERNEL
        base = LINEAR_ATTN_KERNEL
    except Exception:
        base = f"torch-reference/torch-{torch.__version__}"
    return f"{base}/{device()}"


def load_base_model(model_id: str, device_str: str | None = None, *, dtype=None):
    """Load a causal LM + tokenizer on the chosen device, weights frozen.

    The freeze is not cosmetic: upstream's `LogprobReadout` — the zero-shot
    control every arm is measured against — freezes every parameter of the model
    it is handed, and `run_arm` relies on the resident model staying frozen
    between arms (it checksums the resident tower before and after a
    tower-training arm and raises if it moved).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device_str or device()
    tok = AutoTokenizer.from_pretrained(model_id)
    if dtype is None:
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(dev).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    return lm, tok
