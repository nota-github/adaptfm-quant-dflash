"""YAML config loader for the server.

The server expects a config like:

    engine: vllm
    model: Qwen/Qwen3.5-4B
    quant: null               # or "awq", "gptq-4bit", ...
    vllm_args:
      max_model_len: 8704     # 8192 + 512 headroom
      enforce_eager: false
      max_num_seqs: 1
      dtype: auto

`load_config` is intentionally permissive: a missing path or missing keys
fall back to sane defaults so the test stub path requires no config file.
"""

from __future__ import annotations

import os
from typing import Any


_DEFAULT: dict[str, Any] = {
    "engine": "vllm",
    "model": "Qwen/Qwen3.5-4B",
    "quant": None,
    "vllm_args": {
        "max_model_len": 8704,  # 8192 prompt + 512 output headroom
        "enforce_eager": False,
        "max_num_seqs": 1,
        "dtype": "auto",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Copy-merge, recursing into nested dicts (never aliases `base` innards)."""
    out = {k: (_deep_merge(v, {}) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None) -> dict:
    """Load YAML config from `path`, merged onto defaults.

    Returns the default config when `path` is None, empty, or missing.
    """
    if not path or not os.path.isfile(path):
        # Return a deep copy of defaults so callers can't mutate the module-level dict.
        return _deep_merge(_DEFAULT, {})

    import yaml  # local import: keep top-level fast and dep-light

    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        return _deep_merge(_DEFAULT, {})
    return _deep_merge(_DEFAULT, loaded)
