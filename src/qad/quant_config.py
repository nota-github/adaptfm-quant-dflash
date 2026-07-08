#!/usr/bin/env python3
"""qad_quant_config.py — nvidia-modelopt INT4 g=32 sym config matching
the **deployed** cyankiwi/Qwen3.5-4B-AWQ-4bit safetensors.

IMPORTANT — recipe.yaml vs deployed-safetensors divergence.
The recipe.yaml published on the cyankiwi HF Hub repo claims:

    ignore:
      - re:.*embed_tokens
      - re:.*linear_attn.*
      - re:model[.]visual.*
      - re:mtp.*
      - lm_head

But the *deployed* `model.safetensors.index.json` shows a more granular
mask: `mtp.layers.0.{mlp.*, self_attn.{q,k,v,o}_proj}` ARE quantized
(weight_packed/weight_scale present), and so are
`linear_attn.{in_proj_qkv, in_proj_z, out_proj}`. Only specific
Linear-or-Conv1d submodules under linear_attn (in_proj_a, in_proj_b,
conv1d) and the `mtp` top-level `fc` / norms stay BF16.

Since QAD's value is in producing a **drop-in replacement** for the
deployed cyankiwi model, we follow the *deployed* mask (so the exported
checkpoint round-trips through vllm with the same kernel dispatch path
as the current submission).

Categories of modules:
- Quantized (INT4 g32 sym, ~207 modules per safetensors index):
  * model.language_model.layers.{0..31}.linear_attn.{in_proj_qkv, in_proj_z, out_proj}
  * model.language_model.layers.{0..31}.mlp.{gate_proj, up_proj, down_proj}
  * model.language_model.layers.{3,7,11,15,19,23,27,31}.self_attn.{q,k,v,o}_proj
  * mtp.layers.0.mlp.{gate_proj, up_proj, down_proj}
  * mtp.layers.0.self_attn.{q,k,v,o}_proj
- BF16 (kept full-precision):
  * model.language_model.embed_tokens (nn.Embedding → modelopt skips)
  * model.language_model.norm, *.{input_layernorm, post_attention_layernorm}
  * model.language_model.layers.*.linear_attn.{conv1d (Conv1d), in_proj_a, in_proj_b, norm}
  * model.language_model.layers.{Attn-idx}.self_attn.{k_norm, q_norm}
  * model.visual.* (entire vision encoder)
  * mtp.{fc, norm, pre_fc_norm_embedding, pre_fc_norm_hidden}
  * mtp.layers.0.{input_layernorm, post_attention_layernorm, self_attn.{k_norm, q_norm}}
  * lm_head

modelopt's quantization auto-targets Linear modules; nn.Embedding /
Conv1d / RMSNorm / LayerNorm are auto-skipped. So the EXPLICIT ignore
list only needs to cover Linear modules we want BF16:

    .*\\.linear_attn\\.in_proj_a$
    .*\\.linear_attn\\.in_proj_b$
    model\\.visual\\..*        (the vision Linears + Conv2d patch_embed)
    mtp\\.fc$
    lm_head$

API exposed:
- ``QAD_CFG``: dict ready to pass to ``modelopt.torch.quantization.quantize``.
- ``CYANKIWI_BF16_PATTERNS``: regex strings for Linears that must stay BF16
  (used both for modelopt config AND the post-export verifier).
- ``verify_mask(student)``: audits the quantized model against this mask.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

# Regex strings (no `re:` prefix; suitable for re.search). These match the
# Linear modules that the DEPLOYED cyankiwi safetensors keeps in BF16.
# nn.Embedding / Conv1d / RMSNorm / LayerNorm are auto-skipped by modelopt
# (it only quantizes Linear), so we don't need to list them.
CYANKIWI_BF16_PATTERNS: list[str] = [
    r".*\.linear_attn\.in_proj_a$",
    r".*\.linear_attn\.in_proj_b$",
    # linear_attn.conv1d is a ShortConvolution module (NOT nn.Conv1d) — it
    # has a Linear-like weight, so modelopt attaches a weight_quantizer.
    # The default INT4_BLOCKWISE preset's `*` rules leave it enabled, but
    # cyankiwi keeps these in BF16 — we explicitly include the pattern so
    # verify_mask() classifies them as expected-BF16.
    r".*\.linear_attn\.conv1d$",
    r"model\.visual\..*",     # vision encoder: attn.qkv/proj, mlp.linear_fc1/2, merger.linear_fc1/2, patch_embed.proj
    r"mtp\.fc$",
    r"lm_head$",
]

# Compiled once; used by verify_mask() and downstream debug callers.
_COMPILED_BF16 = [re.compile(p) for p in CYANKIWI_BF16_PATTERNS]

# For llmcompressor stage-B export, the `ignore` field expects
# `re:<pattern>` strings; keep a derived list with the prefix attached so
# qad_export.py can emit it directly without re-deriving.
CYANKIWI_BF16_RE_STRINGS: list[str] = [f"re:{p}" for p in CYANKIWI_BF16_PATTERNS]


def is_bf16(module_name: str) -> bool:
    """True if ``module_name`` matches a Linear that cyankiwi keeps BF16."""
    return any(rgx.search(module_name) for rgx in _COMPILED_BF16)


# Sanity sentinels for assertion in tests / smokes — these are the
# *actual* module path strings appearing in the cyankiwi safetensors
# index. is_bf16(...) MUST return True/False as marked below.
EXPECTED_BF16: list[str] = [
    "model.language_model.layers.0.linear_attn.in_proj_a",
    "model.language_model.layers.5.linear_attn.in_proj_b",
    "model.language_model.layers.0.linear_attn.conv1d",
    "model.visual.blocks.0.attn.qkv",
    "model.visual.blocks.0.attn.proj",
    "model.visual.blocks.23.mlp.linear_fc1",
    "model.visual.merger.linear_fc2",
    "mtp.fc",
    "lm_head",
]
EXPECTED_QUANTIZED: list[str] = [
    # All MLPs across language_model and mtp.layers.0
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.31.mlp.down_proj",
    "mtp.layers.0.mlp.up_proj",
    # All linear_attn.{in_proj_qkv, in_proj_z, out_proj}
    "model.language_model.layers.0.linear_attn.in_proj_qkv",
    "model.language_model.layers.5.linear_attn.in_proj_z",
    "model.language_model.layers.30.linear_attn.out_proj",
    # Full-attention layer self_attn proj sets
    "model.language_model.layers.3.self_attn.q_proj",
    "model.language_model.layers.27.self_attn.o_proj",
    # MTP block self_attn (per deployed safetensors)
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.v_proj",
]


# modelopt 0.44 changed quant_cfg from a dict to a list-of-entries
# schema where later entries override earlier ones. We build QAD_CFG by
# **deep-copying INT4_BLOCKWISE_WEIGHT_ONLY_CFG**, overriding the
# default group_size 128 → 32 (cyankiwi), and appending cyankiwi's
# BF16 Linear ignores (in_proj_a/b, visual, mtp.fc; lm_head and
# linear_attn.conv1d are already in the preset's default ignore list).
#
# We import modelopt lazily so that this module stays import-safe in
# environments without modelopt (the test suite and the main `.venv`).
def _build_qad_cfg() -> dict:
    import modelopt.torch.quantization as _mtq  # lazy import
    cfg = copy.deepcopy(_mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    # Override block_size in the *weight_quantizer entry's "cfg" dict.
    for entry in cfg.get("quant_cfg", []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("quantizer_name")
        sub_cfg = entry.get("cfg")
        if name == "*weight_quantizer" and isinstance(sub_cfg, dict):
            sub_cfg.setdefault("block_sizes", {})[-1] = 32
            sub_cfg.pop("axis", None)
    # Cyankiwi BF16 Linears (deployed mask). modelopt globs match against
    # the fully-qualified module name; `*pat*` ≈ Python `.*pat.*`.
    for pat in (
        "*linear_attn.in_proj_a*",
        "*linear_attn.in_proj_b*",
        "*linear_attn.conv1d*",   # match the verify_mask pattern (cyankiwi BF16)
        "*visual*",
        "*mtp.fc*",
    ):
        cfg["quant_cfg"].append({"quantizer_name": pat, "enable": False})
    return cfg


# Pre-built when this module is imported under a venv that has modelopt;
# otherwise None and lazy callers must reconstruct via _build_qad_cfg.
try:
    QAD_CFG: dict | None = _build_qad_cfg()
except Exception:  # noqa: BLE001
    # modelopt not installed in this venv (main .venv, test runners).
    QAD_CFG = None


@dataclass
class MaskAudit:
    """Result of walking a quantized student to confirm the cyankiwi mask."""
    n_quantized: int
    n_skipped: int
    misquantized: list[str]
    misignored: list[str]

    @property
    def ok(self) -> bool:
        return not self.misquantized and not self.misignored


def verify_mask(student,
                exclude_prefixes: tuple[str, ...] = ("_teacher_model",)) -> MaskAudit:
    """Walk ``student.named_modules()`` and audit the QAD quant mask.

    A Linear that matches any cyankiwi BF16 pattern MUST have its
    ``weight_quantizer.is_enabled == False``. Every other Linear with a
    weight_quantizer attached MUST have it enabled. Anything else is a
    mismatch (recorded in MaskAudit).

    ``exclude_prefixes`` skips named-modules whose path begins with any of
    the listed prefixes — used to ignore the frozen teacher branch when
    auditing an `mtd.DistillationModel`-wrapped student (whose
    ``named_modules()`` includes ``_teacher_model.*``).
    """
    misquantized: list[str] = []  # should-be-BF16-but-quantized
    misignored: list[str] = []    # should-be-quantized-but-BF16
    n_quant = 0
    n_skip = 0

    for name, module in student.named_modules():
        # Skip any subtree that is the wrapped teacher branch.
        if any(name == p or name.startswith(p + ".") for p in exclude_prefixes):
            continue
        wq = getattr(module, "weight_quantizer", None)
        if wq is None:
            continue
        is_enabled = bool(getattr(wq, "is_enabled", False))
        if not is_enabled and hasattr(wq, "_disabled"):
            is_enabled = not wq._disabled

        if is_bf16(name):
            if is_enabled:
                misquantized.append(name)
            else:
                n_skip += 1
        else:
            if not is_enabled:
                misignored.append(name)
            else:
                n_quant += 1

    return MaskAudit(
        n_quantized=n_quant,
        n_skipped=n_skip,
        misquantized=misquantized,
        misignored=misignored,
    )


def make_qad_cfg() -> dict:
    """Return a fresh deep-copy of QAD_CFG safe to mutate by the caller."""
    return copy.deepcopy(QAD_CFG)


if __name__ == "__main__":
    # Smoke test: confirm BF16 patterns + selectors work as expected.
    import json

    print("CYANKIWI_BF16_PATTERNS (deployed safetensors mask):")
    for p in CYANKIWI_BF16_PATTERNS:
        print(f"  re:{p}")
    print()
    if QAD_CFG is None:
        print("QAD_CFG: None (modelopt not installed in this venv)")
    else:
        print(f"QAD_CFG (modelopt 0.44 list format, {len(QAD_CFG['quant_cfg'])} entries):")
        # Only print the tail so output is short — the head comes from the
        # INT4_BLOCKWISE_WEIGHT_ONLY_CFG preset.
        for entry in QAD_CFG["quant_cfg"][-7:]:
            print(f"  {json.dumps(entry, default=str)}")
        print(f"  algorithm: {QAD_CFG['algorithm']}")

    print("\n--- pattern dry-run (against deployed cyankiwi module names) ---")
    bad = 0
    for n in EXPECTED_BF16:
        v = is_bf16(n)
        flag = "OK " if v else "FAIL"
        if not v:
            bad += 1
        print(f"  {flag} bf16?      {n}  -> {v}")
    for n in EXPECTED_QUANTIZED:
        v = is_bf16(n)
        flag = "OK " if not v else "FAIL"
        if v:
            bad += 1
        print(f"  {flag} !bf16?     {n}  -> {v}")
    if bad:
        print(f"\n[FAIL] {bad} pattern mismatch(es)")
        raise SystemExit(1)
    print("\n[ok] all patterns match expectations.")
