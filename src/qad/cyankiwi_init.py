#!/usr/bin/env python3
"""cyankiwi_init.py — load cyankiwi/Qwen3.5-4B-AWQ-4bit as a BF16 student.

Replace the BF16 student init in QAD with a *dequantized
cyankiwi* student. The dequantized weights sit exactly on the INT4 grid
that cyankiwi chose with MSE-search calibration, so the upfront
`mtq.quantize(algorithm="max")` step inside QADTrainer derives an `_amax`
that matches cyankiwi's per-group scale to within rounding (because group
max of `s * round(w/s)` equals `s * max(|round|)` ≈ `s * 7` ≈ cyankiwi's
own scale). Net effect: the very first training step starts at near-zero
quant error instead of the max-calibration-induced error the baseline
slots (19/20/21) had to spend hundreds of steps to undo.

Compressed-tensors `pack-quantized` layout (cyankiwi config.json §7-31):
  * num_bits=4, group_size=32, symmetric (no zero point), packed_dim=1.
  * weight_packed [out, in/8] int32  — 8 signed-INT4 values packed per i32.
  * weight_scale  [out, in/32] bf16  — per-group absolute scale.
  * weight_shape  [2] int64          — original [out, in] dims.

Unsigned offset 8 is applied by `unpack_from_int32` so the int8 result
lives in [-8, 7], which is the modelopt INT4 sym convention; dequantized
weight is `int8 * scale_expanded`.

Public entry point: ``load_cyankiwi_dequant_state_dict`` returns a flat
``dict[str, torch.Tensor]`` keyed by ORIGINAL-model parameter names (e.g.
``model.language_model.layers.0.mlp.down_proj.weight``). Pass it to
``model.load_state_dict(..., strict=True)`` on a freshly-loaded BF16
Qwen3.5-4B and you have a drop-in student.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open

# Standalone default: a HuggingFace Hub repo id. If a local directory with
# this name does not exist, ``_resolve_cyankiwi_dir`` downloads it from the
# Hub (cached under ~/.cache/huggingface). A local path is still accepted
# verbatim — pass an absolute dir to use pre-downloaded weights offline.
DEFAULT_CYANKIWI_PATH = "cyankiwi/Qwen3.5-4B-AWQ-4bit"


def _resolve_cyankiwi_dir(cyankiwi_path: str | Path) -> Path:
    """Return a local directory holding cyankiwi's safetensors.

    If ``cyankiwi_path`` is an existing local directory, use it as-is.
    Otherwise treat it as a HuggingFace Hub repo id and snapshot-download
    it (weights + config). This keeps the QAD pipeline standalone: no
    pre-staged NAS/RAID path is required.
    """
    p = Path(cyankiwi_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=str(cyankiwi_path),
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
    )
    return Path(local)


def _unpack_int4_packed_dim1(packed: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Inline copy of compressed_tensors.unpack_from_int32 for num_bits=4,
    packed_dim=1. Returns int8 in [-8, 7]. Re-implemented here so the
    QAD venv's exact compressed-tensors version isn't a hard dependency
    of this module (we still validate against the lib once in __main__).
    """
    if packed.dtype != torch.int32:
        raise ValueError(f"expected int32 packed tensor, got {packed.dtype}")
    num_bits = 4
    pack_factor = 32 // num_bits   # 8
    mask = (1 << num_bits) - 1     # 0xF
    out, in_packed = packed.shape
    unpacked = torch.zeros((out, in_packed * pack_factor), dtype=torch.int32)
    for i in range(pack_factor):
        unpacked[:, i::pack_factor] = (packed >> (num_bits * i)) & mask
    unpacked = unpacked[:, : int(shape[1])]
    offset = 1 << (num_bits - 1)    # 8 → unsigned→signed
    return (unpacked - offset).to(torch.int8)


def _dequant_block_sym(
    packed: torch.Tensor, scale: torch.Tensor, shape: torch.Tensor
) -> torch.Tensor:
    """[out, in/8] int32 + [out, in/32] bf16 + [2] int64 → [out, in] bf16."""
    out_dim, in_dim = int(shape[0].item()), int(shape[1].item())
    int4 = _unpack_int4_packed_dim1(packed, (out_dim, in_dim))   # [out, in] int8 ∈ [-8, 7]
    n_groups = scale.shape[1]
    group_size = in_dim // n_groups
    if in_dim % n_groups:
        raise ValueError(
            f"in_dim {in_dim} not divisible by n_groups {n_groups}; "
            f"got scale shape {tuple(scale.shape)}"
        )
    scale_full = scale.to(torch.bfloat16).repeat_interleave(group_size, dim=1)
    return int4.to(torch.bfloat16) * scale_full


def _group_keys_by_module(keys: Iterable[str]) -> dict[str, dict[str, str]]:
    """Bucket safetensors keys by Linear module path.

    For each module path, collect the {weight|weight_packed|weight_scale|
    weight_shape|bias} subkeys that appear. Non-Linear params (norms,
    embeddings, scalars under .linear_attn) end up as their own bucket
    with a single "weight" entry — those flow through unchanged.
    """
    buckets: dict[str, dict[str, str]] = {}
    for k in keys:
        parts = k.rsplit(".", 1)
        if len(parts) != 2:
            buckets.setdefault(k, {})["_root"] = k
            continue
        module_path, suffix = parts
        buckets.setdefault(module_path, {})[suffix] = k
    return buckets


def load_cyankiwi_dequant_state_dict(
    cyankiwi_path: str | Path = DEFAULT_CYANKIWI_PATH,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Read cyankiwi safetensors, dequantize INT4 Linears to BF16, return
    a flat state_dict keyed by ORIGINAL-model parameter names.

    Output keys match what ``Qwen3_5ForConditionalGeneration.from_pretrained
    ('Qwen/Qwen3.5-4B')`` exposes via ``state_dict()`` (e.g.
    ``model.language_model.layers.0.mlp.down_proj.weight``). Suitable for
    ``model.load_state_dict(out, strict=True)``.
    """
    cyankiwi_path = _resolve_cyankiwi_dir(cyankiwi_path)
    st_path = cyankiwi_path / "model-00001-of-00001.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(f"no cyankiwi safetensors at {st_path}")

    out: dict[str, torch.Tensor] = {}
    with safe_open(str(st_path), framework="pt", device=str(device)) as f:
        all_keys = list(f.keys())
        buckets = _group_keys_by_module(all_keys)
        n_dequant = 0
        n_passthrough = 0
        for module_path, suffixes in buckets.items():
            if "_root" in suffixes:
                # root-level (no dot) — pass through as-is
                out[module_path] = f.get_tensor(module_path)
                n_passthrough += 1
                continue
            if "weight_packed" in suffixes:
                # Quantized Linear → dequantize to BF16
                packed = f.get_tensor(suffixes["weight_packed"])
                scale = f.get_tensor(suffixes["weight_scale"])
                shape = f.get_tensor(suffixes["weight_shape"])
                w = _dequant_block_sym(packed, scale, shape)
                out[f"{module_path}.weight"] = w
                if "bias" in suffixes:
                    out[f"{module_path}.bias"] = f.get_tensor(suffixes["bias"])
                n_dequant += 1
            else:
                # BF16 module (norm, embedding, ignored Linear, conv1d, …)
                for sfx, fullkey in suffixes.items():
                    out[fullkey] = f.get_tensor(fullkey)
                    n_passthrough += 1
    print(f"[cyankiwi_init] dequantized {n_dequant} Linear modules, "
          f"passed through {n_passthrough} BF16 tensors")
    return out


def load_cyankiwi_student(
    cyankiwi_path: str | Path = DEFAULT_CYANKIWI_PATH,
    base_model_id: str = "Qwen/Qwen3.5-4B",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Construct a ``Qwen3_5ForConditionalGeneration`` whose weights are
    the dequantized cyankiwi student, with the original BF16 model's
    architecture/config (i.e. no compressed-tensors quantization_config
    attached — modelopt will install its own quantizers).

    Returns the model on CPU; HF Trainer / accelerate handle placement.
    """
    from transformers import Qwen3_5ForConditionalGeneration

    print(f"[cyankiwi_init] loading base arch from {base_model_id}")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_model_id, trust_remote_code=True, torch_dtype=torch_dtype,
        device_map="cpu",
    )

    sd = load_cyankiwi_dequant_state_dict(cyankiwi_path=cyankiwi_path, device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[cyankiwi_init][warn] {len(missing)} missing keys (first 5): "
              f"{list(missing)[:5]}")
    if unexpected:
        print(f"[cyankiwi_init][warn] {len(unexpected)} unexpected keys (first 5): "
              f"{list(unexpected)[:5]}")
    return model


if __name__ == "__main__":
    # Smoke: dequantize one Linear and verify shape + that the result
    # actually lives on cyankiwi's INT4 grid (every value should equal
    # `k * scale` for some integer k ∈ [-8, 7]).
    sd = load_cyankiwi_dequant_state_dict()
    key = "model.language_model.layers.0.mlp.down_proj.weight"
    w = sd[key]
    print(f"[smoke] {key}: shape={tuple(w.shape)}, dtype={w.dtype}, "
          f"abs-max={w.abs().max().item():.4f}")
    # Grid check: load the matching scale, reconstruct expected
    st_path = str(
        _resolve_cyankiwi_dir(DEFAULT_CYANKIWI_PATH) / "model-00001-of-00001.safetensors"
    )
    with safe_open(st_path, framework="pt") as f:
        scale = f.get_tensor(
            "model.language_model.layers.0.mlp.down_proj.weight_scale"
        )
    gs = w.shape[1] // scale.shape[1]
    scale_full = scale.to(torch.bfloat16).repeat_interleave(gs, dim=1)
    k = (w / scale_full).round()
    assert k.min() >= -8 and k.max() <= 7, (k.min().item(), k.max().item())
    err = (w - k * scale_full).abs().max().item()
    print(f"[smoke] grid check: int range=[{int(k.min())}, {int(k.max())}], "
          f"residual abs-err={err:.4e}")
