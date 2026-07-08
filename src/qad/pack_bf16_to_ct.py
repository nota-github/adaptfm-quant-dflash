#!/usr/bin/env python3
"""pack_bf16_to_ct.py — pack a BF16+amax modelopt checkpoint into the
compressed-tensors (pack-quantized) format that vllm 0.22.1 loads, and
(optionally) its dense-BF16 twin used as the drafter-training target.

Why: ``mtq.compress`` + reload was empirically shown to add KL on top of
the in-memory training state, because modelopt's compress path re-quantizes BF16 latents through its
own scale convention, mismatching the cyankiwi-injected grid. By
quantizing manually from the BF16 latents + injected ``_amax`` we get
EXACTLY the on-grid values the trainer was fake-quanting against → 0
re-quant noise on round-trip.

Steps:

  1. Load the BF16 checkpoint via mto-patched ``from_pretrained``
     (restores quantizers + ``_amax`` from co-saved modelopt_state.pth).
     Weights are full BF16 [out, in].

  2. For each enabled ``weight_quantizer``:
       scale  = wq._amax / 7    (narrow_range=False, see verify_cyankiwi_init)
       int4   = clamp(round(weight / scale), -8, 7)         signed
       packed = pack_to_int32_in_dim(int4)                 [out, in/8]
     Save: ``{base}.weight_packed`` (int32), ``{base}.weight_scale``
     (bf16, = scale), ``{base}.weight_shape`` (int64 [out, in]).

  3. Non-quantized tensors (norms, embeddings, conv1d, lm_head,
     ignored Linears) pass through as BF16.

  4. Optionally stitch ``mtp.*`` BF16 weights from cyankiwi (needed for
     vllm qwen3_5_mtp spec-dec). Base Qwen3.5-4B has no mtp block.

  5. Write ``config.json`` with compressed-tensors quantization_config
     + full ``ignore`` list (visual.* + mtp.* + lm_head + linear_attn
     in_proj_a/b/conv1d). Copy tokenizer + preprocessor sidecars.

  6. With ``--dense-bf16-dst``, also write the DENSE bf16 twin: every
     quantized weight replaced by its dequantized on-grid value
     (int4 × bf16 scale — numerically what vllm computes from the CT
     pack at serve time), config without quantization_config, same
     sidecars. This is the ``local_model_ct_ckptN-bf16`` dir that
     drafter fine-tune / GPTQ quant use as the training/calib target.

Usage:

    scripts/pack_ct.sh checkpoint-5000
        # == .venv/bin/python src/qad/pack_bf16_to_ct.py \\
        #      --src runs/qad_nemotron_regen/ckpts_full/checkpoint-5000 \\
        #      --dst runs/qad_nemotron_regen/local_model_ct_ckpt5000 \\
        #      --dense-bf16-dst runs/qad_nemotron_regen/local_model_ct_ckpt5000-bf16 \\
        #      --stitch-mtp-from cyankiwi
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC_ROOT = Path(__file__).resolve().parents[1]  # src/, so `qad.` imports resolve
sys.path.insert(0, str(SRC_ROOT))

from qad.cyankiwi_init import (  # noqa: E402
    DEFAULT_CYANKIWI_PATH,
    load_cyankiwi_dequant_state_dict,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True,
                   help="BF16 modelopt checkpoint dir (with modelopt_state.pth).")
    p.add_argument("--dst", required=True, help="Output compressed-tensors dir.")
    p.add_argument("--dense-bf16-dst", default=None,
                   help="Also write the dense BF16 twin (dequantized on-grid "
                        "weights, no quantization_config) to this dir — the "
                        "drafter-training / GPTQ-calib target.")
    p.add_argument("--base-model", default="Qwen/Qwen3.5-4B",
                   help="Base HF model id for tokenizer / processor sidecars.")
    p.add_argument("--stitch-mtp-from", default="cyankiwi",
                   choices=("none", "cyankiwi"),
                   help="Where to source mtp.* BF16 tensors. 'cyankiwi' = "
                        "dequantize cyankiwi's mtp.* into BF16. 'none' = skip "
                        "(MTP speculative decoding won't work in vllm).")
    p.add_argument("--cyankiwi-path", default=DEFAULT_CYANKIWI_PATH)
    return p.parse_args()


def pack_compressed_tensors(int4_signed: torch.Tensor) -> torch.Tensor:
    """Pack INT4 (-8..7) into compressed-tensors int32 along in_dim.

    Input  (out, in) int8  values -8..7
    Output (out, in/8) int32  8 nibbles / int32, low nibble first.
    Encoding: offset-binary (shift +8 → 0..15) per compressed-tensors lib.
    """
    out_dim, in_dim = int4_signed.shape
    assert in_dim % 8 == 0, f"in_dim={in_dim} must be divisible by 8"
    shifted = (int4_signed.to(torch.int32) + 8)              # 0..15
    assert shifted.min().item() >= 0 and shifted.max().item() <= 15
    reshaped = shifted.reshape(out_dim, in_dim // 8, 8)      # (out, in/8, 8)
    shifts = torch.arange(8, dtype=torch.int32) * 4
    packed = (reshaped << shifts).sum(dim=-1).to(torch.int32)
    return packed


def quantize_to_int4(weight_bf16: torch.Tensor, amax_flat: torch.Tensor,
                     group_size: int = 32
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a [out, in] BF16 weight using per-group amax (modelopt's
    narrow_range=False with maxbound=7 convention).

    Returns (int4_signed [out, in], scale_bf16 [out, in/group_size],
    dequant_bf16 [out, in]). dequant is computed from the STORED bf16
    scale, so it matches what vllm reconstructs from the CT pack.
    """
    out_dim, in_dim = weight_bf16.shape
    n_groups = in_dim // group_size
    assert in_dim % group_size == 0, f"in_dim={in_dim} not divisible by {group_size}"

    # Reshape amax to (out, n_groups). modelopt may store as flat or
    # [out, n_groups, 1]; either way numel = out*n_groups.
    if amax_flat.numel() != out_dim * n_groups:
        raise ValueError(
            f"amax numel {amax_flat.numel()} != out*n_groups "
            f"{out_dim * n_groups} (out={out_dim}, n_groups={n_groups})")
    amax_2d = amax_flat.float().reshape(out_dim, n_groups)
    scale_2d = amax_2d / 7.0                                  # narrow_range=False, maxbound=7
    scale_2d = scale_2d.clamp_min(1e-30)

    # Expand scale to per-element for quantization
    scale_full = scale_2d.unsqueeze(-1).repeat(1, 1, group_size).reshape(out_dim, in_dim)
    int4 = (weight_bf16.float() / scale_full).round().clamp(-8, 7).to(torch.int8)
    scale_bf16 = scale_2d.to(torch.bfloat16)
    dq_scale_full = scale_bf16.float().unsqueeze(-1).repeat(1, 1, group_size) \
                              .reshape(out_dim, in_dim)
    dequant_bf16 = (int4.float() * dq_scale_full).to(torch.bfloat16)
    return int4, scale_bf16, dequant_bf16


def main() -> int:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    dense_dst = Path(args.dense_bf16_dst) if args.dense_bf16_dst else None
    if dense_dst is not None:
        dense_dst.mkdir(parents=True, exist_ok=True)

    print(f"[pack] loading BF16+amax ckpt from {src}")
    import modelopt.torch.opt as mto
    mto.enable_huggingface_checkpointing()
    from transformers import Qwen3_5ForConditionalGeneration

    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(src), trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    print(f"[pack] load done in {time.time()-t0:.1f}s")

    # Walk and quantize
    new_tensors: dict[str, torch.Tensor] = {}
    dense_tensors: dict[str, torch.Tensor] = {}
    n_quant = 0
    n_passthrough = 0
    quant_module_paths: list[str] = []
    bf16_module_paths: list[str] = []

    for name, mod in model.named_modules():
        wq = getattr(mod, "weight_quantizer", None)
        if wq is not None and getattr(wq, "is_enabled", False):
            amax = getattr(wq, "_amax", None)
            if not isinstance(amax, torch.Tensor):
                continue
            w = mod.weight.data
            try:
                int4, scale, dequant = quantize_to_int4(w, amax)
            except Exception as e:
                print(f"[pack][skip] {name}: {e}", file=sys.stderr)
                continue
            packed = pack_compressed_tensors(int4)
            out_dim, in_dim = w.shape
            new_tensors[f"{name}.weight_packed"] = packed.contiguous()
            new_tensors[f"{name}.weight_scale"] = scale.contiguous()
            new_tensors[f"{name}.weight_shape"] = torch.tensor(
                [out_dim, in_dim], dtype=torch.int64)
            if mod.bias is not None:
                new_tensors[f"{name}.bias"] = mod.bias.data.to(torch.bfloat16).contiguous()
            if dense_dst is not None:
                dense_tensors[f"{name}.weight"] = dequant.contiguous()
                if mod.bias is not None:
                    dense_tensors[f"{name}.bias"] = new_tensors[f"{name}.bias"]
            n_quant += 1
            quant_module_paths.append(name)

    print(f"[pack] quantized {n_quant} Linear modules → packed int32 + bf16 scale")

    # Walk again for non-quantized params (passthrough)
    quant_paths_set = set(quant_module_paths)
    for name, p in model.named_parameters():
        # Detect quantized modules by prefix match
        is_quantized = False
        for qp in quant_module_paths:
            if name == f"{qp}.weight" or name == f"{qp}.bias":
                is_quantized = True
                break
        if is_quantized:
            continue
        # Passthrough as BF16
        new_tensors[name] = p.data.to(torch.bfloat16).contiguous()
        if dense_dst is not None:
            dense_tensors[name] = new_tensors[name]
        n_passthrough += 1
        # Record the module path (strip .weight/.bias) for ignore list
        for suf in (".weight", ".bias"):
            if name.endswith(suf):
                bf16_module_paths.append(name[: -len(suf)])
                break

    print(f"[pack] passthrough {n_passthrough} non-quant params (BF16)")

    # Optional mtp stitch from cyankiwi
    if args.stitch_mtp_from == "cyankiwi":
        print(f"[pack] stitching mtp.* from {args.cyankiwi_path}")
        cyk_sd = load_cyankiwi_dequant_state_dict(args.cyankiwi_path, device="cpu")
        mtp_keys = sorted(k for k in cyk_sd if k.startswith("mtp"))
        n_added = 0
        for k in mtp_keys:
            if k in new_tensors:
                continue
            new_tensors[k] = cyk_sd[k].to(torch.bfloat16).contiguous()
            if dense_dst is not None:
                dense_tensors[k] = new_tensors[k]
            mod_path = k[: -len(".weight")] if k.endswith(".weight") else k
            if mod_path not in bf16_module_paths:
                bf16_module_paths.append(mod_path)
            n_added += 1
        print(f"[pack] stitched {n_added} cyankiwi mtp.* tensors")

    # Build compressed-tensors quantization_config
    # All bf16 Linear modules need to be in 'ignore' (so vllm doesn't try
    # to load them as quantized). Also add visual.*, mtp.*, lm_head,
    # linear_attn.in_proj_a/b/conv1d (cyankiwi mask spec) for safety even
    # if some BF16 module slipped through.
    ignore_set = set()
    for path in bf16_module_paths:
        if "visual" in path or path.startswith("mtp") or path == "lm_head":
            ignore_set.add(path)
        elif "linear_attn.in_proj_a" in path or "linear_attn.in_proj_b" in path \
                or "linear_attn.conv1d" in path:
            ignore_set.add(path)
    ignore_list = sorted(ignore_set)

    # Read source config.json and rewrite quantization_config
    src_cfg_path = src / "config.json"
    with open(src_cfg_path) as f:
        cfg = json.load(f)
    # KDTrainer wraps Qwen3_5 in DistillationModel → saved config.json's
    # architectures field becomes 'DistillQwen3_5ForConditionalGeneration'.
    # vllm doesn't recognize that — restore the base architecture name.
    cfg["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    cfg["quantization_config"] = {
        "config_groups": {
            "group_0": {
                "format": "pack-quantized",
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": 32,
                    "num_bits": 4,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "scale_dtype": None,
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int",
                    "zp_dtype": None,
                },
            },
        },
        "format": "pack-quantized",
        "global_compression_ratio": None,
        "ignore": ignore_list,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
        "version": "0.14.1",
    }
    out_cfg_path = dst / "config.json"
    with open(out_cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[pack] wrote {out_cfg_path}  (ignore: {len(ignore_list)} entries)")

    # Write the single safetensors shard
    out_st_path = dst / "model.safetensors"
    save_file(new_tensors, str(out_st_path), metadata={"format": "pt"})
    print(f"[pack] wrote {out_st_path}  ({len(new_tensors)} tensors)")

    # Dense BF16 twin: same weights dequantized on-grid, config without
    # quantization_config. Drafter fine-tune + GPTQ calib load this dir.
    if dense_dst is not None:
        dense_cfg = dict(cfg)
        dense_cfg.pop("quantization_config", None)
        with open(dense_dst / "config.json", "w") as f:
            json.dump(dense_cfg, f, indent=2)
        save_file(dense_tensors, str(dense_dst / "model.safetensors"),
                  metadata={"format": "pt"})
        print(f"[pack] wrote dense bf16 twin {dense_dst}/model.safetensors "
              f"({len(dense_tensors)} tensors)")

    # Copy tokenizer + preprocessor sidecars from base model snapshot.
    print(f"[pack] copying tokenizer + preprocessor sidecars from base")
    from transformers import AutoTokenizer
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
        proc.save_pretrained(str(dst))
        print(f"[pack] AutoProcessor sidecars saved")
    except Exception as e:
        print(f"[pack][warn] AutoProcessor save failed: {e}")
        tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        tok.save_pretrained(str(dst))

    # AutoProcessor on Qwen3.5 emits processor_config.json but NOT
    # preprocessor_config.json / video_preprocessor_config.json — vllm's
    # cached_get_processor() looks for preprocessor_config.json
    # explicitly and 500s without it. Fall back to direct snapshot copy.
    from huggingface_hub import snapshot_download
    snapshot_dir = Path(snapshot_download(args.base_model))
    for f in ("preprocessor_config.json", "video_preprocessor_config.json",
              "generation_config.json", "merges.txt", "vocab.json"):
        src_f = snapshot_dir / f
        dst_f = dst / f
        if src_f.exists() and not dst_f.exists():
            shutil.copy2(str(src_f), str(dst_f))
            print(f"[pack] copied sidecar {f} from snapshot")

    # Copy chat_template if missing
    src_ct = src / "chat_template.jinja"
    if src_ct.exists() and not (dst / "chat_template.jinja").exists():
        shutil.copy2(str(src_ct), str(dst / "chat_template.jinja"))

    # Dense twin gets the exact same sidecars (tokenizer/processor/template).
    if dense_dst is not None:
        for p in sorted(dst.iterdir()):
            if not p.is_file() or p.name in ("config.json", "model.safetensors"):
                continue
            tgt = dense_dst / p.name
            if not tgt.exists():
                shutil.copy2(str(p), str(tgt))
        print(f"[pack] copied sidecars into dense twin {dense_dst}")

    for out_dir in [dst] + ([dense_dst] if dense_dst is not None else []):
        print(f"\n[pack] OUTPUT FILES ({out_dir}):")
        for p in sorted(out_dir.iterdir()):
            size_mb = p.stat().st_size / (1 << 20) if p.is_file() else 0.0
            suf = f"  ({size_mb:.1f} MB)" if size_mb > 0.1 else ""
            print(f"   {p.name}{suf}")
    print(f"[pack] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
