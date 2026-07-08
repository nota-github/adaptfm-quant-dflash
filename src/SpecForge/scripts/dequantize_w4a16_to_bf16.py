#!/usr/bin/env python3
"""Dequantize a compressed-tensors W4A16 (pack-quantized) checkpoint into a dense
bf16 checkpoint that HF transformers can load without compressed-tensors.

The dense bf16 weights are numerically identical to what the W4A16 forward would
dequantize to, so DFlash QAD faithfulness is preserved while the model loads
cleanly via AutoModelForCausalLM / output_hidden_states.

Usage:
  python scripts/dequantize_w4a16_to_bf16.py --src /path/to/w4a16 --dst /path/to/bf16
"""
import argparse
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--num-bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    st_files = sorted(glob.glob(os.path.join(args.src, "*.safetensors")))
    assert st_files, f"no safetensors in {args.src}"

    # Collect all tensors across shards, grouping quantized triplets by module.
    plain = {}            # key -> tensor (copied as-is)
    quant = {}            # module -> {packed, scale, shape}
    for f in st_files:
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                if k.endswith(".weight_packed"):
                    quant.setdefault(k[: -len(".weight_packed")], {})["packed"] = sf.get_tensor(k)
                elif k.endswith(".weight_scale"):
                    quant.setdefault(k[: -len(".weight_scale")], {})["scale"] = sf.get_tensor(k)
                elif k.endswith(".weight_shape"):
                    quant.setdefault(k[: -len(".weight_shape")], {})["shape"] = sf.get_tensor(k)
                else:
                    plain[k] = sf.get_tensor(k)

    GS = args.group_size
    out_sd = dict(plain)
    for mod, t in quant.items():
        packed, scale, shape = t["packed"], t["scale"], t["shape"]
        out_f, in_f = int(shape[0]), int(shape[1])
        # detect packed dim
        if packed.shape[1] * (32 // args.num_bits) == in_f:
            packed_dim = 1
        elif packed.shape[0] * (32 // args.num_bits) == out_f:
            packed_dim = 0
        else:
            raise ValueError(f"{mod}: cannot infer packed_dim packed={tuple(packed.shape)} shape={[out_f,in_f]}")
        unpacked = unpack_from_int32(packed, args.num_bits, shape, packed_dim=packed_dim)  # int8 [out,in]
        assert unpacked.shape == (out_f, in_f), (mod, unpacked.shape)
        w = unpacked.to(torch.float32).view(out_f, in_f // GS, GS)
        w = w * scale.to(torch.float32).unsqueeze(-1)
        w = w.view(out_f, in_f).to(torch.bfloat16).contiguous()
        out_sd[mod + ".weight"] = w
    print(f"dequantized {len(quant)} modules; total tensors {len(out_sd)}")

    save_file(out_sd, os.path.join(args.dst, "model.safetensors"), metadata={"format": "pt"})

    # config.json without quantization_config
    with open(os.path.join(args.src, "config.json")) as fh:
        cfg = json.load(fh)
    cfg.pop("quantization_config", None)
    with open(os.path.join(args.dst, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    # copy aux files (tokenizer, chat template, processor, etc.)
    for fn in os.listdir(args.src):
        if fn in ("config.json", "model.safetensors") or fn.endswith(".safetensors"):
            continue
        src_p = os.path.join(args.src, fn)
        if os.path.isfile(src_p):
            shutil.copy(src_p, os.path.join(args.dst, fn))
    print(f"wrote dense bf16 checkpoint to {args.dst}")


if __name__ == "__main__":
    main()
