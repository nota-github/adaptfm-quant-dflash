"""Stage 3 (Draft Model, quant step) — GPTQ (calibrated) quantization of the DFlash
drafter to W4A16.

Instead of plain round-to-nearest from each weight tensor alone, GPTQ observes the
drafter's *real* per-Linear input activations (Hessian) and does
Hessian-error-compensated rounding.

Why a custom driver (not llmcompressor `oneshot`): DFlash's forward takes
(target_hidden, noise_embedding, position_ids), not token ids, so the text-driven
oneshot path can't drive it. We construct those inputs ourselves (teacher-forced,
matching the DFlash training recipe) and accumulate the Hessian via forward hooks,
then run a self-contained GPTQ and save in the compressed-tensors pack-quantized
format that vLLM loads as a speculative draft (with the src/vllm_plugin patches).

Capture (teacher-forced, Option A — spec_generate can't be reused because the Qwen3.5
target needs a hybrid linear-attention cache that the drafter's DynamicCache path breaks):
  for each calib conversation (rendered_full from calib_v1.jsonl):
    input_ids -> target(output_hidden_states=True, use_cache=False)
    target_hidden_raw = concat(hidden_states[lid+1] for lid in [1,8,15,22,29])  # (1,S,12800)
    for strided anchors s in the assistant region:
      block_ids   = [input_ids[s], mask_token_id x (block_size-1)]   # teacher-forced anchor
      noise_emb   = target.embed_tokens(block_ids)                   # (1,block,H)
      tgt_ctx     = target_hidden_raw[:, :s, :]                      # causal prefix (1,s,12800)
      pos_ids     = arange(s+block_size)                             # ctx + block positions
      drafter(target_hidden=tgt_ctx, noise_embedding=noise_emb, position_ids=pos_ids)
    -> forward-pre hooks on every drafter Linear accumulate H += xᵀx

GPTQ then quantizes each Linear (per-group sym INT4, group_size 128) using its H, and we
save via compressed_tensors ModelCompressor.

Usage (prefer scripts/quant_dflash_gptq.sh, which wires the paths):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python src/drafter_quant/quant_dflash_gptq.py \
      --calib runs/drafter_quant/calib/calib_v1.jsonl --output runs/drafter_quant/dflash_w4_gptq \
      --target <bf16-target> --drafter <bf16-draft> [--n-convs 256] [--anchors-per-conv 16] [--ignore-fc]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# --target / --drafter are passed explicitly (see scripts/quant_dflash_gptq.sh).
# --target = calib hidden source = the verify target as DENSE bf16 (W4A16 forward is
# numerically identical to its dequantized bf16, so the bf16 copy is the calib source).


# ----------------------------- GPTQ core -----------------------------
class GPTQ:
    """Weight-only per-group symmetric GPTQ (standard IST-DASLab algorithm)."""

    def __init__(self, layer: nn.Linear, dev):
        self.layer = layer
        self.dev = dev
        self.rows, self.columns = layer.weight.shape  # (out, in)
        self.H = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=dev)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        inp = inp.reshape(-1, inp.shape[-1]).to(torch.float32)
        n = inp.shape[0]
        if n == 0:
            return
        self.H *= self.nsamples / (self.nsamples + n)
        self.nsamples += n
        inp = inp * math.sqrt(2.0 / self.nsamples)
        self.H += inp.t() @ inp

    @torch.no_grad()
    def quantize(self, groupsize=128, bits=4, percdamp=0.01):
        W = self.layer.weight.data.clone().to(torch.float32)  # (out, in)
        H = self.H
        qmax = (1 << (bits - 1)) - 1
        qmin = -qmax - 1
        cols = self.columns

        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

        damp = percdamp * torch.mean(torch.diag(H)).clamp(min=1e-8)
        idx = torch.arange(cols, device=self.dev)
        H[idx, idx] += damp
        # Hinv via Cholesky (upper) of inverse
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        Q = torch.zeros_like(W)
        n_groups = (cols + groupsize - 1) // groupsize
        scales = torch.zeros((self.rows, n_groups), dtype=torch.float32, device=self.dev)
        cur_scale = None

        for i in range(cols):
            if i % groupsize == 0:
                g = W[:, i:i + groupsize]
                cur_scale = (g.abs().amax(dim=1) / qmax).clamp(min=1e-9)  # (rows,)
                scales[:, i // groupsize] = cur_scale
            w = W[:, i]
            d = Hinv[i, i]
            q = torch.clamp(torch.round(w / cur_scale), qmin, qmax) * cur_scale
            Q[:, i] = q
            err = (w - q) / d
            if i + 1 < cols:
                W[:, i + 1:] -= err.unsqueeze(1) * Hinv[i, i + 1:].unsqueeze(0)

        return Q.to(self.layer.weight.dtype), scales.to(torch.bfloat16)


# ----------------------------- capture driver -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target", required=True,
                    help="calib hidden source = verify target as dense bf16")
    ap.add_argument("--drafter", required=True, help="BF16 DFlash drafter to quantize")
    ap.add_argument("--n-convs", type=int, default=256)
    ap.add_argument("--anchors-per-conv", type=int, default=16)
    ap.add_argument("--anchor-lo-frac", type=float, default=0.0,
                    help="bias anchors later to grow the calib prefix length s: place anchors in "
                         "[P + frac*(S-block-P), S-block]. 0.0 = full assistant region (default, "
                         "== prior runs); higher frac = longer avg target_hidden prefix s (k/v context).")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--num-bits", type=int, default=4)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--ignore-fc", action="store_true",
                    help="keep fc in BF16 (default: quantize fc too, i.e. ignore=[]).")
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"removing existing {out}", file=sys.stderr)
        shutil.rmtree(out)

    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
    print("loading tokenizer + BF16 target ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16).to(dev).eval()
    embed = target.get_input_embeddings()
    print("loading drafter ...", flush=True)
    draft = AutoModel.from_pretrained(args.drafter, trust_remote_code=True, dtype=torch.bfloat16).to(dev).eval()
    block_size = draft.block_size
    mask_id = draft.mask_token_id
    layer_ids = list(draft.target_layer_ids)
    print(f"block_size={block_size} mask_id={mask_id} target_layer_ids={layer_ids}", flush=True)

    # ---- register GPTQ Hessian hooks on every drafter Linear (except fc if --ignore-fc)
    gptq: dict[str, GPTQ] = {}
    handles = []
    for name, mod in draft.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if args.ignore_fc and name == "fc":
            continue
        gptq[name] = GPTQ(mod, dev)

        def make_hook(nm):
            def hook(module, inp):
                gptq[nm].add_batch(inp[0].detach())
            return hook
        handles.append(mod.register_forward_pre_hook(make_hook(name)))
    print(f"hooked {len(gptq)} Linear modules for Hessian accumulation", flush=True)

    # ---- load calib, drive teacher-forced forwards
    recs = [json.loads(l) for l in open(args.calib)]
    recs = recs[:args.n_convs]
    print(f"driving capture over {len(recs)} conversations", flush=True)
    t0 = time.time()
    n_blocks_total = 0
    s_sum = 0  # sum of prefix lengths s over all anchors -> report avg s (k/v calib context)
    for ci, r in enumerate(recs):
        ids = tok(r["rendered_full"], add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        S = ids.shape[1]
        P = r["prompt_n_tokens"]
        if S <= P + block_size + 1:
            continue
        with torch.inference_mode():
            out_t = target(input_ids=ids, output_hidden_states=True, use_cache=False)
            hs = out_t.hidden_states
            target_hidden_raw = torch.cat([hs[lid + 1] for lid in layer_ids], dim=-1)  # (1,S,12800)

            lo, hi = P, S - block_size
            lo = int(round(lo + args.anchor_lo_frac * (hi - lo)))  # bias anchors later -> longer prefix s
            if hi <= lo:
                continue
            n_anchors = min(args.anchors_per_conv, hi - lo)
            anchors = torch.linspace(lo, hi - 1, n_anchors).round().long().tolist()
            for s in anchors:
                block_ids = torch.full((1, block_size), mask_id, dtype=torch.long, device=dev)
                block_ids[0, 0] = ids[0, s]
                noise_emb = embed(block_ids)                       # (1,block,H)
                tgt_ctx = target_hidden_raw[:, :s, :]              # (1,s,12800)
                pos_ids = torch.arange(s + block_size, device=dev).unsqueeze(0)
                draft(target_hidden=tgt_ctx, noise_embedding=noise_emb,
                      position_ids=pos_ids, use_cache=False)
                n_blocks_total += 1
                s_sum += int(s)
        if (ci + 1) % 25 == 0:
            print(f"  conv {ci+1}/{len(recs)}  blocks={n_blocks_total}  "
                  f"dt={time.time()-t0:.0f}s", flush=True)
    for h in handles:
        h.remove()
    avg_s = (s_sum / n_blocks_total) if n_blocks_total else 0.0
    print(f"capture done: {n_blocks_total} blocks, avg_prefix_s={avg_s:.0f} "
          f"(anchor_lo_frac={args.anchor_lo_frac}), dt={time.time()-t0:.0f}s", flush=True)
    nsamp = {k: v.nsamples for k, v in gptq.items()}
    print("nsamples per layer (min/median/max):",
          min(nsamp.values()), sorted(nsamp.values())[len(nsamp)//2], max(nsamp.values()), flush=True)

    # ---- GPTQ quantize each Linear; write Q + scale back onto the module
    print("running GPTQ per Linear ...", flush=True)
    t1 = time.time()
    scales_map = {}
    for name, g in gptq.items():
        Q, scale = g.quantize(groupsize=args.group_size, bits=args.num_bits, percdamp=args.percdamp)
        g.layer.weight.data = Q
        scales_map[name] = scale
        del g.H
    torch.cuda.empty_cache()
    print(f"GPTQ done dt={time.time()-t1:.0f}s", flush=True)

    # ---- save in compressed-tensors pack-quantized format
    from compressed_tensors.compressors import ModelCompressor
    from compressed_tensors.quantization import (
        QuantizationArgs, QuantizationConfig, QuantizationScheme,
        QuantizationStatus, QuantizationStrategy, QuantizationType,
    )
    from compressed_tensors.quantization.lifecycle.apply import apply_quantization_config

    draft = draft.to(torch.bfloat16).cpu()
    ignore = ["fc"] if args.ignore_fc else []
    weight_args = QuantizationArgs(
        num_bits=args.num_bits, type=QuantizationType.INT, symmetric=True,
        group_size=args.group_size, strategy=QuantizationStrategy.GROUP,
    )
    scheme = QuantizationScheme(targets=["Linear"], weights=weight_args)
    qcfg = QuantizationConfig(
        config_groups={"group_0": scheme}, ignore=ignore,
        quantization_status=QuantizationStatus.FROZEN, format="pack-quantized",
    )
    apply_quantization_config(draft, qcfg)
    nq = 0
    for name, mod in draft.named_modules():
        if not isinstance(mod, nn.Linear) or name not in scales_map:
            continue
        scale = scales_map[name].cpu()
        if not hasattr(mod, "weight_scale") or mod.weight_scale is None:
            mod.register_parameter("weight_scale", nn.Parameter(scale, requires_grad=False))
        else:
            mod.weight_scale.data = scale
        nq += 1
    print(f"attached scales to {nq} Linears; compressing ...", flush=True)
    compressor = ModelCompressor(quantization_config=qcfg)
    compressor.compress_model(draft)
    draft.save_pretrained(str(out))
    compressor.update_config(str(out))

    # metadata
    (out / "gptq_calib_meta.json").write_text(json.dumps({
        "method": "gptq", "group_size": args.group_size, "num_bits": args.num_bits,
        "percdamp": args.percdamp, "ignore_fc": args.ignore_fc,
        "n_convs": len(recs), "anchors_per_conv": args.anchors_per_conv,
        "anchor_lo_frac": args.anchor_lo_frac, "avg_prefix_s": round(avg_s, 1),
        "n_blocks": n_blocks_total, "calib": args.calib,
        "target_hidden_source": args.target,
    }, indent=2))
    print(f"[done] GPTQ drafter written to {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
