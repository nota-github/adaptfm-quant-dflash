# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Codex, Cursor, Copilot, etc.) working
with code in this repository. `CLAUDE.md` imports this file, so the same guidance applies to Claude Code.

## What this repo is

Reproduction repository for the **3rd-place** submission to the **Efficient Qwen
Competition** (AdaptFM workshop, ICML 2026). Task: minimize **Qwen3.5-4B**
inference latency on a single **NVIDIA A10G (24 GB)** while passing three quality
gates — **MMLU-Pro, IFEval, GPQA-Diamond**.

**Scope of work here is reproduction + source cleanup** — a standalone, newcomer-runnable
pipeline that produces the submission artifacts. The repo is fully self-contained. It is a
**frozen reproduction artifact**, not an actively maintained project, and does not solicit external
contributions — the goal is that a newcomer can re-run the pipeline, not extend it.

Root layout: **`scripts/` (entrypoints) + `src/` (everything they run)**. Two src/ trees are
**vendored** upstream code — `src/SpecForge/` (DFlash drafter-training framework; nested `.git`
removed, upstream HEAD `61f9cb0`) and `src/dflash_runtime/` (the `dflash` package: the DFlash
draft-model implementation vLLM imports at serve/eval time) — treat both as ordinary in-repo
source, edit directly. Models and generated artifacts are gitignored (`data/`, `models/`, `runs/`,
`src/data_generator/*.jsonl`, `.venv*`, `.env`, `*.log`, and SpecForge's own `cache/` `outputs/` `wandb/`).

## The pipeline (end-to-end reproduction)

Seven steps. `src/data_generator/` is used **twice** (different teachers); the drafter is trained in
**two phases** (pre-train on 220K → fine-tune on 400K) before quantization:

```
1 data_generator (220K)  →  2 qad            →  3 drafter pre-train  →  4 data_generator (400K)  →  5 drafter fine-tune  →  6 drafter quant  →  7 serve+eval
  teacher=Qwen3.5-4B         QAD INT4 (CT)      scratch on 220K          teacher=QAD ckpt5000        warm-start on 400K       GPTQ-W4A16          W4 target + W4 draft, SWA 1024
```

1. **`src/data_generator/` (220K)** — rebuild the fixed 220K prompt set from the shipped uuid
   manifest (`src/data_generator/manifests/`, via `rebuild_prompts_from_manifest.py` — NOT
   `sample_nemotron.py`, which can't reproduce this draw), serve the **BF16 teacher
   `Qwen/Qwen3.5-4B`**, regenerate responses → `data/qwen3_5_nemotron_combined_regen.jsonl` (~220K convs).
2. **`src/qad/` — Quantization-Aware Distillation.** Load `cyankiwi/Qwen3.5-4B-AWQ-4bit`, **preserve its
   INT4 weight grid exactly** by injecting its per-group scales as modelopt `_amax`, then distill the
   BF16-latent weights against the BF16 teacher on the 220K data with per-token KL. Deploys as
   INT4-weight / FP16-act (compressed-tensors) with **zero re-quant noise**. Final target = **ckpt5000**.
3. **drafter pre-train** (`scripts/train_drafter_pretrain.sh`) — DFlash draft from scratch on the
   **220K** convs, target = BF16 base `Qwen/Qwen3.5-4B` (the 220K teacher; no QAD dependency, so
   this can run in parallel with step 2) → `src/SpecForge/outputs/dflash-pretrain`.
4. **`src/data_generator/` (400K)** — re-run the generator with the **QAD model as teacher** →
   `src/data_generator/regen_ckpt5000.jsonl` (~400K convs). The draft must predict the *QAD-applied* model's outputs.
5. **drafter fine-tune** (`scripts/train_drafter_finetune.sh`) — warm-start the best pre-train ckpt,
   train on the **400K** set → `src/SpecForge/outputs/dflash-finetune`.
6. **drafter quant** (`scripts/quant_dflash_gptq.sh` + `src/drafter_quant/`) — GPTQ-W4A16 the fine-tuned
   draft → compressed-tensors pack (g128 sym INT4). Near-lossless vs the BF16 draft. On vLLM
   0.22.1 the W4A16 target forces the draft to be quantized too, so this is **required**, not optional.
7. **`src/serving/` + `src/eval_harness/`** — FastAPI proxy that spawns `vllm serve` (QAD target + DFlash draft
   as `speculative_config`), served with the drafter plugin: `EQC_DFLASH_QUANT_PATCH=1` (W4 draft) +
   `EQC_DFLASH_SWA_WINDOW=1024` (sliding-window attention). HTTP eval harness for the three gates + latency.

## Virtual environment (single, unified — `uv` + `pyproject.toml`)

**One venv for the whole repo** — a single environment defined by `pyproject.toml` (+ `uv.lock`).
Build from the repo root:

```bash
uv sync                                  # runtime: serve + data-gen + DFlash-eval + eval gates
uv sync --all-extras                     # + QAD train/pack (`qad`) + DFlash train (`train`) deps
uv pip install -e src/SpecForge --no-deps    # SpecForge training framework (its deps are in `train`)
```

Single resolution: **torch 2.11.0+cu130 / transformers 5.5.4 / triton 3.6 / vLLM 0.22.1 /
nvidia-modelopt 0.44**. Why one venv is possible: (1) **sglang is excluded** — it hard-pins
`transformers==4.57.1` (clashes with Qwen3.5's 5.x); SpecForge trains fine on its `hf` backend
without it. (2) **modelopt 0.44 caps `transformers` to ~5.5** and vLLM 0.22.1 accepts 5.5.4, so both
share `transformers==5.5.4`. (3) lm-eval's `[math]` extra is dropped (unused by the gates; antlr pin
clash). `.env` (gitignored) holds `HF_TOKEN` / `WNB_TOKEN` / `HF_HOME`; the train scripts auto-source it.
**`HF_TOKEN` is required** (not optional): `nvidia/Nemotron-Post-Training-Dataset-v2` (step 1 prompts)
and `Idavidrein/gpqa` (GPQA-Diamond gate) are gated — accept each dataset's terms on HF first. The
models and the other eval sets are public.

| stage | dir | notable deps | installed by |
|---|---|---|---|
| serving | `src/serving/` | vLLM 0.22.1, fastapi/uvicorn, `eqc_vllm_plugin` (editable) | base (`uv sync`) |
| data gen | `src/data_generator/` | vLLM 0.22.1, openai | base |
| DFlash eval | — | vLLM 0.22.1 + `dflash` (editable) | base |
| quality/latency eval | `src/eval_harness/` | lm-eval 0.4.11 (HTTP client) | base |
| QAD train + pack | `src/qad/` | modelopt 0.44, deepspeed, fla, causal-conv1d, bitsandbytes | `[qad]` extra |
| DFlash train | `src/SpecForge/` | HF backend, **no sglang** | `[train]` extra + `-e src/SpecForge` |

## Common commands

```bash
# --- Setup ---
uv sync --all-extras && uv pip install -e src/SpecForge --no-deps

# --- Step 2: QAD (train + pack) ---
scripts/prepare_data.sh                                                   # tokenize + pack (CPU, seq 16384)
scripts/train_multi.sh 0,1,2,3,4,5,6,7 src/configs/train_config_smoke.yaml    # 50-step smoke (needs ≥2 GPUs)
scripts/train_multi.sh 0,1,2,3,4,5,6,7 src/configs/train_config.yaml          # full 8000 steps (ZeRO-2, ≥2 GPUs)
scripts/pack_ct.sh checkpoint-5000                                        # → CT pack + dense -bf16 twin

# --- Steps 3/5/6: DFlash drafter (pre-train → fine-tune → quant) ---
scripts/train_drafter_pretrain.sh 8 flex_attention        # scratch on 220K → src/SpecForge/outputs/dflash-pretrain
# edit INIT_CKPT (best pre-train ckpt) in the finetune script, then:
scripts/train_drafter_finetune.sh 8 flex_attention        # warm-start on 400K → src/SpecForge/outputs/dflash-finetune
.venv/bin/python src/drafter_quant/build_calib.py --source src/data_generator/regen_ckpt5000.jsonl \
    --out runs/drafter_quant/calib --total 256
scripts/quant_dflash_gptq.sh                              # GPTQ-W4A16 → runs/drafter_quant/dflash_w4_gptq

# --- Step 7: serve + eval (single-GPU = A10G latency-faithful) ---
export PATH="$PWD/.venv/bin:$PATH"                        # so the spawned `vllm` CLI resolves
# latency path: QAD target + W4 draft + SWA 1024
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src EQC_CONFIG_PATH=src/configs/serve_dflash_gptq.yaml \
  EQC_PORT=8080 EQC_VLLM_INTERNAL_PORT=28080 EQC_GPU_MEM_CAP_FRACTION=0.30 \
  EQC_DFLASH_QUANT_PATCH=1 EQC_DFLASH_SWA_WINDOW=1024 \
  .venv/bin/python -m serving       # wait: curl localhost:8080/ping
# quality gates (dev sample, then full) — harness is HTTP-only, start server first:
QUALITY_LIMIT=0.1 CONTAINER_URL=http://localhost:8080 .venv/bin/python src/eval_harness/run_quality_local.py
CONTAINER_URL=http://localhost:8080 .venv/bin/python src/eval_harness/run_latency_harness.py
```

## Hardware emulation

Deployment target is one A10G (24 GB), but training/serving happens on A100/H100 80 GB boxes.
`scripts/gpu_lock.sh` locks one GPU and sets `EQC_GPU_MEM_CAP_FRACTION=0.30` (≈24/80) to emulate the
A10G cap. **Latency** must be measured on a single GPU with this cap; **quality** eval is throughput
and may run data-parallel (`EQC_VLLM_DP_SIZE`). Tensor-parallel is out of scope. QAD training shards
with DeepSpeed ZeRO-2 because seq-16384 full-vocab (248K) logits don't fit one 80 GB card — **≥2 GPUs
required** (a 1-GPU run OOMs at the first step); effective batch (16) is preserved (grad_accum ÷ world_size).

## Non-obvious gotchas

- **triton 3.6 + tilelang on Hopper**: the unified venv ships `triton==3.6` (torch 2.11 / vLLM 0.22.1)
  **plus `tilelang==0.1.9`** (a vLLM 0.22.1 dependency, so `uv sync` always installs it). On Hopper
  (sm_90) QAD's fla gated backward JIT-compiles via tilelang under triton ≥3.4 — so **re-training QAD
  from scratch on H100 works in the unified venv**, no isolated `triton==3.3.1` env needed.
- **QAD packing**: use `src/qad/pack_bf16_to_ct.py` (manual quant from BF16 latent + injected `_amax`),
  **not** modelopt `mtq.compress` — the latter re-quantizes off-grid and adds KL.
  `scripts/pack_ct.sh checkpoint-N` resolves the ckpt under `ckpts_full/` (or `ckpts_smoke/`) and
  writes **two** dirs: the CT pack `local_model_ct_ckptN` (serving) and the dense dequantized twin
  `local_model_ct_ckptN-bf16` (drafter fine-tune / GPTQ-calib target).
- **Data gen**: one independent DP=1 vLLM server per GPU (not `--data-parallel-size N`, which
  deadlocks); `--keep-truncated` is the biggest throughput lever; `/health` lies during warmup
  (probe with a real generation); spec-decode **garbles** W4A16 checkpoints so `SPEC=off` by default.
  The 220K step-1 corpus is rebuilt from a uuid **manifest** (it drew from the multilingual splits
  and used `id = md5(user text)`; the current sampler is non-ML-only with `id = uuid` — the two id
  schemes must not be mixed, e.g. in `--exclude-ids-file`). See `src/data_generator/README.md`.
- **DFlash training**: `--dataloader-num-workers 0` is **mandatory** (fork-CUDA deadlock); micro-batch
  **2** on H100 80 GB (vocab-248K logits OOM higher); both drafter scripts auto-derive `accum` to keep
  effective batch **32**; sglang must stay uninstalled. Patches are in `src/SpecForge/dflash_hf.patch`.
  Pre-train target = **BF16 base `Qwen/Qwen3.5-4B`**; fine-tune/GPTQ-calib target = **dequantized
  bf16** (`local_model_ct_ckpt5000-bf16`, written by `pack_ct.sh`); eval/serve target = **W4A16 original**.
- **Serving**: FastAPI proxy spawning `vllm serve` (offline `vllm.LLM` deadlocks under the harness);
  it invokes the `vllm` CLI by name, so put `.venv/bin` on `PATH`, and the `serving` package is run
  in place from `src/` (not installed), so set `PYTHONPATH=src`. `/invocations` is polymorphic
  (prompt→completions, messages→chat); thinking defaults **OFF** except GPQA. Set
  `EQC_VLLM_INTERNAL_PORT=28080` (8080+10000 collides with the harness proxy). See `src/serving/README.md`.
- **Eval datasets** must be pre-fetched into `HF_HOME` once (harness runs `HF_HUB_OFFLINE=1`);
  GPQA (`Idavidrein/gpqa`) is gated, so the fetch needs `HF_TOKEN`. Snippet in top-level README.
- **`PYTHONPATH=src` shadowing**: a directory directly under `src/` must not share its name with
  an installed import package — it would resolve as an empty namespace package and shadow the real
  one. That's why the editable-package project dirs differ from their import names:
  `src/vllm_plugin/` holds `eqc_vllm_plugin`, `src/dflash_runtime/` holds `dflash`.

## Where things live

- `src/configs/` — `train_config.yaml` (full QAD), `train_config_smoke.yaml`, `ds_zero2.json`,
  `serve_config.yaml` (QAD target only), `serve_dflash_gptq.yaml` (target + W4 draft, latency path).
- `scripts/` — `prepare_data` · `train_multi` · `pack_ct` (QAD), `train_drafter_pretrain` ·
  `train_drafter_finetune` (drafter), `quant_dflash_gptq`, `gpu_lock`.
- `src/dflash_runtime/` + `src/vllm_plugin/` — editable packages installed by `uv sync`
  (`[tool.uv.sources]`): the DFlash draft-model implementation and the W4-draft/SWA vLLM plugin.
- `runs/` (gitignored) — generated artifacts: `qad_nemotron_regen/ckpts_full/checkpoint-*`, packed CT
  models `local_model_ct_ckpt*`, `drafter_quant/dflash_w4_gptq`.
- `src/templates/qwen_no_think.jinja` — chat template that disables thinking.
- Per-stage runbooks live in `src/data_generator/README.md`, `src/serving/README.md`, `src/eval_harness/README.md`;
  the top-level `README.md` is the newcomer entry point.
