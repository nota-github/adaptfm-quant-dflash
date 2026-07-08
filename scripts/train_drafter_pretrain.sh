#!/usr/bin/env bash
# train_drafter_pretrain.sh — Stage 3 (Draft Model): DFlash drafter PRE-TRAIN (from scratch).
#
# Trains the DFlash draft from a random init (--draft-config-path) against the BF16
# base model Qwen/Qwen3.5-4B — the same teacher that generated the 220K conversation
# set this step trains on. (The QAD target only enters at fine-tune time:
# train_drafter_finetune.sh warm-starts from this run's best checkpoint and switches
# the target to the dense-bf16 QAD model.)
#
# Run from the repo root:
#   scripts/train_drafter_pretrain.sh [NUM_GPUS] [attention_backend]
#     NUM_GPUS            default 8  (accum is auto-derived to keep eff batch 32)
#     attention_backend  sdpa | flex_attention  (default flex_attention)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
SPECFORGE_DIR="${REPO_ROOT}/src/SpecForge"
TORCHRUN="${REPO_ROOT}/.venv/bin/torchrun"   # unified repo-root venv

if [[ -f .env ]]; then set -a; source .env; set +a; export HF_TOKEN="${HF_TOKEN:-}"; fi
if [[ "${HF_HOME:-}" == hf_* ]]; then unset HF_HOME; fi
export WANDB_API_KEY="${WANDB_API_KEY:-${WNB_TOKEN:-}}"

# ---- EDIT THESE ----------------------------------------------------------
TARGET_MODEL_PATH="Qwen/Qwen3.5-4B"                                                       # BF16 base target (= 220K-set teacher; HF hub id, auto-downloaded)
DRAFT_CONFIG_PATH="${SPECFORGE_DIR}/configs/qwen3.5-4b-dflash.json"                       # draft arch (scratch init)
TRAIN_DATA="${REPO_ROOT}/data/qwen3_5_nemotron_combined_regen.jsonl"                      # 220K set (Stage 1, BF16-teacher)
OUTPUT_DIR="${SPECFORGE_DIR}/outputs/dflash-pretrain"                                     # pre-train checkpoints
# --------------------------------------------------------------------------

NUM_GPUS="${1:-8}"
ATTENTION_BACKEND="${2:-flex_attention}"

export HF_DATASETS_CACHE="${SPECFORGE_DIR}/cache/hf_datasets"
export TORCHINDUCTOR_CACHE_DIR="${SPECFORGE_DIR}/cache/compiled_kernels"
export SPECFORGE_DATA_NUM_PROC=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation (big vocab=248320 logits)

# eff batch 32 invariant: 2 (per-device) * accum * world_size == 32  ->  accum = 16 / NUM_GPUS
BATCH_SIZE=2
if (( 16 % NUM_GPUS != 0 )); then
    echo "ERROR: NUM_GPUS=$NUM_GPUS does not divide 16; cannot keep eff batch 32 with batch_size=$BATCH_SIZE." >&2
    exit 1
fi
ACCUM=$(( 16 / NUM_GPUS ))
echo "[cfg] NUM_GPUS=$NUM_GPUS  batch_size=$BATCH_SIZE  accum=$ACCUM  -> eff batch $(( BATCH_SIZE * ACCUM * NUM_GPUS ))"

cd "${SPECFORGE_DIR}"
mkdir -p "${OUTPUT_DIR}"

"${TORCHRUN}" --standalone --nproc_per_node "${NUM_GPUS}" \
    "${SPECFORGE_DIR}/scripts/train_dflash.py" \
    --target-model-path "${TARGET_MODEL_PATH}" \
    --target-model-backend hf \
    --draft-config-path "${DRAFT_CONFIG_PATH}" \
    --train-data-path "${TRAIN_DATA}" \
    --output-dir "${OUTPUT_DIR}" \
    --chat-template qwen3.5 \
    --embedding-key model.language_model.embed_tokens.weight \
    --mask-token-id 248070 \
    --block-size 16 \
    --num-anchors 512 \
    --num-draft-layers 5 \
    --loss-decay-gamma 7.0 \
    --dataloader-num-workers 0 \
    --num-epochs 2 \
    --batch-size "${BATCH_SIZE}" \
    --accumulation-steps "${ACCUM}" \
    --learning-rate 1e-3 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --attention-backend "${ATTENTION_BACKEND}" \
    --log-interval 50 \
    --save-interval "${SAVE_INTERVAL:-1000}" \
    --report-to wandb \
    --wandb-project dflash-pretrain \
    --wandb-name dflash-pretrain-base220k \
    --trust-remote-code
