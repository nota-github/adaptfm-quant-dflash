#!/usr/bin/env bash
# train_multi.sh — Stage 2 (multi-GPU): QAD distillation sharded with DeepSpeed
# ZeRO-2 across several GPUs.
#
# REQUIRES >=2 GPUs (80 GB class): the seq-16384 recipe's full-vocab (248K)
# logits don't fit one 80 GB GPU — a single-GPU run (smoke config included)
# OOMs at the first training step. ZeRO-2 (optimizer + gradient sharding)
# frees enough headroom from 2 GPUs up. Effective batch is preserved
# (grad_accum is divided by world_size inside the trainer), so "8000 steps"
# consumes exactly the same data at any GPU count.
#
# Usage:
#   scripts/train_multi.sh [GPUS] [CONFIG] [DS_CONFIG]
#     GPUS       comma list, e.g. 0,1,2,3,4,5,6,7   (default: all 8)
#     CONFIG     train config yaml                  (default: src/configs/train_config.yaml)
#     DS_CONFIG  deepspeed json                     (default: src/configs/ds_zero2.json)
#
# Examples:
#   scripts/train_multi.sh 0,1,2,3,4,5,6,7 src/configs/train_config_smoke.yaml   # smoke
#   scripts/train_multi.sh 0,1,2,3,4,5,6,7 src/configs/train_config.yaml         # full run
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Pull HF_TOKEN (and any other secrets) from .env if present.
if [[ -f .env ]]; then set -a; source .env; set +a; export HF_TOKEN="${HF_TOKEN:-}"; fi
if [[ "${HF_HOME:-}" == hf_* ]]; then unset HF_HOME; fi

GPUS="${1:-0,1,2,3,4,5,6,7}"
TRAIN_CFG="${2:-src/configs/train_config.yaml}"
DS_CFG="${3:-src/configs/ds_zero2.json}"
NPROC="$(awk -F',' '{print NF}' <<<"${GPUS}")"

if (( NPROC < 2 )); then
    echo "[train_multi.sh] WARNING: QAD needs >=2 GPUs — seq-16384 x 248K-vocab" >&2
    echo "                 logits OOM a single 80 GB GPU at the first train step." >&2
    echo "                 Proceeding anyway (calibration/inject/gate will run)." >&2
fi

CFG_BASE="$(basename "${TRAIN_CFG}" .yaml)"
case "${CFG_BASE}" in
  *smoke*) RUN_TAG="smoke" ;;
  *)       RUN_TAG="full" ;;
esac

PACKED="${REPO_ROOT}/runs/qad_nemotron_regen/train_data/packed.pt"
VAL_PACKED="${REPO_ROOT}/runs/qad_nemotron_regen/train_data/packed_val.pt"
OUT_DIR="${REPO_ROOT}/runs/qad_nemotron_regen/ckpts_${RUN_TAG}"
LOG_DIR="${REPO_ROOT}/runs/qad_nemotron_regen/logs"

if [[ ! -f "${PACKED}" ]]; then
    echo "train_multi.sh: missing ${PACKED}. Run scripts/prepare_data.sh first." >&2
    exit 2
fi

# Fresh start unless a real checkpoint-* exists (a stale modelopt state
# without checkpoints would poison quantizer re-init on the next run).
if [[ -f "${OUT_DIR}/modelopt_state_train.pth" ]] && ! compgen -G "${OUT_DIR}/checkpoint-*" >/dev/null; then
    echo "[train_multi.sh] clearing stale modelopt state in ${OUT_DIR}"
    rm -f "${OUT_DIR}/modelopt_state_train.pth"
fi
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

VAL_ARGS=()
if [[ -f "${VAL_PACKED}" ]]; then VAL_ARGS=(--val-packed "${VAL_PACKED}"); fi

export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# A10G deployment target -> single-GPU inference; this is TRAINING-only DP/ZeRO.

echo "[train_multi.sh] QAD ZeRO-2 on GPUs=${GPUS} (nproc=${NPROC}), config ${TRAIN_CFG}"

exec .venv/bin/torchrun --standalone --nproc_per_node="${NPROC}" \
    src/qad/train_cyankiwi_v2.py \
    --train-config "${TRAIN_CFG}" \
    --packed "${PACKED}" \
    "${VAL_ARGS[@]}" \
    --output-dir "${OUT_DIR}" \
    --log-dir "${LOG_DIR}" \
    --deepspeed "${DS_CFG}"
