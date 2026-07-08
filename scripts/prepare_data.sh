#!/usr/bin/env bash
# prepare_data.sh — Stage 2 (Target Model) data prep: tokenize + pack the
# Stage-1 nemotron-regen corpus (data/qwen3_5_nemotron_combined_regen.jsonl).
#
# CPU-only (tokenizer + packing). Produces:
#   runs/qad_nemotron_regen/train_data/packed.pt      (train chunks)
#   runs/qad_nemotron_regen/train_data/packed_val.pt  (val chunks)
#   runs/qad_nemotron_regen/train_data/pack_meta.json
#
# Usage: scripts/prepare_data.sh [extra args forwarded to data_nemotron_regen.py]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Pull HF_TOKEN / HF_HOME from .env (tokenizer download / HF cache location).
if [[ -f .env ]]; then set -a; source .env; set +a; export HF_TOKEN="${HF_TOKEN:-}"; fi
if [[ "${HF_HOME:-}" == hf_* ]]; then unset HF_HOME; fi

OUT_DIR="${REPO_ROOT}/runs/qad_nemotron_regen/train_data"
mkdir -p "${OUT_DIR}"

exec .venv/bin/python src/qad/data_nemotron_regen.py \
    --input  "${REPO_ROOT}/data/qwen3_5_nemotron_combined_regen.jsonl" \
    --output "${OUT_DIR}/packed.pt" \
    --max-seq-len 16384 \
    --val-ratio 0.05 \
    --shuffle-seed 42 \
    "$@"
