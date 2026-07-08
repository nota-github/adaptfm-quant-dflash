#!/usr/bin/env bash
# train_drafter_finetune.sh — Stage 3 (Draft Model): DFlash drafter FINE-TUNE (warm-start).
# Continue training the PRE-TRAINED draft (from train_drafter_pretrain.sh, target =
# BF16 base Qwen/Qwen3.5-4B) against the DENSE bf16 QAD target on the 400K
# QAD-teacher regen set — the draft must mimic the *QAD-applied* model.
#
# Run from the repo root:
#   scripts/train_drafter_finetune.sh [NUM_GPUS] [attention_backend]
#     NUM_GPUS            default 8  (accum is auto-derived to keep eff batch 32)
#     attention_backend  sdpa | flex_attention  (default flex_attention)
#
# Warm-start mechanism (reuses src/SpecForge/scripts/train_dflash.py as-is): INIT_CKPT is
# exposed as epoch_0 in a FRESH OUTPUT_DIR via per-file symlinks that EXCLUDE
# training_state.pt, so --resume loads it WEIGHTS-ONLY and optimizer/LR-schedule start
# clean at epoch0/step0. (Raw pre-train ckpts DO contain training_state.pt — the
# pre-train epoch counter + optimizer/LR state; exposing it would make --resume adopt
# them: epoch >= num-epochs -> silent zero-step run.)
# The draft arch/config comes from INIT_CKPT's config.json (layers=5,
# target_layer_ids=[1,8,15,22,29], block_size=16), so --num-draft-layers / --draft-config-path
# cannot override it. eff batch 32 is the invariant, not a fixed accum.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
SPECFORGE_DIR="${REPO_ROOT}/src/SpecForge"
TORCHRUN="${REPO_ROOT}/.venv/bin/torchrun"   # unified repo-root venv (no PATH activation needed)

# Pull HF_TOKEN / WNB_TOKEN from .env if present (wandb logging).
if [[ -f .env ]]; then set -a; source .env; set +a; export HF_TOKEN="${HF_TOKEN:-}"; fi
if [[ "${HF_HOME:-}" == hf_* ]]; then unset HF_HOME; fi

# ---- EDIT THESE ----------------------------------------------------------
TARGET_MODEL_PATH="${REPO_ROOT}/runs/qad_nemotron_regen/local_model_ct_ckpt5000-bf16"   # DENSE bf16 QAD target (written by scripts/pack_ct.sh)
INIT_CKPT="${SPECFORGE_DIR}/outputs/dflash-pretrain/epoch_2_step_XXXXX"                  # EDIT: best pre-train ckpt (the 2-epoch run's final save lands at epoch_2_step_*)
TRAIN_DATA="${REPO_ROOT}/src/data_generator/regen_ckpt5000.jsonl"                       # 400K, QAD ckpt5000-teacher regen
OUTPUT_DIR="${SPECFORGE_DIR}/outputs/dflash-finetune"                                    # FRESH dir (must NOT contain prior epoch_* ckpts)
# --------------------------------------------------------------------------

NUM_GPUS="${1:-8}"
ATTENTION_BACKEND="${2:-flex_attention}"   # sdpa or flex_attention (both OK with num-workers 0; flex ~10% faster)

export HF_DATASETS_CACHE="${SPECFORGE_DIR}/cache/hf_datasets"
export TORCHINDUCTOR_CACHE_DIR="${SPECFORGE_DIR}/cache/compiled_kernels"
export SPECFORGE_DATA_NUM_PROC=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation (big vocab=248320 logits)

# eff batch 32 invariant: 2 (per-device) * accum * world_size == 32  ->  accum = 16 / NUM_GPUS
BATCH_SIZE=2
if (( 16 % NUM_GPUS != 0 )); then
    echo "ERROR: NUM_GPUS=$NUM_GPUS does not divide 16; cannot keep eff batch 32 with batch_size=$BATCH_SIZE." >&2
    echo "       Pick NUM_GPUS in {1,2,4,8,16} or adjust BATCH_SIZE/accum by hand." >&2
    exit 1
fi
ACCUM=$(( 16 / NUM_GPUS ))
echo "[cfg] NUM_GPUS=$NUM_GPUS  batch_size=$BATCH_SIZE  accum=$ACCUM  -> eff batch $(( BATCH_SIZE * ACCUM * NUM_GPUS ))"

# warm-start guard: a fresh OUTPUT_DIR must not already hold trained epoch_*_step_* ckpts,
# else get_last_checkpoint() would resume from those instead of the warm-start init.
if compgen -G "${OUTPUT_DIR}/epoch_*_step_*" > /dev/null; then
    echo "ERROR: ${OUTPUT_DIR} already contains trained checkpoints (epoch_*_step_*)." >&2
    echo "       Use a new OUTPUT_DIR or clear it before warm-starting." >&2
    exit 1
fi

# INIT_CKPT's files are symlinked from inside OUTPUT_DIR/epoch_0 — it must be an
# existing dir (use an absolute path; a relative one would produce dangling symlinks
# and the trainer would silently fall back to a scratch draft config).
if [[ ! -d "${INIT_CKPT}" ]]; then
    echo "ERROR: INIT_CKPT does not exist: ${INIT_CKPT}" >&2
    exit 1
fi

# run with SpecForge as cwd so --cache-dir ./cache resolves to src/SpecForge/cache (matches prior runs)
cd "${SPECFORGE_DIR}"
mkdir -p "${OUTPUT_DIR}"
# expose init ckpt as epoch_0 so --resume loads it weights-only. Do NOT symlink the
# whole dir: raw pre-train checkpoints CONTAIN training_state.pt (optimizer/LR state +
# the pre-train epoch counter), and --resume would adopt them — epoch >= num-epochs
# means a silent zero-step run that saves the pre-train weights as "fine-tuned".
# Link every file EXCEPT training_state.pt so the fine-tune starts clean at epoch0/step0.
if [ ! -e "${OUTPUT_DIR}/epoch_0" ]; then
    mkdir -p "${OUTPUT_DIR}/epoch_0"
    for f in "${INIT_CKPT}"/*; do
        [ "$(basename "$f")" = "training_state.pt" ] && continue
        ln -s "$f" "${OUTPUT_DIR}/epoch_0/$(basename "$f")"
    done
fi
# Loud guard against a leftover whole-dir symlink from the older script revision
# (it would re-expose training_state.pt and resurrect the silent no-op).
if [ -e "${OUTPUT_DIR}/epoch_0/training_state.pt" ]; then
    echo "ERROR: ${OUTPUT_DIR}/epoch_0 exposes training_state.pt (old whole-dir symlink?)." >&2
    echo "       Remove ${OUTPUT_DIR}/epoch_0 and rerun so the warm start is weights-only." >&2
    exit 1
fi

"${TORCHRUN}" --standalone --nproc_per_node "${NUM_GPUS}" \
    "${SPECFORGE_DIR}/scripts/train_dflash.py" \
    --target-model-path "${TARGET_MODEL_PATH}" \
    --target-model-backend hf \
    --train-data-path "${TRAIN_DATA}" \
    --output-dir "${OUTPUT_DIR}" \
    --resume \
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
    --wandb-project dflash-finetune \
    --wandb-name dflash-finetune-ckpt5000-400k \
    --trust-remote-code
