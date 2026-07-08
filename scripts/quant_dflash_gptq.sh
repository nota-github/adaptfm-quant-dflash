#!/usr/bin/env bash
# Stage 3 (Draft Model, quant step) — GPTQ W4A16 of the DFlash drafter.
# Calibrated Hessian-error-compensated rounding, exported as compressed-tensors
# (pack-quantized, g128 sym INT4) so vLLM loads it as a speculative draft.
# See src/drafter_quant/quant_dflash_gptq.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ----------------------------- EDIT THESE -----------------------------
PY="${PY:-${REPO_ROOT}/.venv/bin/python}"    # unified repo-root venv
GPU="${GPU:-7}"

# --target = calib hidden source = verify target as DENSE bf16 (numerically == W4A16
# forward); written by scripts/pack_ct.sh alongside the CT pack.
TARGET="${TARGET:-${REPO_ROOT}/runs/qad_nemotron_regen/local_model_ct_ckpt5000-bf16}"
# EDIT: best fine-tuned BF16 draft (train_drafter_finetune.sh; the 2-epoch run's
# final save lands at epoch_2_step_*)
DRAFTER="${DRAFTER:-${REPO_ROOT}/src/SpecForge/outputs/dflash-finetune/epoch_2_step_XXXXX}"
# calibration set — built by src/drafter_quant/build_calib.py from the 400K regen set.
CALIB="${CALIB:-${REPO_ROOT}/runs/drafter_quant/calib/calib_v1.jsonl}"
# output dir for the quantized drafter (served via src/configs/serve_dflash_gptq.yaml)
OUT="${OUT:-${REPO_ROOT}/runs/drafter_quant/dflash_w4_gptq}"

N_CONVS="${N_CONVS:-256}"
ANCHORS="${ANCHORS:-16}"
GROUP_SIZE="${GROUP_SIZE:-128}"
PERCDAMP="${PERCDAMP:-0.01}"
# ----------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$HERE/src/drafter_quant/quant_dflash_gptq.py" \
  --calib   "$CALIB" \
  --target  "$TARGET" \
  --drafter "$DRAFTER" \
  --output  "$OUT" \
  --n-convs "$N_CONVS" --anchors-per-conv "$ANCHORS" \
  --group-size "$GROUP_SIZE" --percdamp "$PERCDAMP"

# vLLM trust_remote_code loads the draft via its dflash.py custom module.
cp "$DRAFTER/dflash.py" "$OUT/dflash.py"
echo "[done] quantized drafter at: $OUT"
