#!/usr/bin/env bash
# pack_ct.sh — Stage 2 (Target Model): pack a trained BF16+amax QAD checkpoint
# into (a) the compressed-tensors INT4 target that vLLM 0.22.1 serves and
# (b) its dense-BF16 twin (dequantized on-grid weights) that the drafter
# fine-tune / GPTQ-quant steps use as their training/calib target.
#
# Usage:
#   scripts/pack_ct.sh <CKPT_SUBDIR> [DST]
#     CKPT_SUBDIR  e.g. checkpoint-5000 or final — resolved under
#                  runs/qad_nemotron_regen/{ckpts_full,ckpts_smoke,ckpts}
#                  (full run vs smoke run output dirs of train_multi.sh)
#     DST          output dir. Default derives from CKPT_SUBDIR:
#                  checkpoint-5000 → runs/qad_nemotron_regen/local_model_ct_ckpt5000
#                  (the name serve_config.yaml / serve_dflash_gptq.yaml expect).
#                  The dense twin always lands at ${DST}-bf16.
#
# Example:
#   scripts/pack_ct.sh checkpoint-5000
#     → runs/qad_nemotron_regen/local_model_ct_ckpt5000        (serve target)
#     → runs/qad_nemotron_regen/local_model_ct_ckpt5000-bf16   (drafter target)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Pull HF_TOKEN / HF_HOME from .env (base-model sidecars come from the HF cache).
if [[ -f .env ]]; then set -a; source .env; set +a; export HF_TOKEN="${HF_TOKEN:-}"; fi
if [[ "${HF_HOME:-}" == hf_* ]]; then unset HF_HOME; fi

CKPT="${1:?usage: scripts/pack_ct.sh <CKPT_SUBDIR> [DST]}"
BASE="${REPO_ROOT}/runs/qad_nemotron_regen"

SRC=""
for d in ckpts_full ckpts_smoke ckpts; do
    if [[ -d "${BASE}/${d}/${CKPT}" ]]; then
        SRC="${BASE}/${d}/${CKPT}"
        break
    fi
done
if [[ -z "${SRC}" ]]; then
    echo "pack_ct.sh: no checkpoint '${CKPT}' under ${BASE}/{ckpts_full,ckpts_smoke,ckpts}" >&2
    exit 2
fi

TAG="${CKPT/#checkpoint-/ckpt}"                    # checkpoint-5000 → ckpt5000
DST="${2:-${BASE}/local_model_ct_${TAG}}"

echo "[pack_ct.sh] src: ${SRC}"
echo "[pack_ct.sh] dst: ${DST}  (+ ${DST}-bf16)"

exec .venv/bin/python src/qad/pack_bf16_to_ct.py \
    --src "${SRC}" \
    --dst "${DST}" \
    --dense-bf16-dst "${DST}-bf16" \
    --base-model Qwen/Qwen3.5-4B \
    --stitch-mtp-from cyankiwi
