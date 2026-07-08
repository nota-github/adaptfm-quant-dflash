#!/usr/bin/env bash
# gpu_lock.sh — exclusive single-GPU lock + A10G-comparable env wrapper.
#
# Usage:
#   scripts/gpu_lock.sh <gpu_id> -- <cmd...>
#   scripts/gpu_lock.sh auto     -- <cmd...>
#
#   <gpu_id> ∈ {0..7, auto}.
#     "auto" scans GPU ids 0..7 and takes the first lock it can grab
#     non-blocking; if all are held it blocks waiting on GPU 0.
#
# Behavior:
#   - Takes an exclusive flock on runs/.locks/.gpu<id>.lock with a 14400s
#     (4 hr) timeout. Exits 124 if the timeout fires.
#   - After acquiring, exports:
#       CUDA_VISIBLE_DEVICES=<id>
#       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#       EQC_GPU_MEM_CAP_FRACTION=0.30   # 24/80, mimic A10G 24 GB on A100 80 GB
#       EQC_GPU_PHYSICAL_ID=<id>        # informational
#   - exec's the command so the shell process is replaced (clean PIDs).
#
# Note: from the application's POV, after CUDA_VISIBLE_DEVICES=<id> the
# physical device appears as `cuda:0`. Always address it that way in code.
#
# Deployment target is a SINGLE A10G — tensor-parallel inference is out of
# scope; this wrapper deliberately exposes exactly one GPU.

set -euo pipefail

# Defensive: a common .bashrc typo sets HF_HOME=<HF_TOKEN value>. Unset it so
# the default ~/.cache/huggingface cache is used.
if [[ "${HF_HOME:-}" == hf_* ]]; then
    unset HF_HOME
fi

# A100 80 GB → mimic the A10G 24 GB cap. Override with EQC_GPU_MEM_CAP_FRACTION.
MEM_CAP="${EQC_GPU_MEM_CAP_FRACTION:-0.30}"
GPU_IDS=(0 1 2 3 4 5 6 7)

usage() {
  cat >&2 <<'EOF'
Usage: scripts/gpu_lock.sh <gpu_id> -- <cmd...>
       scripts/gpu_lock.sh auto     -- <cmd...>

  <gpu_id> ∈ {0,1,2,3,4,5,6,7, auto}
EOF
  exit 2
}

if [[ $# -lt 3 ]]; then usage; fi
GPU_ARG="$1"; shift
if [[ "$1" != "--" ]]; then usage; fi
shift
if [[ $# -lt 1 ]]; then usage; fi

# Resolve repo root from this script's location (scripts/ -> repo root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCK_DIR="${REPO_ROOT}/runs/.locks"
mkdir -p "${LOCK_DIR}"

LOCK_TIMEOUT=14400  # 4 hours

lock_path_for() { printf '%s/.gpu%s.lock' "${LOCK_DIR}" "$1"; }

acquire_specific() {
  # acquire_specific <gpu_id> <blocking:true|false>
  local gpu_id="$1" blocking="$2" lock_file
  lock_file="$(lock_path_for "${gpu_id}")"
  [[ -e "${lock_file}" ]] || : > "${lock_file}"
  exec 9<>"${lock_file}"
  if [[ "${blocking}" == "true" ]]; then
    if ! flock -x -w "${LOCK_TIMEOUT}" 9; then
      echo "gpu_lock: timeout after ${LOCK_TIMEOUT}s waiting for GPU ${gpu_id} (${lock_file})" >&2
      exit 124
    fi
  else
    if ! flock -x -n 9; then exec 9<&-; return 1; fi
  fi
  echo "${gpu_id}"
}

resolve_gpu() {
  case "${GPU_ARG}" in
    0|1|2|3|4|5|6|7)
      acquire_specific "${GPU_ARG}" true
      ;;
    auto)
      local g
      for g in "${GPU_IDS[@]}"; do
        if acquire_specific "${g}" false 2>/dev/null; then echo "${g}"; return; fi
      done
      acquire_specific 0 true
      ;;
    *)
      usage
      ;;
  esac
}

GPU_ID="$(resolve_gpu)"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export EQC_GPU_MEM_CAP_FRACTION="${MEM_CAP}"
export EQC_GPU_PHYSICAL_ID="${GPU_ID}"

echo "gpu_lock: acquired GPU ${GPU_ID} (lock $(lock_path_for "${GPU_ID}"), mem_cap ${MEM_CAP}); exec: $*" >&2

# Replace the shell with the command so PIDs/signals are clean. The flock fd
# (9) is inherited by the child, keeping the lock held for its lifetime.
exec "$@"
