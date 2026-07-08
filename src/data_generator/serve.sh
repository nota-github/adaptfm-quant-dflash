#!/bin/bash
# Launch N independent single-GPU vLLM servers (one per GPU, one port each).
#
# WHY independent servers instead of --data-parallel-size N:
#   vLLM DP on this model deadlocks under sustained load (all engines log
#   "shm_broadcast: No available shared memory broadcast block found in 60s",
#   /health stays 200 but generation hangs). One DP=1 server per GPU has no
#   cross-engine coordination, so it cannot deadlock. The client (generate_responses.py)
#   round-robins across the ports via a comma-separated --base-url.
#
# Usage:  MODEL=/path/to/teacher ./serve.sh         # defaults: GPUs 0 1 2 3 -> ports 8123..8126
#         MODEL=... GPUS="2 3 4 7" PORTS="8124 8125 8126 8127" ./serve.sh
#
# Prereq on B300/sm_103 (spec decode CUTE kernel): see README "CUTE patch".
set -u

# --- environment-specific: override these per machine/run (defaults are examples) ---
MODEL="${MODEL:-/path/to/teacher}"                  # teacher checkpoint to serve (REQUIRED)
SERVED_NAME="${SERVED_NAME:-teacher}"               # name clients pass as --model
GPUS="${GPUS:-0 1 2 3}"                             # one server per GPU
PORTS="${PORTS:-8123 8124 8125 8126}"               # one port per GPU (same length as GPUS)
CACHE="${CACHE:-$HOME/.cache}"                      # keep HF/vLLM/Triton caches on a fast local disk
# Unified repo venv (vLLM 0.22.1; built by `uv sync` — see ../../pyproject.toml).
# Resolves repo-root/.venv (two levels up) by default.
VENV="${VENV:-$(cd "$(dirname "$0")/../.." && pwd)/.venv}"
# --- model/perf knobs (usually fine as-is) ---
# SPEC=on enables dflash spec-decode (~throughput boost). DEFAULT OFF: on the
# qad_nemotron_regen W4A16 checkpoints the dflash/EAGLE draft (which shares the
# target's quantized embed/lm_head) CORRUPTS output into repetition/garbage. Verified
# clean with SPEC=off. Only turn on after confirming a given checkpoint stays coherent.
SPEC="${SPEC:-off}"
DRAFT="${DRAFT:-z-lab/Qwen3.5-4B-DFlash}"           # dflash spec-decode draft (used only when SPEC=on)
NUM_SPEC="${NUM_SPEC:-10}"
LOGDIR="${LOGDIR:-$(pwd)}"

SPEC_ARGS=()
[ "$SPEC" = "on" ] && SPEC_ARGS=(--speculative-config "{\"model\": \"$DRAFT\", \"num_speculative_tokens\": $NUM_SPEC}")

if [ ! -x "$VENV/bin/vllm" ]; then
  echo "ERROR: vllm not found at $VENV/bin/vllm" >&2
  echo "  build it: (cd $(cd "$(dirname "$0")/../.." && pwd) && uv sync)" >&2
  exit 1
fi
if [ "$MODEL" = "/path/to/teacher" ]; then
  echo "ERROR: set MODEL=/path/to/teacher (W4A16 Qwen3.5 checkpoint)" >&2; exit 1
fi

mkdir -p "$CACHE/huggingface/hub" "$CACHE/triton" "$CACHE/torchinductor"
read -ra G <<< "$GPUS"; read -ra P <<< "$PORTS"

for i in "${!G[@]}"; do
  g="${G[$i]}"; p="${P[$i]}"
  # Each independent server needs a DISTINCT internal port base, else their
  # torch.distributed rendezvous races on the same get_open_port() pick and dies
  # with EADDRINUSE when launched together. VLLM_PORT seeds vLLM's internal ports;
  # space bases by 20 (each server uses a few). Stagger launches too, as insurance.
  vp=$((28000 + i * 20))
  CUDA_VISIBLE_DEVICES="$g" VLLM_PORT="$vp" \
  HF_HOME="$CACHE/huggingface" HF_HUB_CACHE="$CACHE/huggingface/hub" HUGGINGFACE_HUB_CACHE="$CACHE/huggingface/hub" \
  XDG_CACHE_HOME="$CACHE" VLLM_CACHE_ROOT="$CACHE/vllm_g$g" TRITON_CACHE_DIR="$CACHE/triton" TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor" \
  setsid nohup "$VENV/bin/vllm" serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --port "$p" --gpu-memory-utilization 0.90 \
    --data-parallel-size 1 --tensor-parallel-size 1 \
    --linear-backend marlin --max-model-len 32768 \
    --limit-mm-per-prompt '{"image":0,"video":0}' --max-num-seqs 512 \
    --max-num-batched-tokens 32768 \
    "${SPEC_ARGS[@]}" \
    > "$LOGDIR/vllm_g${g}_p${p}.log" 2>&1 < /dev/null &
  disown
  echo "launched GPU $g -> port $p (VLLM_PORT=$vp, log: vllm_g${g}_p${p}.log)"
  sleep 4
done

echo
echo "Loading takes ~3-4 min. Wait for real readiness (NOT just /health 200 — that returns"
echo "early in multi-engine mode) with a generation probe, e.g.:"
echo '  for p in '"$PORTS"'; do curl -s http://localhost:$p/v1/chat/completions \'
echo '    -H "Content-Type: application/json" \'
echo '    -d "{\"model\":\"'"$SERVED_NAME"'\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":4}"; done'
