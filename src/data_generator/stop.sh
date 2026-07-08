#!/bin/bash
# Stop all vLLM servers and any running generator, then show GPU memory.
set -u
echo "stopping generator + vLLM servers..."
pkill -9 -f 'generate_responses.py' 2>/dev/null
for r in 1 2 3; do
  pids=$(pgrep -f 'vllm serve|EngineCore|DPCoordinator|ApiServer|VLLM::')
  [ -z "$pids" ] && break
  for p in $pids; do kill -9 "$p" 2>/dev/null; done
  sleep 5
done
pgrep -f 'vllm serve|EngineCore|DPCoordinator|VLLM::' >/dev/null && echo "WARNING: some procs survived" || echo "all down"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
