# eval_harness

Grader-equivalent local eval harness. All scripts talk to a running submission
**server over HTTP** (default `http://localhost:8080`) — they do not load the
model in-process. Start the container/server first, then run a harness script.
Self-contained under the project root.

## Setup (once)

The eval deps ship in the unified project venv — build it from the repo root:

```bash
uv sync            # installs lm-eval 0.4.11 (+ everything else) into .venv
```

`lm-eval[api,ifeval]` is pinned in `pyproject.toml`. The `[math]` extra is
intentionally omitted (the three gates use no math-verify task, and it pulls an
`antlr4-runtime` version that clashes with modelopt). The latency harness needs
nothing beyond stdlib.

## Files

| file | what it does |
|---|---|
| `run_quality_local.py` | Quality gates: MMLU-Pro (5-shot, no-think), IFEval (0-shot, no-think), GPQA-Diamond (0-shot, **thinking ON**). Writes `QUALITY_RESULTS_PATH` (default `/tmp/quality_results.json`). |
| `run_eval_local.py` | Full harness — latency + quality. `EVAL_MODE=latency\|quality\|full`. Writes `/tmp/local_eval_results.json`. |
| `run_latency_harness.py` | Latency-only thin wrapper around `run_eval_local.py` (stubs lm-eval, so no deps needed). Defaults to `CONTAINER_URL=http://localhost:18096`. |

## Run

```bash
# 1. Start the submission server on :8080, wait for /ping to return 200.

# 2. Quality (dev sample first; QUALITY_LIMIT=1.0 for full grader-grade run)
QUALITY_LIMIT=0.1 .venv/bin/python src/eval_harness/run_quality_local.py
QUALITY_LIMIT=1.0 NUM_CONCURRENT=8 .venv/bin/python src/eval_harness/run_quality_local.py

# 3. Latency
CONTAINER_URL=http://localhost:8080 .venv/bin/python src/eval_harness/run_latency_harness.py

# 4. Everything
EVAL_MODE=full .venv/bin/python src/eval_harness/run_eval_local.py
```

## Env knobs

| var | default | meaning |
|---|---|---|
| `CONTAINER_URL` | `http://localhost:8080` (latency wrapper: `:18096`) | server base URL |
| `QUALITY_LIMIT` | `0.1` | fraction of questions; `1.0` = full eval |
| `NUM_CONCURRENT` | `8` | request concurrency for quality |
| `EVAL_MODE` | `quality` | `latency` \| `quality` \| `full` (run_eval_local.py) |
| `QUALITY_RESULTS_PATH` | `/tmp/quality_results.json` | quality output file |
| `HF_HUB_OFFLINE` / `HF_DATASETS_OFFLINE` | `1` | eval datasets are read from `HF_HOME` cache; pre-download them once online |

Quality eval needs the MMLU-Pro / IFEval / GPQA-Diamond datasets in the HF cache
(`HF_HOME`, default `~/.cache/huggingface`). Pre-fetch them once while online —
**GPQA (`Idavidrein/gpqa`) is gated**, so accept its terms on the HF page first and
run the fetch with your `HF_TOKEN` (MMLU-Pro and IFEval are public):

```bash
HF_TOKEN=hf_... .venv/bin/python - <<'PY'
from datasets import load_dataset
load_dataset("TIGER-Lab/MMLU-Pro")                # public
load_dataset("google/IFEval")                     # public
load_dataset("Idavidrein/gpqa", "gpqa_diamond")   # gated → HF_TOKEN required
PY
```
