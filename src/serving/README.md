# serving — standalone submission server

Faithful port of the competition submission server, self-contained under `src/serving/`.
The package is not installed into the venv — run it from the repo root with
`PYTHONPATH=src` so `python -m serving` resolves.

A thin **FastAPI proxy** that, on startup, spawns `vllm serve <model>` as a
subprocess on an internal port and forwards the SageMaker submission contract to
it. We proxy rather than call `vllm.LLM` directly because the offline class
deadlocks under FastAPI's threadpool at lm-eval concurrency and doesn't expose
logprobs/echo for GPQA multi-choice.

## Endpoints (submission contract, port 8080)

| endpoint | use |
|---|---|
| `GET /ping` | 200 once vllm `/health` is up |
| `POST /invocations` | polymorphic: `{"prompt",...}` → `/v1/completions`, `{"messages",...}` → `/v1/chat/completions` |
| `POST /v1/completions` | latency benchmark (raw prompt) |
| `POST /v1/chat/completions` | quality benchmark (chat template) |
| `GET /v1/models` | model id discovery |

**Thinking control** (`app.py:_maybe_transform_invocations`): chat `/invocations`
traffic folds the harness's top-level `"thinking": true` (GPQA) into
`chat_template_kwargs={"enable_thinking": true}` (the only form vllm understands),
and **defaults thinking OFF** for chat traffic that says nothing (MMLU-Pro/IFEval).

## Files

| file | role |
|---|---|
| `app.py` | FastAPI proxy + vllm subprocess lifecycle + retry/stream logic |
| `__main__.py` | `PYTHONPATH=src python -m serving` → uvicorn entrypoint |
| `config.py` | permissive YAML loader (merged onto defaults) |

> DFlash drafter knobs (W4A16 quant loading + attention sliding-window) live in
> the **`src/vllm_plugin`** vLLM general plugin — installed into the serving
> venv, they load in every process (incl. vllm's spawned EngineCore) via the entry
> point, so no `.pth` shim is needed. See "DFlash drafter plugin" below.

## Setup

Build the single unified project venv from the repo root (see `pyproject.toml`):

```bash
uv sync            # torch 2.11 / transformers 5.5.x / vLLM 0.22.1 + the DFlash plugin
```

This serving path runs on **vLLM 0.22.1** (the DFlash speculative-decode line), not
the old 0.19 base image — 0.22.1 is required for the `method: dflash` draft and the
`EQC_DFLASH_QUANT_PATCH` W4A16 draft loader.

## Run

```bash
cd /path/to/edgefm-eqc          # repo root; put .venv/bin on PATH so `vllm` resolves

# Single GPU = A10G submission target (latency-faithful).
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src \
EQC_CONFIG_PATH=src/configs/serve_config.yaml \
EQC_PORT=8080 \
EQC_VLLM_INTERNAL_PORT=28080 \
EQC_GPU_MEM_CAP_FRACTION=0.30 \
  .venv/bin/python -m serving
# wait for: curl -s http://localhost:8080/ping  → {"status":"ok"}
```

`src/configs/serve_config.yaml` already points `model:` at
`runs/qad_nemotron_regen/local_model_ct_ckpt5000` (the packed CT ckpt — name encodes
the source step).

### Quality eval across GPUs 0,1,2,3 (data-parallel)

vllm load-balances lm-eval requests across replicas, so the long MMLU-Pro pole
keeps all 4 cards busy. Quality is throughput, not the single-A10G latency path.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=src \
EQC_CONFIG_PATH=src/configs/serve_config.yaml \
EQC_PORT=8080 EQC_VLLM_INTERNAL_PORT=28080 \
EQC_VLLM_DP_SIZE=4 EQC_VLLM_MAX_NUM_SEQS=8 \
EQC_VLLM_MAX_MODEL_LEN=16384 \
  .venv/bin/python -m serving
```

Then drive it with the grader-equivalent harness (talks HTTP only):

```bash
# dev sample first, then full
QUALITY_LIMIT=0.1 .venv/bin/python src/eval_harness/run_quality_local.py
QUALITY_LIMIT=1.0 NUM_CONCURRENT=8 .venv/bin/python src/eval_harness/run_quality_local.py
# latency (run against the single-GPU server)
CONTAINER_URL=http://localhost:8080 .venv/bin/python src/eval_harness/run_latency_harness.py
```

## Env knobs

| var | default | meaning |
|---|---|---|
| `EQC_PORT` | 8080 | user-facing contract port |
| `EQC_VLLM_INTERNAL_PORT` | `EQC_PORT+10000` | internal vllm port. **Set to 28080** — the harness's MMLU-Pro proxy opens 18080 (= 8080+10000) and would collide |
| `EQC_GPU_MEM_CAP_FRACTION` | 0.78 | gpu-mem-util. **0.30 on A100 80GB** to emulate A10G 24 GB |
| `EQC_VLLM_DP_SIZE` | unset | `--data-parallel-size N` (quality; one replica/GPU) |
| `EQC_VLLM_MAX_NUM_SEQS` | config | batch cap (8 for quality, 1 for latency) |
| `EQC_VLLM_MAX_MODEL_LEN` | config (8704) | raise to **16384** for GPQA-D's 12288-tok thinking output |
| `EQC_DFLASH_QUANT_PATCH` | unset | enable loading a W4A16-quantized DFlash draft (`src/vllm_plugin`; required on vllm 0.22.1 + a `speculative_config` in YAML) |
| `EQC_DFLASH_SWA_WINDOW` | unset | DFlash drafter attention sliding-window, last N tokens (same plugin) |
| `EQC_USE_STUB=1` | — | canned responses, no vllm (contract smoke test) |

## config.yaml knobs (consumed by `_build_vllm_args`)

`model`, `quant` (→ `--quantization`), `tokenizer`, `trust_remote_code`,
`chat_template`, `speculative_config` (dict→JSON for `--speculative-config`),
and `vllm_args.{max_model_len,enforce_eager,max_num_seqs,dtype,
enable_prefix_caching,max_num_batched_tokens,kv_cache_dtype,attention_backend,
language_model_only,...}`.

## DFlash drafter plugin (optional, latency only)

The DFlash drafter patches (W4A16 quant loading + the two latency windows) are a
vLLM general plugin in `src/vllm_plugin`. It is installed **editable by
`uv sync`** (declared in `pyproject.toml`), so vLLM auto-loads it in every process
(incl. the spawned EngineCore) via its entry point — no extra install, no `.pth` shim:

```bash
# verify the entry point is registered:
.venv/bin/python -c "from importlib.metadata import entry_points as e; \
  print([x.name for x in e(group='vllm.general_plugins')])"   # → ['eqc_dflash_quant']
```

Then run the server with a `speculative_config: {method/model: <DFlash draft>,
num_speculative_tokens: K}` in the YAML and one or both of:

```bash
EQC_DFLASH_QUANT_PATCH=1   # load a W4A16-quantized draft (required on vllm 0.22.1)
EQC_DFLASH_SWA_WINDOW=<N>  # symmetric sliding-window over the drafter attention
```

With neither set the plugin's `register()` is a strict no-op, so it's safe to
leave installed.
