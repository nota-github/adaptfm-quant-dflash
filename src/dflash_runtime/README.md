> **Vendored into edgefm-eqc as `src/dflash_runtime/`** — supplies the `dflash` Python
> package (installed editable by `uv sync`): the draft-model implementation that vLLM's
> `method: dflash` speculative decoding imports at serve/eval time (`dflash/model.py`)
> plus the acceptance-length benchmark (`dflash/benchmark.py` — see **Evaluation** below
> for the in-repo usage). Other upstream modules (e.g. the MLX port `dflash.model_mlx`)
> are not included. The upstream README below is kept for reference.

# DFlash: Block Diffusion for Flash Speculative Decoding
[**Paper**](https://arxiv.org/abs/2602.06036) | [**Blog**](https://z-lab.ai/projects/dflash/) | [**Models**](https://huggingface.co/collections/z-lab/dflash)

**DFlash** is a lightweight **block diffusion** model designed for speculative decoding. It enables efficient and high-quality parallel drafting.

![DFlash Architecture](https://raw.githubusercontent.com/jianc99/jianc99.github.io/master/images/dflash_system.png)

https://github.com/user-attachments/assets/5b29cabb-eb95-44c9-8ffe-367c0758de8c

## Supported Models

| Model | DFlash Draft |
|---|---|
| gemma-4-31B-it | [z-lab/gemma-4-31B-it-DFlash](https://huggingface.co/z-lab/gemma-4-31B-it-DFlash) |
| gemma-4-26B-A4B-it | [z-lab/gemma-4-26B-A4B-it-DFlash](https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash) |
| MiniMax-M2.7 (Preview) | [z-lab/MiniMax-M2.7-DFlash](https://huggingface.co/z-lab/MiniMax-M2.7-DFlash) |
| MiniMax-M2.5 (Preview) | [z-lab/MiniMax-M2.5-DFlash](https://huggingface.co/z-lab/MiniMax-M2.5-DFlash) |
| Kimi-K2.6 (Preview) | [z-lab/Kimi-K2.6-DFlash](https://huggingface.co/z-lab/Kimi-K2.6-DFlash) |
| Kimi-K2.5 | [z-lab/Kimi-K2.5-DFlash](https://huggingface.co/z-lab/Kimi-K2.5-DFlash) |
| Qwen3.6-27B | [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) |
| Qwen3.6-35B-A3B | [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) |
| Qwen3.5-4B | [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) |
| Qwen3.5-9B | [z-lab/Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) |
| Qwen3.5-27B | [z-lab/Qwen3.5-27B-DFlash](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash) |
| Qwen3.5-35B-A3B | [z-lab/Qwen3.5-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash) |
| Qwen3.5-122B-A10B | [z-lab/Qwen3.5-122B-A10B-DFlash](https://huggingface.co/z-lab/Qwen3.5-122B-A10B-DFlash) |
| gpt-oss-20b | [z-lab/gpt-oss-20b-DFlash](https://huggingface.co/z-lab/gpt-oss-20b-DFlash) |
| gpt-oss-120b | [z-lab/gpt-oss-120b-DFlash](https://huggingface.co/z-lab/gpt-oss-120b-DFlash) |
| Qwen3-Coder-Next | [z-lab/Qwen3-Coder-Next-DFlash](https://huggingface.co/z-lab/Qwen3-Coder-Next-DFlash) |
| Qwen3-4B (non-thinking) | [z-lab/Qwen3-4B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16) |
| Qwen3-8B (non-thinking) | [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) |
| Qwen3-Coder-30B-A3B | [z-lab/Qwen3-Coder-30B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3-Coder-30B-A3B-DFlash) |
| Llama-3.1-8B-Instruct | [z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat](https://huggingface.co/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat) |
| DeepSeek-V4-Flash | Coming soon |
| DeepSeek-V4-Pro | Coming soon |
| GLM-5.1 | Coming soon |

> Feel free to open a GitHub issue to request support for additional models. We will also open-source the training recipe soon, so you can train your own DFlash draft model to accelerate any LLM.

## 📦 Installation

Use a separate virtual environment for each to avoid conflict.

| Backend | Install command |
|---|---|
| **Transformers** | `uv pip install -e ".[transformers]"` |
| **SGLang** | `uv pip install -e ".[sglang]"` |
| **vLLM** | See below |
| **MLX** (Apple Silicon) | not vendored — see upstream |

**vLLM:** vLLM v0.20.1+ includes core DFlash support. Use the standard install for most models:
```bash
uv pip install -e ".[vllm]"
```

Gemma4 DFlash currently needs our temporary vLLM Gemma4 build. Docker is recommended:
```bash
docker pull ghcr.io/z-lab/vllm-openai:gemma4-dflash-cu130
```

Source fallback for Gemma4:
```bash
uv pip install -U --torch-backend=auto \
  "vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/41703/head"
```

Newer non-Gemma4 SWA draft models use the SWA support branch:
```bash
uv pip install -U --torch-backend=auto \
  "vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/40898/head"
```

## 🚀 Quick Start

### vLLM

Gemma4 with Docker:
```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/z-lab/vllm-openai:gemma4-dflash-cu130 \
  google/gemma-4-26B-A4B-it \
  --host 0.0.0.0 \
  --port 8000 \
  --speculative-config '{"method": "dflash", "model": "z-lab/gemma-4-26B-A4B-it-DFlash", "num_speculative_tokens": 15, "attention_backend": "flash_attn"}' \
  --attention-backend triton_attn \
  --max-num-batched-tokens 32768 \
  --trust-remote-code
```

Non-Gemma4 models:
```bash
vllm serve Qwen/Qwen3.5-27B \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.5-27B-DFlash", "num_speculative_tokens": 15}' \
  --attention-backend flash_attn \
  --max-num-batched-tokens 32768
```

### SGLang

```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# Optional: enable schedule overlapping (experimental, may not be stable)
# export SGLANG_ENABLE_SPEC_V2=1
# export SGLANG_ENABLE_DFLASH_SPEC_V2=1
# export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-35B-A3B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3.5-35B-A3B-DFlash \
    --speculative-num-draft-tokens 16 \
    --tp-size 1 \
    --attention-backend trtllm_mha \
    --speculative-draft-attention-backend fa4 \
    --mem-fraction-static 0.75 \
    --mamba-scheduler-strategy extra_buffer \
    --trust-remote-code
```

### Transformers

Only Qwen3 and LLaMA-3.1 models support the Transformers backend.

```python
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

draft = AutoModel.from_pretrained("z-lab/Qwen3-8B-DFlash-b16", trust_remote_code=True, dtype="auto", device_map="cuda:0").eval()
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype="auto", device_map="cuda:0").eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

messages = [{"role": "user", "content": "How many positive whole-number divisors does 196 have?"}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False).to(draft.device)

output = draft.spec_generate(input_ids=input_ids, max_new_tokens=2048, temperature=0.0, target=target, stop_token_ids=[tokenizer.eos_token_id])
print(tokenizer.decode(output[0], skip_special_tokens=False))
```

### MLX (Apple Silicon)

> `dflash.model_mlx` is **not included** in this vendored copy — see the upstream
> repository for the MLX port.

## 📊 Evaluation

`dflash/benchmark.py` measures decoding throughput and **mean acceptance length**
(MAL — accepted tokens + 1 per draft step). Datasets (gsm8k, math500, humaneval,
mbpp, mt-bench, longbench = LongBench-v2) are downloaded and cached as JSONL in
`cache/` on first run. The edgefm-eqc protocol is **64 prompts / 256 new tokens per
benchmark**, averaged over `gsm8k` + `humaneval` + `longbench`.

For `longbench`, pre-build the cache with the tokenizer-exact 8192-token context
truncation before the first run (both backends read the same cache):

```bash
.venv/bin/python scripts/prepare_longbench_bench.py   # from the repo root; once
```

Without it, first use falls back to a coarser 32,768-char truncation; re-running the
script overwrites a stale fallback cache.

Two backends are exercised in this repo (the `sglang`/`mlx` code paths are upstream
reference — sglang is not installed in the unified venv and `dflash.model_mlx` is
not vendored):

**vLLM** — measures the final serving stack (e.g. W4 target + W4 draft + SWA). Point
`--base-url` at the **internal vLLM port** (`EQC_VLLM_INTERNAL_PORT`, e.g. 28080),
not the :8080 proxy: MAL is derived from vLLM's `/metrics` spec-decode counters,
which the proxy does not forward. `--model` must be the served name
(`Qwen/Qwen3.5-4B`), and the server must be otherwise idle (the counters are
server-wide). `longbench` prompts need a context window ≥ ~9216
(`EQC_VLLM_MAX_MODEL_LEN=16384` as in the top-level README works).

```bash
.venv/bin/python -m dflash.benchmark --backend vllm \
    --base-url http://127.0.0.1:28080 --model Qwen/Qwen3.5-4B \
    --dataset gsm8k --num-prompts 64 --max-new-tokens 256 --concurrency 1
```

**Transformers** — offline target+draft pair, for drafter ablations (e.g. target =
`Qwen/Qwen3.5-4B` or the dense-bf16 QAD twin from `scripts/pack_ct.sh`; draft = a
SpecForge checkpoint or `z-lab/Qwen3.5-4B-DFlash`):

```bash
.venv/bin/python -m dflash.benchmark --backend transformers \
    --model runs/qad_nemotron_regen/local_model_ct_ckpt5000-bf16 \
    --draft-model src/SpecForge/outputs/dflash-finetune/<best-ckpt> \
    --dataset gsm8k --max-samples 64 --max-new-tokens 256
```

Note the two backends pick their prompt subsets differently (server path: first N in
cache-file order after a warmup batch; transformers path: seeded random N), so small
per-dataset deltas between backends are expected.

## Acknowledgement

Huge thanks to [@dcw02](https://github.com/dcw02), [@gongy](https://github.com/gongy), and the team at [@modal-labs](https://github.com/modal-labs) for their fast, high-quality support in bringing DFlash to SGLang. And huge thanks as well to [@benchislett](https://github.com/benchislett) at NVIDIA for his work in bringing DFlash to vLLM and helping make it available to the broader serving community.

## Citation
If you find DFlash useful, please cite our work. To share feedback on DFlash or request new model support, please fill out this form: [DFlash Feedback](https://forms.gle/4YNwfqb4nJdqn6hq9).

```bibtex
@article{chen2026dflash,
  title   = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}
```
