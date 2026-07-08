"""Pre-build the LongBench-v2 cache jsonl for the vendored dflash benchmark.

The benchmark (src/dflash_runtime/dflash/benchmark.py) reads rows of
{"turns": [...]} from src/dflash_runtime/cache/<dataset>.jsonl. LongBench contexts
are huge (~hundreds of K tokens), so we truncate each context to EXACTLY 8192
tokens (Qwen3.5 tokenizer) and append the question. Prompt ~= 8192 + question
tokens -> serve with a context window >= ~9216 (EQC_VLLM_MAX_MODEL_LEN=16384, as
in the top-level README's serve command, works).

Run this BEFORE the first `--dataset longbench` benchmark run: benchmark.py builds
a coarser fallback cache (32,768-char truncation) only when the cache file is
missing, and then reuses whichever cache exists. Running this script overwrites
any stale fallback cache with the tokenizer-exact one.

Run once with .venv (needs datasets + transformers). CPU only.
Usage (from the repo root):
    .venv/bin/python scripts/prepare_longbench_bench.py
Env:
    TOK_SRC   tokenizer id/dir (default: Qwen/Qwen3.5-4B — the QAD target shares it)
    OUT       output jsonl (default: src/dflash_runtime/cache/longbench.jsonl, where
              benchmark.py's CACHE_DIR looks)
    HF_HOME   HF cache location (dataset downloads here on first run)
"""
import json
import os
from pathlib import Path

TRUNC_TOKENS = 8192
TOK_SRC = os.environ.get("TOK_SRC", "Qwen/Qwen3.5-4B")  # tokenizer only; QAD shares it
# benchmark.py reads Path(<benchmark.py>).parent.parent / "cache" == src/dflash_runtime/cache.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO_ROOT / "src" / "dflash_runtime" / "cache" / "longbench.jsonl"
OUT = os.environ.get("OUT", str(_DEFAULT_OUT))


def main() -> int:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOK_SRC, trust_remote_code=True)
    ds = load_dataset("zai-org/LongBench-v2", split="train")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with open(OUT, "w") as f:
        for r in ds:
            ids = tok.encode(r["context"], add_special_tokens=False)[:TRUNC_TOKENS]
            ctx = tok.decode(ids, skip_special_tokens=True)
            prompt = ctx + "\n\n" + r["question"]
            f.write(json.dumps({"turns": [prompt]}) + "\n")
            n += 1
    print(f"wrote {n} rows -> {OUT} (context truncated to {TRUNC_TOKENS} tok + question)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
