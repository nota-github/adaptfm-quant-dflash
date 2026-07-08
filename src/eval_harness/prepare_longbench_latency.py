"""Build realistic latency prompts by slicing LongBench-v2 contexts.

Replaces the synthetic FILLER prompt in the latency harness with real prose,
sliced to EXACTLY the short/medium/long token budgets (Qwen3.5-4B tokenizer).
Picks N documents whose context is >= the longest budget (8192 tok) so the same
docs yield aligned short/medium/long slices, then writes a JSON the harness reads
(so the harness itself needs no tokenizer/datasets at eval time):

    {"short": [<N raw prompts>], "medium": [...], "long": [...], "meta": {...}}

Prompts are the raw sliced text (no chat template, no question appended) — this
matches the competition latency contract (raw /v1/completions, temp 0) and gives
exact prompt-length control for prefill-latency comparison.

Run once with .venv (needs datasets + transformers). CPU only.
Usage:
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python src/eval_harness/prepare_longbench_latency.py
Env:
    TOK_SRC   tokenizer id/dir (default: Qwen/Qwen3.5-4B — the QAD target shares it)
    N_DOCS    docs per category to emit (default 60 = 50 measured + a few warmup)
    OUT       output json (default: src/eval_harness/latency_prompts_longbench.json)
    HF_HOME   HF cache location (dataset downloads here on first run)
"""
import json
import os

# short/medium/long token budgets — must match PROMPT_CONFIGS in run_eval_local.py.
BUDGETS = {"short": 64, "medium": 2048, "long": 8192}
MAXBUD = max(BUDGETS.values())

TOK_SRC = os.environ.get("TOK_SRC", "Qwen/Qwen3.5-4B")   # tokenizer only; QAD shares it
N_DOCS = int(os.environ.get("N_DOCS", "60"))
OUT = os.environ.get(
    "OUT", os.path.join(os.path.dirname(__file__), "latency_prompts_longbench.json")
)


def main() -> int:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOK_SRC, trust_remote_code=True)
    ds = load_dataset("zai-org/LongBench-v2", split="train")
    print(f"LongBench-v2 loaded: {len(ds)} docs | need {N_DOCS} with >= {MAXBUD} ctx tok", flush=True)

    out = {cat: [] for cat in BUDGETS}
    kept = 0
    scanned = 0
    tok_lens = {cat: [] for cat in BUDGETS}
    for r in ds:
        scanned += 1
        ids = tok.encode(r["context"], add_special_tokens=False)
        if len(ids) < MAXBUD:
            continue  # need docs long enough for the 8192 slice
        for cat, n in BUDGETS.items():
            text = tok.decode(ids[:n], skip_special_tokens=True)
            out[cat].append(text)
            # record the actual re-tokenized length (decode->encode can drift a token or two)
            tok_lens[cat].append(len(tok.encode(text, add_special_tokens=False)))
        kept += 1
        if kept >= N_DOCS:
            break

    if kept < N_DOCS:
        print(f"WARNING: only {kept}/{N_DOCS} docs had >= {MAXBUD} tok (scanned {scanned})", flush=True)

    meta = {
        "source": "zai-org/LongBench-v2",
        "tokenizer": TOK_SRC,
        "budgets": BUDGETS,
        "n_docs": kept,
        "scanned": scanned,
        # sanity: re-tokenized length (should be ~= budget)
        "retok_len_sample": {cat: sorted(set(tok_lens[cat]))[:5] for cat in BUDGETS},
    }
    out["meta"] = meta
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"wrote {OUT}", flush=True)
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
