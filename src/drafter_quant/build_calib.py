"""Stage 0 (drafter quant) — DFlash GPTQ calibration *seed corpus* builder.

Resamples the QAD corpus (data_generator's `qwen3_5_nemotron_combined_regen.jsonl`)
into a calibration seed set that mimics the model's eval-time generation distribution.

Decisions:
  - mix = eval-matched proportions (science 384 / general 384 / math 128 / code 128
    = 1024 conversations); proportions derived from the eval-set category mix.
    Data is ONLY ever sampled from the regen jsonl — never from the eval set.
  - think policy = FLATTEN: strip <think>...</think> TAGS but keep the reasoning
    text inline before the answer (question -> reasoning -> answer as one plain text).
  - length cutoff: drop conversations whose rendered token count (Qwen3.5-4B
    tokenizer, qwen_no_think.jinja, add_generation_prompt=False) exceeds 12288.
  - classification: 100% content heuristic (source has no category label; id is md5).
    Priority code > math > science > general.
  - output is shuffled (seed) so any prefix (e.g. the first 256 convs the GPTQ run
    uses) preserves the eval mix — no category ordering.

Output (--out dir):
  calib_v1.jsonl  — one record per conversation:
      {"src_id","bucket","n_tokens","n_assistant_tokens","prompt_n_tokens",
       "messages":[{role,content}x3],   # think-flattened
       "rendered_full": "...",          # add_generation_prompt=False (teacher-forced)
       "rendered_prompt": "..."}        # sys+user, add_generation_prompt=True
  summary.json    — mix, per-bucket counts, token percentiles, drop stats, seed.

Usage (.venv has transformers; CPU-only):
  .venv/bin/python src/drafter_quant/build_calib.py \
      --out runs/drafter_quant/calib [--mix eval] [--total 1024] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import hashlib
import random
from pathlib import Path

# FINAL calib source = the QAD-generated 400K set (regen_ckpt5000.jsonl): the QAD-applied
# W4A16 target's OWN outputs, i.e. the drafter's training distribution → most aligned for
# drafter PTQ calibration. (dev source = the 220K BF16-original set
# data/qwen3_5_nemotron_combined_regen.jsonl — pass via --source to A/B it.)
SOURCE = "src/data_generator/regen_ckpt5000.jsonl"
# Qwen3.5-4B tokenizer == the target model's tokenizer (HF id; cached via HF_HOME).
TOKENIZER = "Qwen/Qwen3.5-4B"
# No-think chat template (in-repo copy).
TEMPLATE = "src/templates/qwen_no_think.jinja"
MAX_TOKENS = 12288

# Eval-matched bucket proportions with per-bucket floor 128.
MIX_EVAL = {"science": 384, "general": 384, "math": 128, "code": 128}
MIX_BAL = {"science": 256, "general": 256, "math": 256, "code": 256}

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
RE_CODE_FENCE = re.compile(r"```(?:python|py|cpp|c\+\+|c|js|javascript|ts|java|go|rust|sh|bash|html|css|sql|json)?\b", re.I)
RE_CODE_SIG = re.compile(
    r"(?:\bdef\s+\w+\s*\(|\bclass\s+\w+\s*[\(:]|\bimport\s+\w|\bfrom\s+\w+\s+import\b"
    r"|console\.log\(|printf\(|System\.out\.println|=>\s*\{|function\s*\(|std::"
    r"|\bpublic\s+(?:static\s+)?(?:void|int|String)\b)", re.I)
RE_MATH_LATEX = re.compile(r"\\\(|\\\[|\\frac\{|\\sum_|\\int_|\\boxed|\$\$|\\begin\{|\\sqrt\{|\\partial")
RE_MATH_WORDS = re.compile(
    r"\b(solve|derive|derivative|integral|integrate|equation|prove|theorem|"
    r"polynomial|matrix|probability|factorize|differentiate)\b", re.I)
RE_SCI = re.compile(
    r"\b(biology|chemistry|chemical|physics|biological|molecular|molecule|enzyme|"
    r"protein|cell|organism|gene|genome|quantum|thermodynam|voltage|reaction|atom|"
    r"electron|velocity|acceleration|wavelength|clinical|diagnosis|patient|disease|"
    r"engineering|circuit|torque|momentum|psycholog|cognitive)\b", re.I)
RE_MCQ = re.compile(r"which of the following|\(A\)|\bOptions?:", re.I)


def flatten_think(content: str) -> str:
    """Strip <think>/</think> TAGS but keep the reasoning text inline before the answer.

    Handles every corpus shape we feed in:
      - both tags: reasoning between the tags, answer after </think>.
      - </think> ONLY — the QAD 400K regen (regen_ckpt5000.jsonl): the opening <think>
        lives in the prompt template, so the saved assistant text is
        "reasoning … </think>\\n\\n answer"; split on </think>.
        (length-truncated rows have neither tag — reasoning only, returned as-is.)
      - <think> only / neither: strip any stray tag, keep the text.
    """
    if "<think>" in content and "</think>" in content:
        m = THINK_RE.search(content)
        if m:
            reasoning = m.group(1).strip()
            answer = content[m.end():].strip()
            return (reasoning + "\n\n" + answer).strip() if reasoning else answer
    if "</think>" in content:
        reasoning, _, answer = content.partition("</think>")
        reasoning, answer = reasoning.strip(), answer.strip()
        return (reasoning + "\n\n" + answer).strip() if reasoning else answer
    return content.replace("<think>", "").strip()


def classify(user: str, asst: str) -> str:
    blob = user + "\n" + asst
    low = blob.lower()
    if RE_CODE_FENCE.search(blob) or RE_CODE_SIG.search(blob):
        return "code"
    if RE_MATH_LATEX.search(blob) or len(RE_MATH_WORDS.findall(low)) >= 2:
        return "math"
    if RE_SCI.search(low) or RE_MCQ.search(blob):
        return "science"
    return "general"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--template", default=TEMPLATE)
    ap.add_argument("--mix", choices=["eval", "balanced"], default="eval")
    ap.add_argument("--total", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--candidate-factor", type=float, default=3.0)
    ap.add_argument("--max-scan", type=int, default=400000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "build_calib.log", "w")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        print(msg, file=log_f, flush=True)

    base = MIX_EVAL if args.mix == "eval" else MIX_BAL
    scale = args.total / sum(base.values())
    target = {k: max(128, round(v * scale)) for k, v in base.items()}
    log(f"mix={args.mix} target per bucket={target} (sum={sum(target.values())})")

    from transformers import AutoTokenizer
    log("loading tokenizer", args.tokenizer)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    jinja = Path(args.template).read_text()

    def render(messages, add_gen):
        return tok.apply_chat_template(
            messages, chat_template=jinja, tokenize=False,
            add_generation_prompt=add_gen)

    cap = {k: int(round(v * args.candidate_factor)) for k, v in target.items()}
    cand = {k: [] for k in target}
    seen_hash = set()
    n_scanned = n_parsed = n_flattened = n_drop_len = n_drop_dup = 0

    def buckets_full():
        return all(len(cand[k]) >= cap[k] for k in target)

    log("streaming", args.source)
    with open(args.source) as f:
        for line in f:
            if n_scanned >= args.max_scan or buckets_full():
                break
            n_scanned += 1
            try:
                r = json.loads(line)
            except Exception:
                continue
            conv = r.get("conversations")
            if not conv or len(conv) < 2:
                continue
            d = {t.get("role"): t.get("content", "") for t in conv if isinstance(t, dict)}
            user = (d.get("user") or "").strip()
            asst_raw = d.get("assistant") or ""
            system = (d.get("system") or "You are a helpful assistant.").strip()
            if not user or not asst_raw.strip():
                continue
            n_parsed += 1

            asst = flatten_think(asst_raw)
            if "<think>" in asst_raw or "</think>" in asst_raw:
                n_flattened += 1   # count either tag (400K regen carries only </think>)

            bucket = classify(user, asst)
            if len(cand[bucket]) >= cap[bucket]:
                continue

            uh = hashlib.md5(re.sub(r"\s+", " ", user.lower()).encode()).hexdigest()
            if uh in seen_hash:
                n_drop_dup += 1
                continue

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": asst},
            ]
            try:
                rendered_full = render(messages, add_gen=False)
                rendered_prompt = render(messages[:2], add_gen=True)
            except Exception:
                continue
            n_tok = len(tok(rendered_full, add_special_tokens=False).input_ids)
            if n_tok > args.max_tokens:
                n_drop_len += 1
                continue
            prompt_ntok = len(tok(rendered_prompt, add_special_tokens=False).input_ids)
            seen_hash.add(uh)
            cand[bucket].append({
                "src_id": r.get("id"),
                "bucket": bucket,
                "n_tokens": n_tok,
                "n_assistant_tokens": max(0, n_tok - prompt_ntok),
                "prompt_n_tokens": prompt_ntok,
                "messages": messages,
                "rendered_full": rendered_full,
                "rendered_prompt": rendered_prompt,
            })
            if n_scanned % 4000 == 0:
                log(f"  scanned={n_scanned} parsed={n_parsed} cand={{ "
                    f"{', '.join(f'{k}:{len(cand[k])}/{cap[k]}' for k in target)} }}")

    log(f"scan done: scanned={n_scanned} parsed={n_parsed} flattened={n_flattened} "
        f"drop_len={n_drop_len} drop_dup={n_drop_dup}")
    for k in target:
        log(f"  candidates {k}: {len(cand[k])} (need {target[k]})")

    rng = random.Random(args.seed)
    final = []
    short = {}
    for k in target:
        pool = cand[k]
        rng.shuffle(pool)
        take = min(target[k], len(pool))
        if take < target[k]:
            short[k] = target[k] - take
        final.extend(pool[:take])
    rng.shuffle(final)  # mix shuffle: any prefix preserves the eval proportions

    with open(out / "calib_v1.jsonl", "w") as wf:
        for rec in final:
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    toks = [r["n_tokens"] for r in final]
    atoks = [r["n_assistant_tokens"] for r in final]
    per_bucket = {k: sum(1 for r in final if r["bucket"] == k) for k in target}

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0

    summary = {
        "mix_mode": args.mix,
        "think_policy": "flatten",
        "target_per_bucket": target,
        "actual_per_bucket": per_bucket,
        "total": len(final),
        "shortfall": short,
        "n_scanned": n_scanned,
        "n_parsed": n_parsed,
        "n_flattened": n_flattened,
        "n_dropped_len": n_drop_len,
        "n_dropped_dup": n_drop_dup,
        "n_tokens": {"p50": pct(toks, .5), "p95": pct(toks, .95), "max": max(toks) if toks else 0},
        "n_assistant_tokens": {"p50": pct(atoks, .5), "p95": pct(atoks, .95),
                               "max": max(atoks) if atoks else 0},
        "max_tokens_cutoff": args.max_tokens,
        "seed": args.seed,
        "source": args.source,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log("SUMMARY:", json.dumps(summary, ensure_ascii=False))
    if short:
        log("WARNING shortfall:", short)
    ok = len(final) >= 512 and all(per_bucket[k] >= 128 for k in target) and (max(toks) if toks else 0) <= args.max_tokens
    log("PASS" if ok else "CHECK: constraints not fully met")
    log_f.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
