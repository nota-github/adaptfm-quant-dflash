#!/usr/bin/env python3
"""qad_data.py — JSONL → tokenized + packed dataset for QAD.

Supports two input formats (selected via ``--format``):

1. **jackrong** (default; small reasoning-distilled set, deprecated for
   final runs after the GPQA-D gate miss):
   `data/Jackrong__Qwen3.5-reasoning-700x/distilled_stage2.jsonl`
   shape ``{"conversation": [{"from": "human"/"gpt", "value": str}, ...]}``.

   For each Jackrong sample we emit TWO tokenized sequences:
   - **think** view (GPQA-Diamond, thinking ON eval): keep the
     `<think>…</think>` block, apply tokenizer's BUNDLED chat template.
   - **no_think** view (MMLU-Pro + IFEval, thinking OFF eval): pre-strip
     every `<think>…</think>` block, apply `src/templates/qwen_no_think.jinja`
     (emits empty `<think>\\n\\n</think>` shell + final answer).
   `--mode {both, only_no_think, only_think}` selects views; default both.

2. **train2** (larger, mixed reasoning + IF + code + math + science MCQ):
   `data/train2_v1_cleaned.jsonl`
   shape ``{"messages": [{"role": "system"/"user"/"assistant", "content": str}, ...]}``.
   ~49,547 records, ~80% contain `<think>` blocks naturally, ~15% math
   `\\boxed{}` problems. Multi-turn (2-9 messages).

   Each record emits a **single** view with the BUNDLED chat template,
   keeping `<think>` blocks intact. The natural mix of think/no-think
   records in the corpus alone produces the dual-mode coverage we used
   to manufacture in Jackrong via duplication. ``--mode`` is ignored
   for ``--format train2``.

Output (both formats): a torch.LongTensor of shape [N, seq_len] containing
packed input_ids. Position-level labels = same input_ids (full-sequence
KL distillation, the trainer recomputes them).

Pack policy: concatenate per-sample id streams with an `<|endoftext|>`
separator, split into fixed-size ``--max-seq-len`` chunks. The last
partial chunk is padded with pad_token_id (or eos if no pad).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator

import torch
from transformers import AutoTokenizer


SRC_ROOT = Path(__file__).resolve().parents[1]  # src/
REPO_ROOT = SRC_ROOT.parent
DEFAULT_INPUT = str(REPO_ROOT / "data" / "distilled_stage2.jsonl")
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "qad_nemotron_regen" / "train_data" / "packed.pt"
DEFAULT_TEMPLATE = SRC_ROOT / "templates" / "qwen_no_think.jinja"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", default="jackrong", choices=("jackrong", "train2"),
                   help="Input JSONL schema. 'jackrong' (default) expects the "
                        "Jackrong distilled_stage2 shape with `conversation`; "
                        "'train2' expects train2_v1_cleaned.jsonl with `messages`.")
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="Input JSONL path (default = Jackrong distilled).")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Output .pt file (packed input_ids tensor).")
    p.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B",
                   help="HF tokenizer id (or local path).")
    p.add_argument("--no-think-template", default=str(DEFAULT_TEMPLATE),
                   help="Path to the no-think .jinja template (jackrong only; "
                        "used for the no-think view).")
    p.add_argument("--think-template", default="default",
                   help="Path to the think .jinja template, or 'default' "
                        "to use the tokenizer's bundled chat template. For "
                        "--format train2 this is the only template used (each "
                        "record emits a single view with this template).")
    p.add_argument("--mode", default="both", choices=("both", "only_no_think", "only_think"),
                   help="(jackrong only) Which views to emit per sample. 'both' "
                        "emits BOTH think and no_think — doubles corpus.")
    p.add_argument("--val-ratio", type=float, default=0.05,
                   help="Fraction of SOURCE samples reserved for held-out "
                        "evaluation (default 0.05 = 5%%). Split is by source "
                        "sample, not by sequence, so think + no_think views of "
                        "the same sample never leak across train/val.")
    p.add_argument("--max-seq-len", type=int, default=16384,
                   help="Pack to this fixed sequence length. For --format "
                        "train2, recommended 13000 (covers p99 ~11K tok; "
                        "attention compute ~63%% of seq 16384).")
    p.add_argument("--shuffle-seed", type=int, default=42,
                   help="Seed for sequence-order shuffle before packing.")
    p.add_argument("--max-samples", type=int, default=-1,
                   help="Cap on samples for smoke runs (-1 = all).")
    p.add_argument("--report", action="store_true",
                   help="Print stats only, do not save.")
    return p.parse_args()


def load_jsonl(path: str) -> Iterator[dict]:
    """Yield one record per JSONL line (format-agnostic)."""
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except json.JSONDecodeError as e:
                print(f"[warn] skipping malformed line: {e}", file=sys.stderr)


# Backwards-compat alias (Jackrong called explicitly elsewhere).
load_jackrong = load_jsonl


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _strip_think_blocks(text: str) -> str:
    """Remove all `<think>…</think>` blocks (and leading whitespace after)
    from an assistant value. If no block exists, return text unchanged."""
    if _THINK_OPEN not in text:
        return text
    out = []
    rest = text
    while True:
        i = rest.find(_THINK_OPEN)
        if i < 0:
            out.append(rest)
            break
        out.append(rest[:i])
        j = rest.find(_THINK_CLOSE, i)
        if j < 0:
            # Unclosed think block — drop everything after the opener (rare).
            break
        rest = rest[j + len(_THINK_CLOSE):]
        # Drop a single leading newline if present (template emits "</think>\n\n").
        rest = rest.lstrip("\n")
    return "".join(out).lstrip()


def to_messages(sample: dict, strip_think: bool) -> list[dict]:
    """Convert Jackrong's `conversation` to chat-format messages.

    - human → user, gpt → assistant.
    - If `strip_think` is True, remove every `<think>…</think>` block from
      every assistant value (used for the no_think training subset).
    """
    msgs: list[dict] = []
    for turn in sample.get("conversation", []):
        src = turn.get("from", "")
        if src == "human":
            role = "user"
        elif src == "gpt":
            role = "assistant"
        else:
            continue
        content = turn.get("value", "")
        if not isinstance(content, str):
            continue
        if role == "assistant" and strip_think:
            content = _strip_think_blocks(content)
            if not content:
                # Pure-think samples become empty after stripping → skip
                # rather than emitting a degenerate assistant turn.
                return []
        msgs.append({"role": role, "content": content})
    if not msgs or msgs[-1]["role"] != "assistant":
        return []
    return msgs


def to_messages_train2(sample: dict) -> list[dict]:
    """Validate + filter a train2_v1_cleaned record's `messages` list.

    The dataset is already in ChatML shape (system / user / assistant
    roles). We:
    - drop any record whose roles include anything outside the trio,
    - drop any record whose last turn isn't `assistant` (training signal
      lives in the last assistant turn),
    - keep `<think>` blocks untouched.

    NOTE: we do NOT enforce strict role alternation — the dataset has
    multi-turn records like (system, user, assistant, user, assistant, …)
    and the bundled Qwen3.5 chat template handles them correctly.
    """
    raw = sample.get("messages", [])
    if not isinstance(raw, list) or not raw:
        return []
    msgs: list[dict] = []
    for turn in raw:
        if not isinstance(turn, dict):
            return []
        role = turn.get("role")
        content = turn.get("content", "")
        if role not in ("system", "user", "assistant"):
            return []
        if not isinstance(content, str):
            return []
        msgs.append({"role": role, "content": content})
    if msgs[-1]["role"] != "assistant":
        return []
    return msgs


def _render_and_tokenize(tokenizer, chat_template: str | None,
                          msgs: list[dict], src_id) -> list[int] | None:
    """Apply chat template + tokenize a pre-built message list.

    Implementation note: we do **two steps** rather than passing
    `tokenize=True` to apply_chat_template, because transformers >=5.x
    returns a `tokenizers.Encoding` object from `tokenize=True` instead
    of a list of ints, which silently breaks downstream `len()` /
    packing logic. The two-step (render → tokenize) avoids that.
    """
    try:
        rendered = tokenizer.apply_chat_template(
            msgs,
            chat_template=chat_template,
            add_generation_prompt=False,
            tokenize=False,
        )
    except Exception as e:
        print(f"[warn] apply_chat_template failed for id={src_id}: {e}",
              file=sys.stderr)
        return None
    if not isinstance(rendered, str):
        try:
            rendered = "".join(rendered)
        except Exception:
            return None
    try:
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    except Exception as e:
        print(f"[warn] tokenize failed for id={src_id}: {e}",
              file=sys.stderr)
        return None
    return list(ids)


def encode_sample(tokenizer, chat_template: str | None, sample: dict,
                  strip_think: bool) -> list[int] | None:
    """Apply the chosen chat template + tokenize one Jackrong sample.

    `strip_think=True` removes `<think>…</think>` blocks from the assistant
    content BEFORE templating, so the no_think template emits an empty
    `<think>\\n\\n</think>` block followed by the final answer only.

    `chat_template=None` falls back to the tokenizer's bundled template
    (used for the think-mode subset to match GPQA-Diamond eval).
    """
    msgs = to_messages(sample, strip_think=strip_think)
    if not msgs:
        return None
    return _render_and_tokenize(tokenizer, chat_template, msgs,
                                 sample.get("id", "?"))


def encode_sample_train2(tokenizer, chat_template: str | None,
                          sample: dict) -> list[int] | None:
    """Tokenize one train2_v1_cleaned record (single view, no think-strip)."""
    msgs = to_messages_train2(sample)
    if not msgs:
        return None
    return _render_and_tokenize(tokenizer, chat_template, msgs,
                                 sample.get("id", "?"))


def pack(ids_iter: Iterable[list[int]], seq_len: int, sep_id: int,
         pad_id: int) -> torch.Tensor:
    """Concatenate sequences with a separator token, split into fixed seq_len.

    The trailing partial chunk is padded with pad_id (so the dataloader can
    treat all rows uniformly; the loss masks the padded positions later if
    needed).
    """
    buf: list[int] = []
    chunks: list[list[int]] = []
    for ids in ids_iter:
        buf.extend(ids)
        buf.append(sep_id)
        while len(buf) >= seq_len:
            chunks.append(buf[:seq_len])
            buf = buf[seq_len:]
    if buf:
        # Pad the final chunk to seq_len.
        buf.extend([pad_id] * (seq_len - len(buf)))
        chunks.append(buf)
    if not chunks:
        return torch.empty(0, seq_len, dtype=torch.long)
    return torch.tensor(chunks, dtype=torch.long)


def main() -> int:
    args = parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[fatal] input not found: {in_path}", file=sys.stderr)
        return 2

    # Resolve both templates (file path or 'default' or inline string).
    def _resolve_template(spec: str) -> str | None:
        if spec == "default":
            return None
        p = Path(spec)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        return spec

    no_think_tpl = _resolve_template(args.no_think_template) if args.mode != "only_think" else None
    think_tpl = _resolve_template(args.think_template) if args.mode != "only_no_think" else None
    print(f"[info] mode={args.mode}")
    print(f"[info] no_think_template: {args.no_think_template} "
          f"({'inline/file' if no_think_tpl else 'tokenizer default'})")
    print(f"[info] think_template:    {args.think_template} "
          f"({'inline/file' if think_tpl else 'tokenizer default'})")

    # Load tokenizer.
    print(f"[info] loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Resolve separator + pad token. Qwen3.5 tokenizer has <|endoftext|> and
    # <|im_end|>; we use <|endoftext|> (a.k.a. eos) as separator. If pad_token
    # is not set, fall back to eos.
    sep_id = tokenizer.eos_token_id
    if sep_id is None:
        sep_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else sep_id
    print(f"[info] sep_id={sep_id}  pad_id={pad_id}  vocab_size={tokenizer.vocab_size}")

    # Load + filter samples.
    print(f"[info] loading samples from: {in_path}")
    t0 = time.time()
    samples = list(load_jackrong(str(in_path)))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    print(f"[info] loaded {len(samples)} samples in {time.time()-t0:.2f}s")

    # Split source samples into train/val BEFORE tokenization so that the
    # think and no_think views of the same source sample stay on the same
    # side of the split (avoids val leak through the duplicated view).
    rng_split = random.Random(args.shuffle_seed)
    sample_order = list(range(len(samples)))
    rng_split.shuffle(sample_order)
    n_val = max(0, int(len(samples) * args.val_ratio))
    val_sample_idx = set(sample_order[:n_val])
    print(f"[info] held-out val split: {n_val} source samples "
          f"({100 * args.val_ratio:.1f}% of {len(samples)})")

    def _tokenize_subset(subset: list[dict], label: str) -> dict:
        print(f"[info] tokenizing [{label}] subset of {len(subset)} samples…")
        t = time.time()
        ids_list: list[list[int]] = []
        modes_list: list[str] = []
        lens_by: dict[str, list[int]] = {"think": [], "no_think": [], "train2": []}
        skip_t = skip_nt = skip_tr2 = 0
        for s in subset:
            if args.format == "train2":
                ids = encode_sample_train2(tokenizer, think_tpl, s)
                if ids is None:
                    skip_tr2 += 1
                else:
                    ids_list.append(ids); modes_list.append("train2")
                    lens_by["train2"].append(len(ids))
                continue
            # jackrong dual-view path
            if args.mode in ("both", "only_think"):
                ids = encode_sample(tokenizer, think_tpl, s, strip_think=False)
                if ids is None:
                    skip_t += 1
                else:
                    ids_list.append(ids); modes_list.append("think")
                    lens_by["think"].append(len(ids))
            if args.mode in ("both", "only_no_think"):
                ids = encode_sample(tokenizer, no_think_tpl, s, strip_think=True)
                if ids is None:
                    skip_nt += 1
                else:
                    ids_list.append(ids); modes_list.append("no_think")
                    lens_by["no_think"].append(len(ids))
        if args.format == "train2":
            print(f"[info]   [{label}] tokenized {len(ids_list)} sequences "
                  f"(skipped={skip_tr2}) in {time.time()-t:.2f}s")
        else:
            print(f"[info]   [{label}] tokenized {len(ids_list)} sequences "
                  f"(think={len(lens_by['think'])} skipped={skip_t} | "
                  f"no_think={len(lens_by['no_think'])} skipped={skip_nt}) "
                  f"in {time.time()-t:.2f}s")
        for mn, ls in lens_by.items():
            if not ls:
                continue
            ls_s = sorted(ls); n = len(ls)
            print(f"[info]   [{label}] {mn:>9s} token-len: "
                  f"min={ls_s[0]} p50={ls_s[n//2]} p95={ls_s[int(n*0.95)]} max={ls_s[-1]} "
                  f"total={sum(ls)}")
        return {"ids": ids_list, "modes": modes_list, "lens_by": lens_by,
                "skipped": {"think": skip_t, "no_think": skip_nt, "train2": skip_tr2}}

    train_samples = [s for i, s in enumerate(samples) if i not in val_sample_idx]
    val_samples = [s for i, s in enumerate(samples) if i in val_sample_idx]

    train_tok = _tokenize_subset(train_samples, "train")
    val_tok = _tokenize_subset(val_samples, "val") if val_samples else None

    if args.report:
        for label, blob in (("train", train_tok), ("val", val_tok)):
            if blob is None:
                continue
            tl = sum(blob["lens_by"]["think"])
            ntl = sum(blob["lens_by"]["no_think"])
            tr2 = sum(blob["lens_by"]["train2"])
            tot = tl + ntl + tr2 + len(blob["ids"])  # +1 sep per seq
            chunks = (tot + args.max_seq_len - 1) // args.max_seq_len
            if args.format == "train2":
                print(f"[report] {label}: {len(blob['ids'])} seqs "
                      f"({tr2} train2 tokens) → "
                      f"{chunks} chunks of {args.max_seq_len}")
            else:
                print(f"[report] {label}: {len(blob['ids'])} seqs "
                      f"({tl} think + {ntl} no_think tokens) → "
                      f"{chunks} chunks of {args.max_seq_len}")
        return 0

    def _shuffle_and_pack(blob: dict, label: str) -> tuple[torch.Tensor, dict]:
        rng_local = random.Random(args.shuffle_seed)
        order = list(range(len(blob["ids"])))
        rng_local.shuffle(order)
        ids = [blob["ids"][i] for i in order]
        modes = [blob["modes"][i] for i in order]
        print(f"[info] packing [{label}] {len(ids)} seqs at seq_len={args.max_seq_len}…")
        t = time.time()
        packed = pack(ids, args.max_seq_len, sep_id, pad_id)
        print(f"[info] [{label}] packed.shape = {tuple(packed.shape)} in {time.time()-t:.2f}s")
        meta_local = {
            "split": label,
            "format": args.format,
            "input": str(in_path),
            "tokenizer": args.tokenizer,
            "no_think_template_path": args.no_think_template,
            "think_template_path": args.think_template,
            "mode": args.mode,
            "max_seq_len": args.max_seq_len,
            "val_ratio": args.val_ratio,
            "n_samples_in_jsonl": len(samples),
            "max_samples_cap": args.max_samples,
            "n_samples_in_split": len(train_samples) if label == "train" else len(val_samples),
            "n_sequences_total": len(ids),
            "n_sequences_think": len(blob["lens_by"]["think"]),
            "n_sequences_no_think": len(blob["lens_by"]["no_think"]),
            "n_sequences_train2": len(blob["lens_by"]["train2"]),
            "n_skipped_think": blob["skipped"]["think"],
            "n_skipped_no_think": blob["skipped"]["no_think"],
            "n_skipped_train2": blob["skipped"]["train2"],
            "n_chunks": int(packed.shape[0]),
            "total_tokens": int(packed.numel()),
            "sep_id": int(sep_id),
            "pad_id": int(pad_id),
        }
        return packed, meta_local

    train_packed, train_meta = _shuffle_and_pack(train_tok, "train")
    if train_packed.numel() == 0:
        print("[fatal] no train chunks produced — check input and template.", file=sys.stderr)
        return 2

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": train_packed, "meta": train_meta}, out_path)
    print(f"[ok] saved -> {out_path}  ({train_packed.shape[0]} chunks × "
          f"{train_packed.shape[1]} tokens = {train_packed.numel()} tokens)")

    if val_tok is not None and val_tok["ids"]:
        val_packed, val_meta = _shuffle_and_pack(val_tok, "val")
        if val_packed.numel() > 0:
            val_path = out_path.with_name(out_path.stem + "_val" + out_path.suffix)
            torch.save({"input_ids": val_packed, "meta": val_meta}, val_path)
            print(f"[ok] saved -> {val_path}  ({val_packed.shape[0]} chunks × "
                  f"{val_packed.shape[1]} tokens = {val_packed.numel()} tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
