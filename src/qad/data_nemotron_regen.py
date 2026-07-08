#!/usr/bin/env python3
"""data_nemotron_regen.py — JSONL pack for qwen3_5_nemotron_combined_regen.

Standalone pre-pack for the nemotron-regen reasoning corpus. Imports the
tokenize / pack helpers from ``src/qad/data.py`` without modifying it.

Input  : ``data/qwen3_5_nemotron_combined_regen.jsonl``
         shape ``{"id": str, "conversations": [{"role": str, "content": str}, ...]}``
         The top-level key is ``conversations`` (plural with -s), distinct
         from the train2 format's ``messages``. Role strings are identical
         (system / user / assistant) so no role mapping is needed.
Output : ``packed.pt`` (train) and ``packed_val.pt`` (val) — each a dict
         ``{"input_ids": torch.LongTensor[N, seq_len], "meta": {...}}``
         compatible with ``train_cyankiwi_v2.py``'s loader. Plus
         ``pack_meta.json`` summarising row counts, sha256 of input, etc.

Pack policy mirrors data.py: concatenate per-sample id streams with an
``<|endoftext|>`` separator and split into fixed ``--max-seq-len``
chunks. Final partial chunk pad-filled. No filtering beyond the minimal
schema check (all msgs must be valid dicts with allowed roles + string
content; record must end with an assistant turn).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Reuse shared helpers without modifying data.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_jsonl, _render_and_tokenize, pack  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = str(_REPO_ROOT / "data" / "qwen3_5_nemotron_combined_regen.jsonl")
DEFAULT_OUTPUT = _REPO_ROOT / "runs" / "qad_nemotron_regen" / "train_data" / "packed.pt"


def to_messages_nemotron_regen(sample: dict) -> list[dict]:
    """Validate + return the assistant-terminated message list.

    No filtering beyond schema: top-level ``conversations`` must be a
    non-empty list; every turn must be a dict with ``role`` in
    {system, user, assistant} and string ``content``; final turn must be
    assistant. User opted out of any content-based filtering (no
    ``</think>`` filter, no language filter).
    """
    raw = sample.get("conversations", [])
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


def encode_sample_nemotron_regen(tokenizer, chat_template: str | None,
                                 sample: dict) -> list[int] | None:
    msgs = to_messages_nemotron_regen(sample)
    if not msgs:
        return None
    return _render_and_tokenize(tokenizer, chat_template, msgs,
                                sample.get("id", "?"))


def _sha256_of_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help=f"Input JSONL (default {DEFAULT_INPUT})")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help=f"Output train packed.pt (default {DEFAULT_OUTPUT}); "
                        "val saved alongside as packed_val.pt")
    p.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B",
                   help="HF tokenizer id (default Qwen/Qwen3.5-4B)")
    p.add_argument("--think-template", default="default",
                   help="Path to .jinja chat template, or 'default' to use "
                        "the tokenizer's bundled template (recommended).")
    p.add_argument("--max-seq-len", type=int, default=16384,
                   help="Pack chunk length (default 16384).")
    p.add_argument("--val-ratio", type=float, default=0.05,
                   help="Fraction of source samples held out for eval "
                        "(default 0.05). Split is sample-level pre-tokenize.")
    p.add_argument("--shuffle-seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=-1,
                   help="Cap on source samples for smoke runs (-1 = all).")
    p.add_argument("--report", action="store_true",
                   help="Dry-run: tokenize + report chunk counts, do not save.")
    return p.parse_args()


def _resolve_template(spec: str) -> str | None:
    if spec == "default":
        return None
    p = Path(spec)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return spec


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[fatal] input not found: {in_path}", file=sys.stderr)
        return 2

    chat_template = _resolve_template(args.think_template)
    print(f"[info] think_template: {args.think_template} "
          f"({'inline/file' if chat_template else 'tokenizer default'})")

    print(f"[info] loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    sep_id = tokenizer.eos_token_id
    if sep_id is None:
        sep_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else sep_id
    print(f"[info] sep_id={sep_id}  pad_id={pad_id}  vocab_size={tokenizer.vocab_size}")

    print(f"[info] loading samples from: {in_path}")
    t0 = time.time()
    samples = list(load_jsonl(str(in_path)))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    n_source = len(samples)
    print(f"[info] loaded {n_source} source samples in {time.time()-t0:.2f}s")

    rng_split = random.Random(args.shuffle_seed)
    order = list(range(n_source))
    rng_split.shuffle(order)
    n_val = max(0, int(n_source * args.val_ratio))
    val_idx = set(order[:n_val])
    train_samples = [s for i, s in enumerate(samples) if i not in val_idx]
    val_samples = [s for i, s in enumerate(samples) if i in val_idx]
    print(f"[info] split: train={len(train_samples)}  val={len(val_samples)}")

    def _tokenize_subset(subset: list[dict], label: str) -> dict:
        print(f"[info] tokenizing [{label}] {len(subset)} samples…")
        t = time.time()
        ids_list: list[list[int]] = []
        lens: list[int] = []
        skipped = 0
        for s in subset:
            ids = encode_sample_nemotron_regen(tokenizer, chat_template, s)
            if ids is None:
                skipped += 1
                continue
            ids_list.append(ids)
            lens.append(len(ids))
        elapsed = time.time() - t
        if lens:
            sl = sorted(lens)
            print(f"[info]   [{label}] tokenized {len(ids_list)} seqs "
                  f"(skipped={skipped}) in {elapsed:.1f}s "
                  f"({len(ids_list)/elapsed:.1f} seqs/s)")
            print(f"[info]   [{label}] tok-len min={sl[0]} p50={sl[len(sl)//2]} "
                  f"p95={sl[int(len(sl)*0.95)]} p99={sl[int(len(sl)*0.99)]} "
                  f"max={sl[-1]} total={sum(lens)}")
        else:
            print(f"[info]   [{label}] tokenized 0 seqs (all skipped)")
        return {"ids": ids_list, "lens": lens, "skipped": skipped}

    train_tok = _tokenize_subset(train_samples, "train")
    val_tok = _tokenize_subset(val_samples, "val") if val_samples else None

    if args.report:
        for label, blob in (("train", train_tok), ("val", val_tok)):
            if blob is None or not blob["ids"]:
                continue
            tot = sum(blob["lens"]) + len(blob["ids"])  # +1 sep per seq
            n_chunks = (tot + args.max_seq_len - 1) // args.max_seq_len
            print(f"[report] {label}: {len(blob['ids'])} seqs → "
                  f"{n_chunks} chunks of {args.max_seq_len} "
                  f"({tot} tokens incl. sep)")
        # Steps/epoch hint
        if train_tok["ids"]:
            tot = sum(train_tok["lens"]) + len(train_tok["ids"])
            chunks = (tot + args.max_seq_len - 1) // args.max_seq_len
            for batch in (16, 8):
                steps = (chunks + batch - 1) // batch
                print(f"[report] effective_batch={batch}: ~{steps} steps/epoch")
        return 0

    print(f"[info] computing sha256 of input ({in_path.stat().st_size/1e9:.2f} GB)…")
    t = time.time()
    input_sha = _sha256_of_file(in_path)
    print(f"[info] sha256 = {input_sha}  ({time.time()-t:.1f}s)")

    def _shuffle_and_pack(blob: dict, label: str) -> tuple[torch.Tensor, dict]:
        rng_local = random.Random(args.shuffle_seed)
        idx = list(range(len(blob["ids"])))
        rng_local.shuffle(idx)
        ids = [blob["ids"][i] for i in idx]
        print(f"[info] packing [{label}] {len(ids)} seqs at seq_len={args.max_seq_len}…")
        t = time.time()
        packed = pack(ids, args.max_seq_len, sep_id, pad_id)
        print(f"[info] [{label}] packed.shape={tuple(packed.shape)} in {time.time()-t:.1f}s")
        meta_local = {
            "split": label,
            "format": "nemotron_regen",
            "input": str(in_path),
            "input_sha256": input_sha,
            "tokenizer": args.tokenizer,
            "chat_template": args.think_template,
            "max_seq_len": args.max_seq_len,
            "val_ratio": args.val_ratio,
            "shuffle_seed": args.shuffle_seed,
            "n_source_rows": n_source,
            "n_samples_in_split": len(blob["ids"]) + blob["skipped"],
            "n_skipped": blob["skipped"],
            "n_kept": len(blob["ids"]),
            "n_chunks": int(packed.shape[0]) if packed.numel() else 0,
            "total_tokens_packed": int(packed.numel()),
            "sep_id": int(sep_id),
            "pad_id": int(pad_id),
        }
        return packed, meta_local

    train_packed, train_meta = _shuffle_and_pack(train_tok, "train")
    if train_packed.numel() == 0:
        print("[fatal] no train chunks produced — check input / template.", file=sys.stderr)
        return 2

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": train_packed, "meta": train_meta}, out_path)
    print(f"[ok] saved -> {out_path}  ({train_packed.shape[0]} × "
          f"{train_packed.shape[1]} = {train_packed.numel()} tokens)")

    val_meta = None
    if val_tok is not None and val_tok["ids"]:
        val_packed, val_meta = _shuffle_and_pack(val_tok, "val")
        if val_packed.numel() > 0:
            val_path = out_path.with_name(out_path.stem + "_val" + out_path.suffix)
            torch.save({"input_ids": val_packed, "meta": val_meta}, val_path)
            print(f"[ok] saved -> {val_path}  ({val_packed.shape[0]} × "
                  f"{val_packed.shape[1]} = {val_packed.numel()} tokens)")

    pack_meta_path = out_path.parent / "pack_meta.json"
    summary = {
        "train": train_meta,
        "val": val_meta,
        "wall_seconds": int(time.time() - t0),
    }
    with open(pack_meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[ok] pack_meta.json -> {pack_meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
