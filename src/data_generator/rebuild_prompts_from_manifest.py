#!/usr/bin/env python3
"""Rebuild the original 220K prompt set from the shipped uuid manifest.

Why this exists: the 220K distillation corpus is a derived artifact of the gated
``nvidia/Nemotron-Post-Training-Dataset-v2`` and must not be redistributed. What the
repo ships instead is ``manifests/nemotron_regen_220k_uuids.txt`` — the ``uuid`` of
every source row of the 220K corpus (219,602 unique prompts). This script downloads
the dataset with YOUR credentials (HF_TOKEN; accept the terms on the dataset page
first), filters it to exactly those uuids, and emits the same ``prompts.jsonl``
schema as sample_nemotron.py — ready for generate_responses.py.

Unlike sample_nemotron.py (non-multilingual splits only), the original 220K drew from
the multilingual splits too, so this scans ALL data shards (~42 GiB download once).

Output schema per line (identical contract to sample_nemotron.py):
  {"id": <uuid>, "category": ..., "conversations": [{system},{user}], "reasoning": ...}

Usage (from src/data_generator/):
  HF_TOKEN=... ../../.venv/bin/python rebuild_prompts_from_manifest.py \
      --output prompts_220k.jsonl
  # then: generate_responses.py --input prompts_220k.jsonl --output <regen out> ...
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample_nemotron import REPO, to_prompt  # noqa: E402

DEFAULT_MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "manifests", "nemotron_regen_220k_uuids.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="uuid list, one per line (default: the shipped 220K manifest)")
    ap.add_argument("--output", default="prompts_220k.jsonl")
    ap.add_argument("--cache-dir", default=None, help="HF cache dir (default: HF default)")
    ap.add_argument("--no-download", action="store_true",
                    help="assume all shards already cached; skip snapshot_download")
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle the output (seeded) so categories interleave for the generator")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed")
    args = ap.parse_args()

    with open(args.manifest) as f:
        wanted = {line.strip() for line in f if line.strip()}
    print(f"manifest: {len(wanted):,} uuids from {args.manifest}", file=sys.stderr)

    from huggingface_hub import snapshot_download

    if not args.no_download:
        print("downloading ALL data shards incl. multilingual (~42GiB once, resumable; "
              "gated -> needs HF_TOKEN)...", file=sys.stderr)
    local = snapshot_download(
        REPO, repo_type="dataset", cache_dir=args.cache_dir,
        allow_patterns=["data/*"], local_files_only=args.no_download,
    )
    shards = sorted(glob.glob(os.path.join(local, "data", "*.parquet")))
    print(f"local snapshot: {local} ({len(shards)} shards)", file=sys.stderr)

    import pyarrow.parquet as pq

    out, seen = [], set()
    for p in shards:
        t = pq.read_table(p, columns=["uuid", "category", "reasoning", "messages"])
        for r in t.to_pylist():
            u = r["uuid"]
            if u not in wanted or u in seen:
                continue
            conv = to_prompt(r["messages"])
            if conv is None:
                continue
            seen.add(u)
            out.append({"id": u, "category": r.get("category"),
                        "conversations": conv, "reasoning": r.get("reasoning")})
        print(f"  {os.path.basename(p)}: cum {len(out):,}/{len(wanted):,}", file=sys.stderr)

    if args.shuffle:
        import random
        random.seed(args.seed)
        random.shuffle(out)

    with open(args.output, "w") as o:
        for obj in out:
            o.write(json.dumps(obj, ensure_ascii=False) + "\n")

    missing = len(wanted) - len(out)
    print(f"\nwrote {len(out):,} -> {args.output}", file=sys.stderr)
    if missing:
        print(f"WARNING: {missing:,} manifest uuids were NOT found in the dataset — "
              f"the upstream dataset may have changed. Expected 0.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
