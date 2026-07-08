"""Generate assistant responses for sampled prompts using a running vLLM server.

Reads the JSONL produced by the prompt sampler (rows with ``id``, ``category`` and
``conversations: [system, user]``), sends each conversation to an OpenAI-compatible
vLLM endpoint (``vllm serve``), and writes a "regen" JSONL where the generated
assistant turn is appended to ``conversations``.

The output is:
  * valid SpecForge / ShareGPT-style SFT data (system / user / assistant turns), and
  * directly reusable as the sampler's ``--exclude-ids-file`` (top-level ``id`` is
    preserved), so successive sampling rounds never re-draw prompts already generated.

Resumable: on restart, every ``id`` already present in ``--output`` is skipped, so a
crashed or interrupted run continues where it left off. Requests are fanned out with a
bounded worker pool; transient errors are retried with exponential backoff.

Example (start the server first):
  vllm serve Qwen/Qwen3-8B --port 8000 --tensor-parallel-size 1
  uv run python generate_responses.py \
      --model Qwen/Qwen3-8B \
      --input  sampled_prompts.jsonl \
      --output regen.jsonl \
      --concurrency 128 --max-tokens 2048 --thinking off
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time

import openai
from openai import AsyncOpenAI


# ----------------------------------------------------------------------------- helpers

def iter_jsonl(path: str):
    """Yield parsed rows from a JSONL file, tolerating blank / truncated lines."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def load_done_ids(path: str) -> set:
    """IDs already present in an existing output file (for resume)."""
    if not path or not os.path.exists(path):
        return set()
    done = set()
    for row in iter_jsonl(path):
        rid = row.get("id")
        if rid is not None:
            done.add(str(rid))
    return done


def seed_messages(row: dict):
    """Return the prompt messages (system/user, no assistant) to send to the model.

    Accepts either ``conversations`` (sampler output) or ``messages``. Any pre-existing
    assistant turn is dropped so re-generation starts from the prompt.
    """
    conv = row.get("conversations") or row.get("messages")
    if not isinstance(conv, list):
        return None
    msgs = []
    for m in conv:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or m.get("from")
        content = m.get("content")
        if content is None:
            content = m.get("value")
        if role in ("user", "human"):
            role = "user"
        elif role in ("assistant", "gpt"):
            continue  # drop existing answer
        elif role != "system":
            continue
        if isinstance(content, str) and content.strip():
            msgs.append({"role": role, "content": content})
    # Need at least one user turn to generate from.
    if not any(m["role"] == "user" for m in msgs):
        return None
    return msgs


def extract_reasoning(message) -> str | None:
    """Pull out the reasoning trace when a reasoning parser is enabled server-side.

    Field naming varies by vLLM version: newer builds (>=0.22) put it in ``reasoning``,
    older ones in ``reasoning_content``. Unknown fields also land in pydantic's
    ``model_extra``. Check all of them.
    """
    extra = getattr(message, "model_extra", None) or {}
    for key in ("reasoning_content", "reasoning"):
        r = getattr(message, key, None)
        if r is None and isinstance(extra, dict):
            r = extra.get(key)
        if isinstance(r, str) and r.strip():
            return r
    return None


def build_extra_body(args) -> dict:
    """vLLM-specific knobs that aren't first-class OpenAI params."""
    eb: dict = {}
    if args.thinking in ("on", "off"):
        eb["chat_template_kwargs"] = {"enable_thinking": args.thinking == "on"}
    if args.top_k is not None and args.top_k >= 0:
        eb["top_k"] = args.top_k
    if args.min_p is not None:
        eb["min_p"] = args.min_p
    if args.repetition_penalty is not None:
        eb["repetition_penalty"] = args.repetition_penalty
    return eb


# --------------------------------------------------------------------------- generation

class Stats:
    def __init__(self, total: int):
        self.total = total
        self.done = 0          # successfully generated this run
        self.failed = 0        # gave up after retries
        self.empty = 0         # model returned empty content
        self.truncated = 0     # hit max_tokens (finish_reason=="length"); dropped if --drop-truncated
        self.start = time.monotonic()


async def generate_one(client: AsyncOpenAI, args, extra_body: dict, messages: list):
    """One chat completion with retry/backoff. Returns the assistant message or None.

    Raises ``PermanentError`` for non-retryable failures (bad request / context length).
    """
    last_err = None
    for attempt in range(args.max_retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                presence_penalty=args.presence_penalty,
                frequency_penalty=args.frequency_penalty,
                seed=args.seed,
                stop=args.stop or None,
                extra_body=extra_body or None,
                timeout=args.timeout,
            )
            return resp.choices[0]
        except openai.BadRequestError as e:
            # 400s (e.g. prompt exceeds context length) won't succeed on retry.
            raise PermanentError(str(e)) from e
        except (openai.APITimeoutError, openai.APIConnectionError,
                openai.RateLimitError, openai.InternalServerError) as e:
            last_err = e
        except openai.APIStatusError as e:
            if e.status_code and 400 <= e.status_code < 500 and e.status_code != 429:
                raise PermanentError(str(e)) from e
            last_err = e
        except Exception as e:  # noqa: BLE001 - last-resort transient catch
            last_err = e
        # backoff with jitter before the next attempt
        if attempt < args.max_retries:
            delay = min(args.backoff * (2 ** attempt), args.max_backoff)
            await asyncio.sleep(delay + random.uniform(0, args.backoff))
    raise TransientError(str(last_err))


class TransientError(Exception):
    pass


class PermanentError(Exception):
    pass


def to_assistant_content(choice, args) -> str:
    """Reconstruct the verbatim assistant text the model produced.

    Behaviour depends on whether the server runs a reasoning parser and on the
    thinking mode:
      * no parser              -> ``content`` already holds the full text (any
                                  <think>...</think> is inline); use it as-is.
      * parser on, thinking on -> ``content`` is the answer and the trace is split
                                  into ``reasoning``; stitch them back together so
                                  the saved target matches the raw generation.
      * parser on, thinking off-> the model emits no <think>, so the parser leaves
                                  ``content`` empty and dumps the *answer* into
                                  ``reasoning``; use it directly, do NOT wrap it.
    """
    content = (choice.message.content or "").strip()
    reasoning = extract_reasoning(choice.message)
    if args.thinking == "off":
        return content or (reasoning or "")
    if reasoning and args.include_reasoning:
        return f"<think>\n{reasoning}\n</think>\n\n{content}".strip()
    # thinking on, no reasoning parser: the chat template emits the opening "<think>\n"
    # into the PROMPT, so the completion starts mid-trace (e.g. "...</think>\n\nanswer").
    # Re-attach the opening tag so the stored assistant message is self-contained.
    if (args.thinking == "on" and args.include_reasoning
            and content and not content.lstrip().startswith("<think>")):
        return "<think>\n" + content
    return content


async def worker(name, in_q, out_q, client, args, extra_body, stats):
    while True:
        try:
            row, msgs = in_q.get_nowait()
        except asyncio.QueueEmpty:
            return
        try:
            choice = await generate_one(client, args, extra_body, msgs)
            # Drop responses that hit the max_tokens cap (truncated, not naturally finished):
            # finish_reason=="length" means it exceeded --max-tokens, so the trace is cut mid-thought.
            if args.drop_truncated and choice.finish_reason == "length":
                stats.truncated += 1
                continue  # don't persist truncated; resume will re-attempt (temp>0 may finish next time)
            content = to_assistant_content(choice, args)
            if not content:
                stats.empty += 1
                continue  # don't persist empties; resume will retry them
            out_row = dict(row)
            out_row["conversations"] = msgs + [{"role": "assistant", "content": content}]
            out_row["gen_model"] = args.model
            out_row["finish_reason"] = choice.finish_reason
            await out_q.put(out_row)
        except PermanentError as e:
            stats.failed += 1
            print(f"  [skip:permanent] id={row.get('id')}: {e}", flush=True)
        except TransientError as e:
            stats.failed += 1
            print(f"  [skip:transient] id={row.get('id')} after "
                  f"{args.max_retries} retries: {e}", flush=True)


async def writer(out_q, out_f, stats, log_every):
    """Drain generated rows to disk (line-buffered) and report progress."""
    while True:
        item = await out_q.get()
        if item is None:
            return
        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        stats.done += 1
        if stats.done % log_every == 0:
            elapsed = time.monotonic() - stats.start
            rate = stats.done / elapsed if elapsed else 0.0
            remaining = stats.total - stats.done - stats.failed
            eta = remaining / rate if rate else float("inf")
            print(
                f"  {stats.done:,}/{stats.total:,} done "
                f"({stats.failed} failed, {stats.empty} empty, {stats.truncated} truncated) | "
                f"{rate:.1f}/s | ETA {eta/60:.1f} min",
                flush=True,
            )


async def run(args):
    # ----- load input + resume state
    print(f"reading prompts from {args.input} ...", flush=True)
    done = load_done_ids(args.output) if args.resume else set()
    if done:
        print(f"  resuming: {len(done):,} ids already in {args.output}", flush=True)

    pending = []
    n_total_in = n_skip_done = n_skip_noprompt = 0
    for row in iter_jsonl(args.input):
        n_total_in += 1
        rid = row.get("id")
        if rid is not None and str(rid) in done:
            n_skip_done += 1
            continue
        msgs = seed_messages(row)
        if msgs is None:
            n_skip_noprompt += 1
            continue
        pending.append((row, msgs))
        if args.limit and len(pending) >= args.limit:
            break

    print(f"  {n_total_in:,} input rows | {len(pending):,} to generate "
          f"| skipped {n_skip_done:,} done, {n_skip_noprompt:,} no-prompt", flush=True)

    if args.priority_categories:
        prio = {c: i for i, c in enumerate(args.priority_categories.split(","))}
        # stable sort: priority categories first (in given order), rest keep original order
        pending.sort(key=lambda rm: prio.get(rm[0].get("category"), len(prio)))
        n_prio = sum(1 for rm in pending if rm[0].get("category") in prio)
        print(f"  prioritizing {n_prio:,} rows from {list(prio)} ahead of the rest",
              flush=True)

    if not pending:
        print("nothing to do.", flush=True)
        return

    extra_body = build_extra_body(args)
    if args.dry_run:
        print("\n--- dry run: would send ---")
        print(f"model         : {args.model}")
        print(f"base_url      : {args.base_url}")
        print(f"concurrency   : {args.concurrency}")
        print(f"sampling      : temp={args.temperature} top_p={args.top_p} "
              f"max_tokens={args.max_tokens} thinking={args.thinking}")
        print(f"extra_body    : {extra_body}")
        print(f"first messages: {json.dumps(pending[0][1], ensure_ascii=False)[:600]}")
        return

    # ----- connect + verify server(s)
    # --base-url may be a comma-separated list of endpoints (e.g. 5 independent single-GPU
    # vllm servers). Workers round-robin across them, which sidesteps the DP-coordinator
    # shared-memory deadlock that --data-parallel-size hits under sustained load.
    urls = [u.strip() for u in args.base_url.split(",") if u.strip()]
    clients = [AsyncOpenAI(base_url=u, api_key=args.api_key, max_retries=0) for u in urls]
    print(f"  endpoints ({len(clients)}): {urls}", flush=True)
    for u, c in zip(urls, clients):
        try:
            served = [m.id for m in (await c.models.list()).data]
            tag = "OK" if args.model in served else f"WARN(model not in {served})"
            print(f"  {u} -> {tag}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: could not list models from {u}: {e}", flush=True)

    # ----- fan out: N workers drain the input queue, one writer persists results
    in_q: asyncio.Queue = asyncio.Queue()
    for item in pending:
        in_q.put_nowait(item)
    out_q: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    stats = Stats(total=len(pending))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mode = "a" if (args.resume and os.path.exists(args.output)) else "w"
    with open(args.output, mode, buffering=1) as out_f:
        writer_task = asyncio.create_task(writer(out_q, out_f, stats, args.log_every))
        workers = [
            asyncio.create_task(
                worker(i, in_q, out_q, clients[i % len(clients)], args, extra_body, stats))
            for i in range(args.concurrency)
        ]
        try:
            await asyncio.gather(*workers)
        finally:
            await out_q.put(None)   # signal writer to stop
            await writer_task
    for c in clients:
        await c.close()

    elapsed = time.monotonic() - stats.start
    print(f"\nDONE: wrote {stats.done:,} rows to {args.output} in {elapsed/60:.1f} min "
          f"({stats.failed} failed, {stats.empty} empty, {stats.truncated} truncated@max_tokens"
          f"{' [dropped]' if args.drop_truncated else ' [kept]'})", flush=True)


# --------------------------------------------------------------------------------- cli

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # I/O
    p.add_argument("--input", required=True,
                   help="JSONL of sampled prompts (sampler output)")
    p.add_argument("--output", required=True,
                   help="Destination 'regen' JSONL (also usable as sampler --exclude-ids-file)")
    p.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="Skip ids already present in --output (default: on)")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Overwrite --output instead of resuming")
    p.add_argument("--limit", type=int, default=0,
                   help="Generate at most this many (0 = all). Useful for smoke tests.")
    p.add_argument("--priority-categories", default=None,
                   help="Comma-separated categories to generate first, e.g. chat,stem,math,code")

    # server / model
    p.add_argument("--model", required=True, help="Model id as served by vLLM")
    p.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL",
                                                         "http://localhost:8000/v1"))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--concurrency", type=int, default=128,
                   help="Number of in-flight requests (worker pool size)")
    p.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout (s)")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--backoff", type=float, default=1.0, help="Base backoff (s)")
    p.add_argument("--max-backoff", type=float, default=30.0)

    # sampling
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20,
                   help="vLLM top_k (extra_body); set -1 to disable")
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--frequency-penalty", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--drop-truncated", dest="drop_truncated", action="store_true", default=True,
                   help="Do NOT save responses that hit --max-tokens (finish_reason=='length'); "
                        "they are cut mid-thought. On by default.")
    p.add_argument("--keep-truncated", dest="drop_truncated", action="store_false",
                   help="Persist truncated (length-capped) responses too (legacy behaviour).")
    p.add_argument("--stop", action="append", default=None,
                   help="Stop string (repeatable)")
    p.add_argument("--seed", type=int, default=None,
                   help="Fixed sampling seed for reproducibility (default: server random)")

    # reasoning / chat template
    p.add_argument("--thinking", choices=["auto", "on", "off"], default="auto",
                   help="Qwen3-style enable_thinking; 'auto' lets the chat template decide")
    p.add_argument("--include-reasoning", dest="include_reasoning",
                   action="store_true", default=True,
                   help="Merge server-split reasoning_content into the saved answer "
                        "as <think>...</think> (default: on)")
    p.add_argument("--no-include-reasoning", dest="include_reasoning",
                   action="store_false")

    # misc
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--dry-run", action="store_true",
                   help="Show the plan + first prompt without calling the server")
    args = p.parse_args()

    if args.concurrency < 1:
        print("--concurrency must be >= 1", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
