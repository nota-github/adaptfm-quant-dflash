#!/usr/bin/env python3
"""
Run eval harness against a LOCAL Docker container instead of SageMaker.
Usage:
  1. Start container: docker run -d --gpus all -p 8080:8080 --name test my-submission:latest
  2. Wait ~3 min for vLLM to load: curl http://localhost:8080/ping
  3. Run: EVAL_MODE=full python3 run_eval_local.py

Results saved to /tmp/local_eval_results.json
"""
import os, sys, json, time, statistics, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict
import urllib.request

# ─── Config ──────────────────────────────────────────────────────────────────
CONTAINER_URL = os.environ.get("CONTAINER_URL", "http://localhost:8080")
QUALITY_LIMIT = float(os.environ.get("QUALITY_LIMIT", "0.1"))
NUM_CONCURRENT = int(os.environ.get("NUM_CONCURRENT", "8"))
EVAL_MODE = os.environ.get("EVAL_MODE", "quality")  # latency | quality | full

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

WARMUP_RUNS = 5
NUM_RUNS = int(os.environ.get("NUM_RUNS", "50"))
FILLER = "The quick brown fox jumps over the lazy dog. "
# Optional realistic prompts: JSON {"short":[...], "medium":[...], "long":[...]} of
# raw text pre-sliced to each category's token budget (see prepare_longbench_latency.py).
# When set, each category is measured over NUM_RUNS *distinct* slices (median), instead
# of repeating the synthetic FILLER. Unset → FILLER (backward-compatible default).
LATENCY_PROMPTS_FILE = os.environ.get("LATENCY_PROMPTS_FILE")
PROMPT_CONFIGS = {
    "short":  {"num_tokens": 64,   "max_new_tokens": 128},
    "medium": {"num_tokens": 2048, "max_new_tokens": 256},
    "long":   {"num_tokens": 8192, "max_new_tokens": 256},
}
BASELINE_LATENCY = {"short": 2582, "medium": 5441, "long": 6576, "average": 4866}
QUALITY_TASKS = [
    ("mmlu_pro",                  "mmlu_pro",     5, "exact_match,custom-extract", 0.621, False),
    ("ifeval",                    "ifeval",       0, "inst_level_strict_acc,none",  0.814, False),
    ("gpqa_diamond_cot_zeroshot", "gpqa_diamond", 0, "exact_match,flexible-extract",0.630, True),
]


def _http_invoke(payload: dict, timeout: int = 600) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{CONTAINER_URL}/invocations",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())


def _invoke(prompt, max_tokens, temperature=0.0):
    t0 = time.perf_counter()
    result = _http_invoke({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature})
    latency_ms = (time.perf_counter() - t0) * 1000
    choices = result.get("choices", [])
    text = choices[0].get("text", "") if choices else ""
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    return text, latency_ms


def _invoke_chat(prompt, max_tokens, temperature=0.0, thinking=False):
    payload = {"messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": temperature}
    if thinking:
        payload["thinking"] = True
    else:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.perf_counter()
    result = _http_invoke(payload, timeout=600)
    latency_ms = (time.perf_counter() - t0) * 1000
    choices = result.get("choices", [])
    text = choices[0].get("message", {}).get("content", "") if choices else ""
    return text, latency_ms


# ─── Wait for container ───────────────────────────────────────────────────────
def wait_for_container():
    print(f"Waiting for container at {CONTAINER_URL}/ping ...", flush=True)
    for i in range(120):
        try:
            urllib.request.urlopen(f"{CONTAINER_URL}/ping", timeout=2)
            print("✅ Container ready", flush=True)
            return
        except Exception:
            if i % 12 == 0:
                print(f"  [{i*5}s] still waiting...", flush=True)
            time.sleep(5)
    raise RuntimeError("Container not ready after 10 min")


# ─── Latency eval ─────────────────────────────────────────────────────────────
def _latency_prompts(cat, cfg):
    """(warmup_prompts, measure_prompts) for a category.

    With LATENCY_PROMPTS_FILE: NUM_RUNS distinct realistic slices measured once
    each + WARMUP_RUNS separate slices (cycled if the pool is short). Without it:
    the synthetic FILLER repeated (original behavior)."""
    if not LATENCY_PROMPTS_FILE:
        p = FILLER * max(1, cfg["num_tokens"] // 10)
        return [p] * WARMUP_RUNS, [p] * NUM_RUNS
    with open(LATENCY_PROMPTS_FILE) as f:
        pool = json.load(f)[cat]
    if not pool:
        raise RuntimeError(f"{LATENCY_PROMPTS_FILE} has no prompts for '{cat}'")
    measure = [pool[i % len(pool)] for i in range(NUM_RUNS)]
    warm = [pool[(NUM_RUNS + i) % len(pool)] for i in range(WARMUP_RUNS)]
    if len(pool) < NUM_RUNS:
        print(f"  [{cat}] WARN: only {len(pool)} slices for {NUM_RUNS} runs — cycling", flush=True)
    return warm, measure


def run_latency_eval():
    print("\n" + "="*70, flush=True)
    src = LATENCY_PROMPTS_FILE if LATENCY_PROMPTS_FILE else "synthetic FILLER"
    print(f"LATENCY EVALUATION (prompts: {src})", flush=True)
    results: Dict[str, Any] = {}
    for cat, cfg in PROMPT_CONFIGS.items():
        warm, measure = _latency_prompts(cat, cfg)
        for p in warm:
            try: _invoke(p, cfg["max_new_tokens"])
            except: pass
        print(f"  [{cat}] warmup done", flush=True)
        latencies = []
        for i, p in enumerate(measure):
            try:
                _, ms = _invoke(p, cfg["max_new_tokens"])
                latencies.append(ms)
                if (i+1) % 10 == 0: print(f"  [{cat}] {i+1}/{NUM_RUNS} — {ms:.1f}ms", flush=True)
            except Exception as e:
                print(f"  [{cat}] run {i+1} FAILED: {e}", flush=True)
        if latencies:
            median = statistics.median(latencies)
            results[cat] = {"median_ms": round(median, 2), "num_runs": len(latencies)}
            bl = BASELINE_LATENCY.get(cat, 0)
            print(f"  [{cat}] Median: {median:.2f}ms (baseline: {bl}ms, speedup: {bl/median:.2f}x)", flush=True)
    medians = [r["median_ms"] for r in results.values() if "median_ms" in r]
    results["overall_avg_median_ms"] = round(statistics.mean(medians), 2) if medians else None
    results["hardware"] = "Local GPU"
    return results


# ─── Quality eval ─────────────────────────────────────────────────────────────
from lm_eval.api.model import LM as _LM

class LocalLM(_LM):
    def __init__(self, thinking=False):
        super().__init__()
        self.thinking = thinking

    def generate_until(self, requests):
        total = len(requests)
        print(f"  [generate_until] {total} requests (thinking={self.thinking}, concurrency={NUM_CONCURRENT})", flush=True)
        out = [None] * total
        completed = 0

        def _do(idx):
            context, gen_kwargs = requests[idx].args
            default_max = 12288 if self.thinking else 512
            max_tokens = min(gen_kwargs.get("max_gen_toks", gen_kwargs.get("max_new_tokens", default_max)), default_max)
            for attempt in range(3):
                try:
                    if self.thinking:
                        text, _ = _invoke_chat(context, max_tokens, thinking=True)
                    elif context.rstrip().endswith("step by step.") or "Answer: Let" in context[-30:]:
                        text, _ = _invoke(context, max_tokens)
                    else:
                        text, _ = _invoke_chat(context, max_tokens, thinking=False)
                    return idx, text
                except Exception as e:
                    if attempt < 2: time.sleep(5*(attempt+1))
            return idx, ""

        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=NUM_CONCURRENT) as executor:
            futures = {executor.submit(_do, i): i for i in range(total)}
            for future in as_completed(futures):
                idx, text = future.result()
                out[idx] = text
                completed += 1
                if completed == 1:
                    print(f"  sample output: [{text[:120]}]", flush=True)
                if completed % max(1, total//10) == 0 or completed == total:
                    elapsed = time.perf_counter() - t_start
                    rate = completed/elapsed if elapsed > 0 else 0
                    eta = (total-completed)/rate if rate > 0 else 0
                    print(f"  {completed}/{total} | {rate:.1f} req/s | ETA {eta:.0f}s", flush=True)
        return out

    def loglikelihood(self, requests): return [(0.0, False)] * len(requests)
    def loglikelihood_rolling(self, requests): return [(0.0,)] * len(requests)
    @property
    def eot_token_id(self): return 0
    @property
    def max_length(self): return 16384
    @property
    def max_gen_toks(self): return 512
    @property
    def batch_size(self): return 1
    @property
    def device(self): return "cpu"
    def tok_encode(self, s): return list(s.encode())
    def tok_decode(self, t): return bytes(t).decode(errors="replace")
    def set_cache_hook(self, cache_hook): pass


def run_quality_eval():
    from lm_eval import simple_evaluate

    print("\n" + "="*70, flush=True)
    print("QUALITY EVALUATION", flush=True)
    results: Dict[str, Any] = {}

    for task_name, result_key, num_fewshot, metric_key, threshold, thinking in QUALITY_TASKS:
        print(f"\n[{result_key}] {task_name} ({num_fewshot}-shot, thinking={thinking})...", flush=True)
        limit = QUALITY_LIMIT if QUALITY_LIMIT < 1.0 else None
        if limit: print(f"  [DEV MODE] QUALITY_LIMIT={limit}", flush=True)

        if task_name == "mmlu_pro":
            from http.server import HTTPServer, BaseHTTPRequestHandler
            from socketserver import ThreadingMixIn
            PROXY_PORT = 18080

            class _Server(ThreadingMixIn, HTTPServer):
                daemon_threads = True

            class _Handler(BaseHTTPRequestHandler):
                protocol_version = "HTTP/1.1"
                def log_message(self, *a): pass
                def do_POST(self):
                    try:
                        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                        req = urllib.request.Request(f"{CONTAINER_URL}/invocations",
                            data=body, headers={"Content-Type": "application/json"})
                        resp = urllib.request.urlopen(req, timeout=120)
                        result = resp.read()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(result)))
                        self.send_header("Connection", "keep-alive")
                        self.end_headers()
                        self.wfile.write(result)
                        self.wfile.flush()
                    except BrokenPipeError: pass
                    except Exception as e:
                        try:
                            err = json.dumps({"error": str(e)}).encode()
                            self.send_response(500)
                            self.send_header("Content-Length", str(len(err)))
                            self.end_headers()
                            self.wfile.write(err)
                        except: pass

            proxy = _Server(("127.0.0.1", PROXY_PORT), _Handler)
            threading.Thread(target=proxy.serve_forever, daemon=True).start()
            print(f"  [proxy] localhost:{PROXY_PORT} → {CONTAINER_URL}", flush=True)

            eval_out = simple_evaluate(
                model="local-chat-completions",
                model_args=(f"model=Qwen/Qwen3.5-4B,"
                            f"base_url=http://localhost:{PROXY_PORT}/v1/chat/completions,"
                            f"tokenized_requests=False,num_concurrent=8,"
                            f"eos_string=<|im_end|>,timeout=120"),
                tasks=[task_name], num_fewshot=num_fewshot, batch_size=1,
                limit=limit, apply_chat_template=True,
                random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
                confirm_run_unsafe_code=True,
            )
        else:
            model = LocalLM(thinking=thinking)
            eval_out = simple_evaluate(
                model=model, tasks=[task_name], num_fewshot=num_fewshot,
                batch_size=1, limit=limit,
                random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
                confirm_run_unsafe_code=True,
            )

        task_results = eval_out.get("results", {})
        if task_name == "mmlu_pro":
            subtask_scores = [v.get(metric_key) for k, v in task_results.items()
                              if k.startswith("mmlu_pro_") and isinstance(v, dict)
                              and v.get(metric_key) is not None]
            score = round(sum(subtask_scores)/len(subtask_scores), 4) if subtask_scores else None
        else:
            single = task_results.get(task_name, {})
            score = single.get(metric_key)
            if score is None:
                base = metric_key.split(",")[0]
                for k, v in single.items():
                    if base in k and isinstance(v, (int, float)):
                        score = v; break
        score = round(float(score), 4) if score is not None else None
        passed = score is not None and score >= threshold
        results[result_key] = {"score": score, "threshold": threshold, "passed": passed}
        print(f"  [{result_key}] {'PASS ✅' if passed else 'FAIL ❌'} — score={score}", flush=True)

    results["all_passed"] = all(r.get("passed", False) for r in results.values() if isinstance(r, dict) and "passed" in r)
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    wait_for_container()
    all_results = {}

    if EVAL_MODE in ("latency", "full"):
        all_results["latency"] = run_latency_eval()
        print(json.dumps(all_results["latency"], indent=2))

    if EVAL_MODE in ("quality", "full"):
        all_results["quality"] = run_quality_eval()
        print(json.dumps(all_results["quality"], indent=2))

    with open("/tmp/local_eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n✅ Results saved to /tmp/local_eval_results.json")
