#!/usr/bin/env python3
"""
Run QUALITY eval only against a LOCAL Docker container.
Usage:
  docker run -d --gpus all -p 8080:8080 --name test my-submission:latest
  # wait ~3 min
  HF_HOME=/path/to/hf_cache QUALITY_LIMIT=0.1 python3 run_quality_local.py
"""
import os, sys, json, time, re, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

CONTAINER_URL = os.environ.get("CONTAINER_URL", "http://localhost:8080")
QUALITY_LIMIT = float(os.environ.get("QUALITY_LIMIT", "0.1"))
NUM_CONCURRENT = int(os.environ.get("NUM_CONCURRENT", "8"))
# Output path is configurable so concurrent runs (e.g. GPU-2 checkpoint
# sweep + GPU-7 final sweep) don't clobber each other's /tmp file. This
# is I/O plumbing only — measurement logic is unchanged.
RESULTS_PATH = os.environ.get("QUALITY_RESULTS_PATH", "/tmp/quality_results.json")

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

QUALITY_TASKS = [
    ("mmlu_pro",                  "mmlu_pro",     5, "exact_match,custom-extract", 0.621, False),
    ("ifeval",                    "ifeval",       0, "inst_level_strict_acc,none",  0.814, False),
    ("gpqa_diamond_cot_zeroshot", "gpqa_diamond", 0, "exact_match,flexible-extract",0.630, True),
]

# Optional task subset (config plumbing only — measurement logic unchanged).
# Comma-separated result_keys or task_names, e.g. QUALITY_ONLY_TASKS=gpqa_diamond
# runs ONLY GPQA-Diamond (used by the per-checkpoint GPQA repeat sweep).
_ONLY = os.environ.get("QUALITY_ONLY_TASKS", "").strip()
if _ONLY:
    _keep = {t.strip() for t in _ONLY.split(",") if t.strip()}
    QUALITY_TASKS = [t for t in QUALITY_TASKS if t[0] in _keep or t[1] in _keep]
    if not QUALITY_TASKS:
        raise SystemExit(f"QUALITY_ONLY_TASKS={_ONLY!r} matched no task")


def _http(payload, timeout=600):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{CONTAINER_URL}/invocations", data=body,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def _invoke(prompt, max_tokens):
    r = _http({"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0})
    text = r.get("choices", [{}])[0].get("text", "")
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()


def _invoke_chat(prompt, max_tokens, thinking=False):
    payload = {"messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0}
    if thinking:
        payload["thinking"] = True
    else:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    r = _http(payload, timeout=600)
    return r.get("choices", [{}])[0].get("message", {}).get("content", "")


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
                        return idx, _invoke_chat(context, max_tokens, thinking=True)
                    elif context.rstrip().endswith("step by step.") or "Answer: Let" in context[-30:]:
                        return idx, _invoke(context, max_tokens)
                    else:
                        return idx, _invoke_chat(context, max_tokens, thinking=False)
                except Exception as e:
                    if attempt < 2: time.sleep(5*(attempt+1))
            return idx, ""

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=NUM_CONCURRENT) as executor:
            futures = {executor.submit(_do, i): i for i in range(total)}
            for future in as_completed(futures):
                idx, text = future.result()
                out[idx] = text
                completed += 1
                if completed == 1:
                    print(f"  sample output: [{text[:120]}]", flush=True)
                if completed % max(1, total//10) == 0 or completed == total:
                    elapsed = time.perf_counter() - t0
                    rate = completed/elapsed if elapsed > 0 else 0
                    print(f"  {completed}/{total} | {rate:.1f} req/s | ETA {(total-completed)/rate if rate>0 else 0:.0f}s", flush=True)
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


if __name__ == "__main__":
    from lm_eval import simple_evaluate

    print(f"Checking {CONTAINER_URL}/ping ...", flush=True)
    urllib.request.urlopen(f"{CONTAINER_URL}/ping", timeout=5)
    print("✅ Container ready", flush=True)

    results = {}
    limit = QUALITY_LIMIT if QUALITY_LIMIT < 1.0 else None
    if limit: print(f"[DEV MODE] QUALITY_LIMIT={limit} ({int(limit*100)}% of questions)", flush=True)

    for task_name, result_key, num_fewshot, metric_key, threshold, thinking in QUALITY_TASKS:
        print(f"\n[{result_key}] {task_name} ({num_fewshot}-shot, thinking={thinking})...", flush=True)

        if task_name == "mmlu_pro":
            from http.server import HTTPServer, BaseHTTPRequestHandler
            from socketserver import ThreadingMixIn
            PROXY_PORT = int(os.environ.get("MMLU_PROXY_PORT", "18080"))

            class _S(ThreadingMixIn, HTTPServer):
                daemon_threads = True

            class _H(BaseHTTPRequestHandler):
                protocol_version = "HTTP/1.1"
                def log_message(self, *a): pass
                def do_POST(self):
                    try:
                        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                        req = urllib.request.Request(f"{CONTAINER_URL}/invocations",
                            data=body, headers={"Content-Type": "application/json"})
                        result = urllib.request.urlopen(req, timeout=120).read()
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

            threading.Thread(target=_S(("127.0.0.1", PROXY_PORT), _H).serve_forever, daemon=True).start()
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
            eval_out = simple_evaluate(
                model=LocalLM(thinking=thinking),
                tasks=[task_name], num_fewshot=num_fewshot, batch_size=1,
                limit=limit, random_seed=0, numpy_random_seed=1234,
                torch_random_seed=1234, confirm_run_unsafe_code=True,
            )

        task_results = eval_out.get("results", {})
        if task_name == "mmlu_pro":
            scores = [v.get(metric_key) for k, v in task_results.items()
                      if k.startswith("mmlu_pro_") and isinstance(v, dict) and v.get(metric_key) is not None]
            score = round(sum(scores)/len(scores), 4) if scores else None
        else:
            single = task_results.get(task_name, {})
            score = single.get(metric_key)
            if score is None:
                for k, v in single.items():
                    if metric_key.split(",")[0] in k and isinstance(v, (int, float)):
                        score = v; break
        score = round(float(score), 4) if score is not None else None
        passed = score is not None and score >= threshold
        results[result_key] = {"score": score, "threshold": threshold, "passed": passed}
        print(f"  [{result_key}] {'PASS ✅' if passed else 'FAIL ❌'} — score={score}, threshold={threshold}", flush=True)

    results["all_passed"] = all(r.get("passed", False) for r in results.values() if isinstance(r, dict) and "passed" in r)
    print(f"\nQuality gates: {'ALL PASSED ✅' if results['all_passed'] else 'SOME FAILED ❌'}")
    print(json.dumps(results, indent=2))

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {RESULTS_PATH}")
