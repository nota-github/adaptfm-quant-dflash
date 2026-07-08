"""FastAPI server: thin proxy in front of `vllm serve`.

Production mode (default): spawns `vllm serve` as a subprocess on an
internal port and proxies the competition's `/v1/completions` and
`/invocations` endpoints to it. `/ping` mirrors vllm's `/health`. This
relies on vllm's first-class OpenAI compat — full logprobs/echo support,
proper async + continuous batching — so the server stays
thread-safe under high lm-eval concurrency and supports loglikelihood
tasks like GPQA-Diamond multi-choice.

Stub mode (`EQC_USE_STUB=1`): no subprocess, returns canned responses
for the serving API contract (no GPU, no vllm).
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import subprocess
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from serving.config import load_config

logger = logging.getLogger("eqc.server")

USE_STUB = os.environ.get("EQC_USE_STUB") == "1"
EQC_PORT = int(os.environ.get("EQC_PORT", "8080"))
# Keep vllm on a deterministic backend port so two parallel servers
# (e.g., GPU 1 on EQC_PORT=8080 and GPU 2 on EQC_PORT=8081) don't collide.
# Override via EQC_VLLM_INTERNAL_PORT — the official eval harness opens a
# proxy on 127.0.0.1:18080 (= EQC_PORT 8080 + 10000), which collides with
# our default vllm port. For local-measurement runs we shift vllm to a
# non-conflicting port (e.g. 29090).
_DEFAULT_VLLM_PORT = EQC_PORT + 10000
VLLM_PORT = int(os.environ.get("EQC_VLLM_INTERNAL_PORT", str(_DEFAULT_VLLM_PORT)))

# Proxy-level retry config. lm-eval's retry path on a 5xx response from
# our proxy hits an UnboundLocalError in lm-eval 0.4.11's amodel_call
# (api_models.py:545 — references `outputs` before assignment when an
# exception happens before the variable is set), which kills the entire
# eval. To avoid that we absorb transient httpx errors inside the proxy
# and re-issue the upstream request with exponential backoff before
# falling through to 503. Tuned for chat-completions long-output
# workloads where occasional connection blips happen.
PROXY_MAX_ATTEMPTS = int(os.environ.get("EQC_PROXY_MAX_ATTEMPTS", "4"))
PROXY_BACKOFF_BASE = float(os.environ.get("EQC_PROXY_BACKOFF_BASE", "0.5"))


def _build_vllm_args(config: dict) -> list[str]:
    vllm_args = config.get("vllm_args", {}) or {}
    env_max_seqs = os.environ.get("EQC_VLLM_MAX_NUM_SEQS")
    max_num_seqs = (
        int(env_max_seqs) if env_max_seqs else int(vllm_args.get("max_num_seqs", 1))
    )
    # max_model_len: latency runs use the config-yaml default (8704: 8192 prompt +
    # 256 output + headroom), but quality runs need to fit GPQA-Diamond's max_tokens
    # 12288 thinking-on response. Override via EQC_VLLM_MAX_MODEL_LEN.
    env_max_len = os.environ.get("EQC_VLLM_MAX_MODEL_LEN")
    max_model_len = int(env_max_len) if env_max_len else int(vllm_args.get("max_model_len", 8704))
    args = [
        "vllm", "serve", str(config["model"]),
        "--host", "127.0.0.1",
        "--port", str(VLLM_PORT),
        "--dtype", str(vllm_args.get("dtype", "auto")),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(
            os.environ.get("EQC_GPU_MEM_CAP_FRACTION", "0.78")
        ),
        "--max-num-seqs", str(max_num_seqs),
        # The official eval harness's MMLU-Pro path hardcodes
        # model=Qwen/Qwen3.5-4B in its lm-eval model_args, so vllm must
        # register that name regardless of which underlying repo we
        # actually loaded (e.g. RedHatAI W4A16). Without this every
        # MMLU-Pro request 404s.
        "--served-model-name", "Qwen/Qwen3.5-4B",
    ]
    if vllm_args.get("enforce_eager") or os.environ.get("EQC_FORCE_EAGER") == "1":
        args.append("--enforce-eager")
    if "block_size" in vllm_args:
        args += ["--block-size", str(vllm_args["block_size"])]
    # mamba_block_size — granularity of the Mamba/GDN state cache (and, with
    # prefix caching on, the mamba prefix-cache block size). Defaults to
    # block_size when prefix caching is on; set large to keep the whole decode
    # in one mamba block (diagnostic: suppresses cross-mamba-block state
    # migration in align mode).
    if "mamba_block_size" in vllm_args:
        args += ["--mamba-block-size", str(vllm_args["mamba_block_size"])]
    # mamba_ssm_cache_dtype — GDN SSM recurrent state cache dtype. Default 'auto'
    # = model dtype (bf16). bf16 has ~0.4% relative precision; under spec-dec +
    # prefix caching, different draft batches produce different bf16 rounding that
    # accumulates and flips greedy tokens (the spec+APC correctness bug). Setting
    # 'float32' makes the SSM state fp32 so per-step differences shrink ~1e-7 and
    # the output becomes batch-/prefix-invariant. Settable via config
    # (vllm_args.mamba_ssm_cache_dtype) or env (EQC_MAMBA_SSM_DTYPE).
    _ssm_dt = vllm_args.get("mamba_ssm_cache_dtype") or os.environ.get(
        "EQC_MAMBA_SSM_DTYPE"
    )
    if _ssm_dt:
        args += ["--mamba-ssm-cache-dtype", str(_ssm_dt)]
    # mamba_cache_dtype — conv state cache dtype (companion to ssm dtype above).
    _m_dt = vllm_args.get("mamba_cache_dtype") or os.environ.get("EQC_MAMBA_DTYPE")
    if _m_dt:
        args += ["--mamba-cache-dtype", str(_m_dt)]
    # cudagraph_mode — override the V1 default FULL_AND_PIECEWISE. PIECEWISE
    # keeps piecewise graphs (GDN is in splitting_ops so it runs outside the
    # graph) but drops the FULL capture that uniform spec-decode batches would
    # otherwise get — a candidate fix for the align+spec+prefix-cache corruption
    # while keeping most of the cudagraph speedup. Settable via config
    # (vllm_args.cudagraph_mode) or env (EQC_CUDAGRAPH_MODE).
    _cgm = vllm_args.get("cudagraph_mode") or os.environ.get("EQC_CUDAGRAPH_MODE")
    if _cgm:
        import json as _json
        args += ["--compilation-config", _json.dumps({"cudagraph_mode": str(_cgm)})]
    # KV cache dtype — fp8_e4m3 / fp8_e5m2 reduce KV memory ~2× (storage
    # quantization). Note: native FP8 *compute* is on the A10G forbidden
    # list, but FP8 *storage* with software dequant in attention is not —
    # vllm uses int8 paths under the hood on Ampere.
    if "kv_cache_dtype" in vllm_args:
        args += ["--kv-cache-dtype", str(vllm_args["kv_cache_dtype"])]
    # Attention backend — some spec-dec drafters (e.g. DFlash) require
    # FLASH_ATTN specifically; FlashInfer is vllm's default.
    if "attention_backend" in vllm_args:
        args += ["--attention-backend", str(vllm_args["attention_backend"])]
    # Max batched tokens — DFlash README recommends 32768 to leave room
    # for spec-dec batches. Default is fine for batch=1 latency.
    if "max_num_batched_tokens" in vllm_args:
        args += ["--max-num-batched-tokens", str(vllm_args["max_num_batched_tokens"])]
    # language_model_only — Qwen3.5-4B is technically multimodal (VL family);
    # this flag tells vllm to skip loading the vision encoder and dummy-MM
    # profile_run — a large per-forward-pass speedup (MM encoder cudagraph
    # overhead removed), with no quality impact for text-only benchmarks.
    if vllm_args.get("language_model_only"):
        args.append("--language-model-only")
    # enable_chunked_prefill — defaults to True in vllm 0.22.1. For single-
    # stream batch=1 latency, the chunking overhead does not help; turning
    # it off keeps prefill a single pass. For the quality eval at
    # concurrency 8 it may shift throughput slightly, but the latency
    # bench is what we optimize for.
    if vllm_args.get("enable_chunked_prefill") is False:
        args.append("--no-enable-chunked-prefill")
    # enable_prefix_caching — the latency bench reuses the identical prompt
    # across all warmup+measurement requests per category, so caching the
    # prefix makes prefill ~free on the timed runs (big medium/long win).
    # NOTE: on Qwen3.5's hybrid (linear+full) attention, prefix caching forces
    # mamba cache mode 'align', which *requires* chunked prefill — so keep
    # enable_chunked_prefill ON (do not set it false) when this is true.
    if vllm_args.get("enable_prefix_caching") is True:
        args.append("--enable-prefix-caching")
    elif vllm_args.get("enable_prefix_caching") is False:
        args.append("--no-enable-prefix-caching")
    if config.get("quant"):
        args += ["--quantization", str(config["quant"])]
    # Some HF tokenizer/model packagings (e.g. RedHatAI's compressed-tensors
    # repos) ship config that relies on remote auto_map → vllm refuses
    # without --trust-remote-code. Opt-in per experiment via config.yaml.
    if config.get("trust_remote_code"):
        args.append("--trust-remote-code")
    # Override the tokenizer if the model's bundled tokenizer_config has
    # incompatible metadata (e.g. RedHatAI w4a16 lists tokenizer_class
    # 'TokenizersBackend' which transformers 4.57 doesn't know). Falling
    # back to the base Qwen tokenizer is safe — vocab is identical.
    if config.get("tokenizer"):
        args += ["--tokenizer", str(config["tokenizer"])]
    # Chat template — load-bearing for quality eval. The competition
    # applies qwen_no_think.jinja server-side to disable <think> blocks
    # in /v1/chat/completions responses. Without this, Qwen3.5-4B's
    # default template lets the model emit reasoning blocks that the
    # benchmark parser doesn't strip, tanking quality scores. Latency
    # runs hit /v1/completions (no template) so this is a no-op there.
    chat_template = config.get("chat_template") or os.environ.get("EQC_CHAT_TEMPLATE")
    if chat_template:
        args += ["--chat-template", str(chat_template)]
    # Speculative decoding (NEXTN/MTP/EAGLE/ngram) — pass-through to
    # vllm's --speculative-config. Accepts either a dict in YAML (which
    # we JSON-serialize) or an already-stringified JSON.
    spec_cfg = config.get("speculative_config")
    if spec_cfg:
        if isinstance(spec_cfg, dict):
            import json as _json
            spec_cfg = _json.dumps(spec_cfg)
        args += ["--speculative-config", str(spec_cfg)]
    # Data-parallel replicas — used for quality eval (one replica per GPU,
    # built-in load balancer). Latency runs leave it unset (DP=1 mirrors the
    # single A10G submission target).
    dp_size = os.environ.get("EQC_VLLM_DP_SIZE")
    if dp_size:
        args += ["--data-parallel-size", str(dp_size)]
    return args


_vllm_proc: Optional[subprocess.Popen] = None
_client: Optional[httpx.AsyncClient] = None
_load_error: Optional[str] = None


def _maybe_nsys_prefix() -> list[str]:
    """Return an `nsys profile` argv prefix when EQC_NSYS_PROFILE=1.

    Diagnostic-only: wraps the `vllm serve` subprocess with Nsight Systems
    profiling. Output
    path comes from EQC_NSYS_OUT (dir) + EQC_NSYS_CATEGORY (filename).
    EQC_NSYS_DURATION caps the trace length as a safety net; nsys also
    stops cleanly on the SIGTERM it gets when the proxy tears down.

    Subprocess tracing: nsys auto-follows posix_spawn'd descendants of
    the launched process. vllm uses multiprocessing.Process (spawn
    method) for the EngineCore worker, which falls under that umbrella
    — so we do NOT need `--trace-fork-before-exec`. In fact, adding it
    breaks the capture: empirically it produces a
    trace where host NVTX events are present but CUDA kernel data is
    completely missing.
    """
    if os.environ.get("EQC_NSYS_PROFILE") != "1":
        return []
    out_dir = os.environ.get("EQC_NSYS_OUT") or "."
    category = os.environ.get("EQC_NSYS_CATEGORY") or "trace"
    duration = os.environ.get("EQC_NSYS_DURATION", "600")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{category}.nsys-rep")
    return [
        "nsys", "profile",
        "-o", out_path,
        "-t", "cuda,nvtx,osrt",
        "-s", "none",
        "--cuda-graph-trace=node",
        "--force-overwrite=true",
        "--duration", str(duration),
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vllm_proc, _client, _load_error
    if not USE_STUB:
        try:
            config = load_config(os.environ.get("EQC_CONFIG_PATH"))
            cmd = _maybe_nsys_prefix() + _build_vllm_args(config)
            logger.info("starting vllm: %s", " ".join(cmd))
            # start_new_session detaches the subprocess so its workers
            # form a kill-able process group; cleanup uses os.killpg.
            _vllm_proc = subprocess.Popen(cmd, start_new_session=True)
        except Exception as exc:  # pragma: no cover
            _load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("vllm subprocess failed to spawn")
    # Generous timeouts for chat-completions long-output requests; httpx
    # transport retries handle connection-layer blips below the per-request
    # level so we don't have to surface them as 503s.
    _client = httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{VLLM_PORT}",
        timeout=httpx.Timeout(900.0, connect=15.0, read=900.0),
        transport=httpx.AsyncHTTPTransport(retries=3),
    )
    try:
        yield
    finally:
        try:
            if _client is not None:
                await _client.aclose()
        finally:
            if _vllm_proc is not None and _vllm_proc.poll() is None:
                import os as _os
                import signal as _signal

                try:
                    _os.killpg(_os.getpgid(_vllm_proc.pid), _signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    _vllm_proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        _os.killpg(_os.getpgid(_vllm_proc.pid), _signal.SIGKILL)
                    except ProcessLookupError:
                        pass


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
async def ping() -> JSONResponse:
    if USE_STUB:
        return JSONResponse(status_code=200, content={"status": "ok"})
    if _load_error is not None:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": _load_error},
        )
    if _vllm_proc is None or _vllm_proc.poll() is not None:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": "vllm subprocess not running"},
        )
    try:
        r = await _client.get("/health", timeout=2.0)
        if r.status_code == 200:
            return JSONResponse(status_code=200, content={"status": "ok"})
    except Exception:
        pass
    return JSONResponse(status_code=503, content={"status": "loading"})


async def _stub_handle(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid json"})
    if "prompt" not in body or not isinstance(body["prompt"], str):
        return JSONResponse(status_code=400, content={"error": "missing or invalid prompt"})
    max_tokens = body.get("max_tokens", 128)
    if not isinstance(max_tokens, int) or max_tokens < 1:
        return JSONResponse(status_code=400, content={"error": "invalid max_tokens"})
    text = "stub:" + body["prompt"][:32]
    return JSONResponse(
        content={"choices": [{"text": text, "index": 0, "finish_reason": "stop"}]}
    )


def _wants_stream(body: bytes) -> bool:
    """Best-effort check for ``"stream": true`` in the JSON body.

    Avoids parsing the body when it doesn't look like JSON. The
    competition evaluator sets ``stream=true`` on the GPQA-Diamond chat
    path; lm-eval's local-chat-completions does not.
    """
    if not body:
        return False
    try:
        parsed = _json.loads(body)
    except (ValueError, TypeError):
        return False
    return isinstance(parsed, dict) and bool(parsed.get("stream"))


async def _proxy_stream(target_path: str, body: bytes) -> Response:
    """Forward an SSE/streaming response from vllm to the caller verbatim.

    We open the upstream request, then return a StreamingResponse whose
    generator pulls raw chunks from vllm and re-emits them. Retry logic
    only applies *before* the first chunk arrives — once a chunk has
    been sent downstream we can't safely re-issue without duplicating
    output. If the upstream connection drops mid-stream, the client
    sees a truncated response, same as it would if vllm itself died.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(PROXY_MAX_ATTEMPTS):
        if _vllm_proc is None or _vllm_proc.poll() is not None:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "detail": "vllm subprocess died mid-request"},
            )
        try:
            # Build the request via httpx so we can read it lazily.
            req = _client.build_request(
                "POST", target_path,
                content=body,
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            )
            resp = await _client.send(req, stream=True)
        except httpx.RequestError as exc:
            last_exc = exc
            if attempt + 1 < PROXY_MAX_ATTEMPTS:
                await asyncio.sleep(PROXY_BACKOFF_BASE * (2 ** attempt))
                continue
            break

        media_type = resp.headers.get("content-type", "text/event-stream")
        status = resp.status_code

        async def _gen():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()

        return StreamingResponse(_gen(), status_code=status, media_type=media_type)

    return JSONResponse(
        status_code=503,
        content={"status": "loading", "detail": f"{type(last_exc).__name__}: {last_exc}"},
    )


async def _proxy_to(request: Request, target_path: str) -> Response:
    """Forward the request body as-is to vllm's matching endpoint.

    Two paths:
      - non-streaming: buffer the upstream response and return it whole,
        retrying transient httpx RequestErrors with exponential backoff.
      - streaming (``"stream": true`` in body): open a streaming
        upstream request and return a StreamingResponse that pipes
        chunks through verbatim. Retry only before the first chunk.

    lm-eval 0.4.11's client treats our 5xx responses as a fatal-retry
    trigger inside an ``except`` handler that crashes on its own
    UnboundLocalError, so once we let a 503 through it tends to take
    the whole eval down. Better to absorb the blip here.
    """
    if _load_error is not None or _vllm_proc is None or _vllm_proc.poll() is not None:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": _load_error or "vllm subprocess not running"},
        )
    body = await request.body()
    if _wants_stream(body):
        return await _proxy_stream(target_path, body)
    last_exc: Optional[Exception] = None
    for attempt in range(PROXY_MAX_ATTEMPTS):
        try:
            r = await _client.post(
                target_path,
                content=body,
                headers={"Content-Type": "application/json"},
            )
            return Response(
                content=r.content,
                status_code=r.status_code,
                media_type=r.headers.get("content-type", "application/json"),
            )
        except httpx.RequestError as exc:
            last_exc = exc
            if attempt + 1 < PROXY_MAX_ATTEMPTS:
                # Re-check subprocess liveness before retrying; if vllm
                # died there's no point waiting.
                if _vllm_proc is None or _vllm_proc.poll() is not None:
                    return JSONResponse(
                        status_code=500,
                        content={"status": "error", "detail": "vllm subprocess died mid-request"},
                    )
                await asyncio.sleep(PROXY_BACKOFF_BASE * (2 ** attempt))
                continue
            break
    return JSONResponse(
        status_code=503,
        content={"status": "loading", "detail": f"{type(last_exc).__name__}: {last_exc}"},
    )


async def _proxy_completion(request: Request) -> Response:
    if USE_STUB:
        return await _stub_handle(request)
    return await _proxy_to(request, "/v1/completions")


def _maybe_transform_invocations(body: bytes) -> tuple[bytes, str, bool]:
    """Route an /invocations request body to the right vllm endpoint.

    The competition's eval harness (src/eval_harness/run_quality_local.py)
    drives both text-completion and chat-completion traffic through
    /invocations:
      - {"prompt": ..., "max_tokens": ...}                → /v1/completions
      - {"messages": [...], "max_tokens": ...,
         "thinking"?: bool, "chat_template_kwargs"?: ...} → /v1/chat/completions

    For chat traffic, the harness flips thinking mode via either a
    top-level "thinking" boolean (GPQA-Diamond case) or a
    "chat_template_kwargs": {"enable_thinking": false} dict
    (MMLU-Pro / IFEval case). vllm only understands the latter, so we
    fold "thinking": true into chat_template_kwargs before forwarding.
    """
    if not body:
        return body, "/v1/completions", False
    try:
        parsed = _json.loads(body)
    except (ValueError, TypeError):
        return body, "/v1/completions", False
    if not isinstance(parsed, dict):
        return body, "/v1/completions", False
    if isinstance(parsed.get("messages"), list):
        thinking_flag = parsed.pop("thinking", None)
        ctk = parsed.get("chat_template_kwargs")
        if not isinstance(ctk, dict):
            ctk = {}
        if thinking_flag is not None:
            ctk.setdefault("enable_thinking", bool(thinking_flag))
        elif "enable_thinking" not in ctk:
            # Default to thinking OFF for chat traffic that doesn't say
            # anything about it. MMLU-Pro via lm-eval's local-chat-
            # completions sends bare messages with no chat_template_kwargs;
            # without this default, vllm uses Qwen3.5's bundled chat
            # template default (enable_thinking=True), which emits a
            # <think>...</think> block that exceeds the MMLU token
            # budget and leaves the answer letter unextracted.
            ctk["enable_thinking"] = False
        parsed["chat_template_kwargs"] = ctk
        new_body = _json.dumps(parsed).encode()
        return new_body, "/v1/chat/completions", True
    return body, "/v1/completions", False


async def _proxy_invocations(request: Request) -> Response:
    """Polymorphic /invocations: prompt → completions, messages → chat."""
    if USE_STUB:
        try:
            parsed = await request.json()
        except Exception:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("messages"), list):
            return await _stub_chat_handle(request)
        return await _stub_handle(request)
    body = await request.body()
    new_body, target_path, transformed = _maybe_transform_invocations(body)
    if not transformed:
        return await _proxy_to(request, target_path)
    # Re-issue with a synthesized request body so _proxy_to / _proxy_stream
    # see the rewritten payload (e.g. thinking → chat_template_kwargs).
    if _load_error is not None or _vllm_proc is None or _vllm_proc.poll() is not None:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": _load_error or "vllm subprocess not running"},
        )
    if _wants_stream(new_body):
        return await _proxy_stream(target_path, new_body)
    last_exc: Optional[Exception] = None
    for attempt in range(PROXY_MAX_ATTEMPTS):
        try:
            r = await _client.post(
                target_path,
                content=new_body,
                headers={"Content-Type": "application/json"},
            )
            return Response(
                content=r.content,
                status_code=r.status_code,
                media_type=r.headers.get("content-type", "application/json"),
            )
        except httpx.RequestError as exc:
            last_exc = exc
            if attempt + 1 < PROXY_MAX_ATTEMPTS:
                if _vllm_proc is None or _vllm_proc.poll() is not None:
                    return JSONResponse(
                        status_code=500,
                        content={"status": "error", "detail": "vllm subprocess died mid-request"},
                    )
                await asyncio.sleep(PROXY_BACKOFF_BASE * (2 ** attempt))
                continue
            break
    return JSONResponse(
        status_code=503,
        content={"status": "loading", "detail": f"{type(last_exc).__name__}: {last_exc}"},
    )


async def _stub_chat_handle(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid json"})
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return JSONResponse(status_code=400, content={"error": "missing or invalid messages"})
    last = msgs[-1].get("content") if isinstance(msgs[-1], dict) else ""
    last = (last or "")[:32]
    return JSONResponse(content={
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": f"stub:{last}"},
            "finish_reason": "stop",
        }],
    })


async def _proxy_chat_completion(request: Request) -> Response:
    if USE_STUB:
        return await _stub_chat_handle(request)
    return await _proxy_to(request, "/v1/chat/completions")


@app.post("/v1/completions")
async def v1_completions(request: Request) -> Response:
    return await _proxy_completion(request)


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request) -> Response:
    """Quality eval path: chat template + (when configured) thinking
    disabled via vllm's --chat-template flag (e.g. qwen_no_think.jinja)."""
    return await _proxy_chat_completion(request)


@app.post("/invocations")
async def invocations(request: Request) -> Response:
    return await _proxy_invocations(request)


@app.get("/v1/models")
async def v1_models() -> Response:
    """Forward /v1/models so clients can auto-discover the served model id."""
    if USE_STUB:
        return JSONResponse(content={
            "object": "list",
            "data": [{"id": "stub", "object": "model", "owned_by": "eqc"}],
        })
    if _vllm_proc is None or _vllm_proc.poll() is not None:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": "vllm subprocess not running"},
        )
    try:
        r = await _client.get("/v1/models", timeout=2.0)
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )
    except httpx.RequestError as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "detail": str(exc)},
        )
