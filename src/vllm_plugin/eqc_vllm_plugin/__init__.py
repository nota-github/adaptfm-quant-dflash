"""vLLM general plugin: make a W4A16-quantized DFlash drafter loadable + runnable.

vLLM 0.21's `qwen3_dflash.py` has two issues that block a quantized DFlash
drafter (compressed-tensors W4A16):

  Bug 1 — `DFlashQwen3Model` builds its decoder layers WITHOUT forwarding
          `quant_config`, so every q/k/v/o_proj + MLP loads unquantized and
          the checkpoint's `weight_packed` keys have no destination
          (`KeyError: 'layers.0.mlp.down_proj.weight_packed'`).

  Bug 2 — `DFlashQwen3Model._build_fused_kv_buffers` reads `a.qkv_proj.weight`
          directly to fuse all layers' K/V projection into one GEMM. A
          quantized `QKVParallelLinear` exposes `weight_packed`/`weight_scale`
          instead of `.weight` (`AttributeError: 'QKVParallelLinear' object
          has no attribute 'weight'`).

This plugin patches both. It is gated by ``EQC_DFLASH_QUANT_PATCH=1`` so it
is a strict no-op for unquantized drafters and unrelated runs. Registered
under the ``vllm.general_plugins`` entry point, which vLLM loads in every
process — so the patch reaches the EngineCore worker where the drafter is
actually built (a plain monkey-patch in the parent would not survive the
multiprocessing spawn).

**Bug 1 fix**: wrap ``DFlashQwen3DecoderLayer.__init__`` to default
``quant_config`` from the draft model's quant config
(``get_draft_quant_config``) when the caller passed ``None``.

**Bug 2 fix**: wrap ``DFlashQwen3Model._build_fused_kv_buffers`` to first
attach a dequantized ``.weight`` to each quantized ``qkv_proj`` — recovered
via an identity-matrix forward (``qkv_proj(I) == Wᵀ``, robust to whatever
internal layout Marlin uses), numerically identical to the layer's own
quantized GEMM — then delegate to the original method unchanged. The fused
context-K/V projection therefore runs in BF16 at the exact same values the
quantized layer would produce; the per-step decode forward stays INT4.
"""

from __future__ import annotations

import os
import sys

_ENV = "EQC_DFLASH_QUANT_PATCH"


def _log(msg: str) -> None:
    print(f"[eqc_vllm_plugin] {msg}", file=sys.stderr, flush=True)


def _patch_decoder_layer(m) -> None:
    Layer = m.DFlashQwen3DecoderLayer
    orig = Layer.__init__
    if getattr(orig, "_eqc_patched", False):
        return

    def __init__(self, vllm_config, *, config, cache_config=None,
                 quant_config=None, prefix=""):
        if quant_config is None:
            try:
                from vllm.config import get_current_vllm_config
                from vllm.model_executor.models.utils import get_draft_quant_config
                quant_config = get_draft_quant_config(get_current_vllm_config())
            except Exception as exc:  # pragma: no cover
                _log(f"could not resolve draft quant_config: {type(exc).__name__}: {exc}")
        orig(self, vllm_config, config=config, cache_config=cache_config,
             quant_config=quant_config, prefix=prefix)

    __init__._eqc_patched = True
    Layer.__init__ = __init__
    _log("patched DFlashQwen3DecoderLayer.__init__ (Bug 1: quant_config forwarding)")


def _dequant_qkv_weight(qkv):
    """Recover the dequantized [out_features, hidden] weight of a (possibly
    quantized) linear via an identity-matrix forward: qkv(I) == Wᵀ."""
    import torch

    hidden = getattr(qkv, "input_size", None) or getattr(qkv, "input_size_per_partition")
    dt = None
    for attr in ("weight_scale", "scale"):
        t = getattr(qkv, attr, None)
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            dt = t.dtype
            break
    if dt is None:
        dt = torch.bfloat16
    dev = None
    for p in qkv.parameters():
        dev = p.device
        break
    eye = torch.eye(hidden, dtype=dt, device=dev)
    with torch.inference_mode():
        out = qkv(eye)
    if isinstance(out, tuple):
        out = out[0]
    return out.t().contiguous()  # [out_features, hidden]


def _patch_fused_kv(m) -> None:
    Model = m.DFlashQwen3Model
    orig = Model._build_fused_kv_buffers
    if getattr(orig, "_eqc_patched", False):
        return

    def _build(self):
        # `_build_fused_kv_buffers` is first called inside load_weights, BEFORE
        # the quant method's process_weights_after_loading (which sets the
        # Marlin buffers like g_idx_sort_indices). Running qkv(eye) then raises
        # AttributeError. precompute_and_store_context_kv has a lazy fallback
        # (`if not hasattr(self, "_num_attn_layers"): self._build_fused_kv_buffers()`)
        # that fires at inference time — after Marlin post-processing — so we
        # DEFER (return without setting _num_attn_layers) until the dequant
        # forward succeeds.
        for layer in self.layers:
            qkv = layer.self_attn.qkv_proj
            if hasattr(qkv, "weight"):
                continue  # unquantized, or already attached on a prior attempt
            try:
                W = _dequant_qkv_weight(qkv)
            except (AttributeError, RuntimeError) as exc:
                # vLLM 0.21 (Marlin) raised AttributeError here when called before
                # process_weights_after_loading; vLLM 0.22.1 (Machete) instead raises
                # RuntimeError (grouped-scale layout not repacked yet). Both mean the
                # quant buffers aren't ready → DEFER to the lazy fallback in
                # precompute_and_store_context_kv (fires at inference, post-processing done).
                _log(f"fused-KV build deferred (quant weights not post-processed yet): "
                     f"{type(exc).__name__}: {exc}")
                return  # lazy fallback retries once the kernel buffers are ready
            object.__setattr__(qkv, "weight", W)
            _log(f"attached dequantized .weight {tuple(W.shape)} to {type(qkv).__name__}")
        return orig(self)

    _build._eqc_patched = True
    Model._build_fused_kv_buffers = _build
    _log("patched DFlashQwen3Model._build_fused_kv_buffers (Bug 2: dequant fused K/V)")


def _patch_attention_forward(m) -> None:
    """Bug 3 (cudagraph): replace DFlashQwen3Attention.forward so the q/k/v
    projection goes through the quant-aware `self.qkv_proj(x)` apply instead of
    the raw `F.linear(x, self.qkv_proj.weight, bias)` at line ~134.

    On a W4A16 draft the raw `.weight` doesn't exist (only weight_packed). With
    cudagraph the engine aot-compiles `forward` during profile_run, and the
    compiled trace resolves `qkv_proj.weight` through the module's parameter dict
    → `KeyError: 'weight'` (Bug 2's lazy `.weight` attach fires too late, and even
    pre-attaching it as a buffer doesn't survive the compiled trace). Calling
    `self.qkv_proj(hidden_states)` uses the layer's own quant method (Machete) —
    exactly what every other vLLM linear does — so the compiled graph never
    references a raw `.weight`. Numerically identical to the original (same GEMM +
    bias); for a BF16 draft it's the ordinary F.linear path. Enables eager too.

    The rest of forward (per-head q/k norm, RoPE, attention, o_proj) is copied
    verbatim from vLLM 0.22.1 qwen3_dflash.py — keep in sync if that file changes.
    """
    Attn = m.DFlashQwen3Attention
    if getattr(Attn.forward, "_eqc_patched", False):
        return

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        ).view(q_shape)
        k = self.k_norm(
            k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        ).view(k_shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    forward._eqc_patched = True
    Attn.forward = forward
    _log("patched DFlashQwen3Attention.forward (Bug 3: quant-aware qkv_proj apply)")


_SWA_ENV = "EQC_DFLASH_SWA_WINDOW"
_SWA_STATIC_ENV = "EQC_DFLASH_SWA_STATIC"


def _patch_drafter_sliding_window(m, window: int, static: bool = False) -> None:
    """Apply a SYMMETRIC sliding-window of `window` to the DFlash drafter attention.

    DFlashQwen3Attention is NON-CAUSAL (mask-token queries attend over the
    context AND each other bidirectionally — block-diffusion parallel
    drafting). The FlashAttention impl reads `self.sliding_window=(left,right)`
    per forward and passes it as `window_size` to flash_attn_varlen (which
    honors it for non-causal too). We set ONLY the impl window, NOT the layer's
    `sliding_window` int, so `get_kv_cache_spec` keeps a FullAttentionSpec
    (full cache, no rotation/eviction, no `tuple - int` spec crash). A SYMMETRIC
    window `(W-1, W-1)` lets a query attend keys in `[q-(W-1), q+(W-1)]`, which
    masks nothing iff seqlen <= W.

    TWO MODES:
      - CONDITIONAL (default): wrap the impl forward and set the window
        PER-FORWARD only when `attn_metadata.max_seq_len > window`. For seqlen
        <= window the window would mask nothing anyway, so we pass FULL
        attention `(-1, -1)` — the dense kernel path, NOT flash-attn's
        local-attention path. This makes SWA a TRUE no-op below the window.
        Why it matters: even when the window clips nothing, flash-attn's
        local-attention code path is numerically/kernel-wise different from the
        dense path; on FILLER (huge drafter margins) acceptance is unmoved, but
        on REAL prose (marginal drafts, accept ~2-3) that tiny perturbation
        flips short-category drafts → short regressed +15% (smoke_test.md §19).
        Reading `max_seq_len` is a python int (no GPU sync); the attn impl
        forward runs OUTSIDE the piecewise cudagraph (attention is the split
        point) and `window_size` is a per-call kernel arg, so per-forward
        mutation is graph-safe.
      - STATIC (`EQC_DFLASH_SWA_STATIC=1`): the original behavior — set
        `impl.sliding_window = (W-1, W-1)` once in __init__. Kept for A/B.

    (Historical note: the earlier broken attempt used `per_layer_sliding_window`
    → LEFT-ONLY `(W-1, 0)` which imposes causality on the non-causal drafter and
    degraded ANY window; the symmetric impl-only window fixed that.)
    """
    Attn = m.DFlashQwen3Attention
    orig = Attn.__init__
    if getattr(orig, "_eqc_swa_patched", False):
        return
    sym = (window - 1, window - 1)
    full = (-1, -1)

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        impl = getattr(self.attn, "impl", None)
        if impl is None:
            return
        if static:
            impl.sliding_window = sym
            return
        if getattr(impl, "_eqc_cond_swa", False):
            return
        impl.sliding_window = full  # default: no window
        orig_fwd = impl.forward  # bound method on THIS impl instance

        def _cond_forward(*a, **kw):
            # impl.forward(layer, query, key, value, kv_cache, attn_metadata, ...)
            am = kw.get("attn_metadata")
            if am is None and len(a) >= 6:
                am = a[5]
            msl = getattr(am, "max_seq_len", None)
            impl.sliding_window = sym if (msl is not None and msl > window) else full
            return orig_fwd(*a, **kw)

        impl.forward = _cond_forward
        impl._eqc_cond_swa = True

    __init__._eqc_swa_patched = True
    Attn.__init__ = __init__
    mode = "static" if static else "conditional(no-op below window)"
    _log(f"patched DFlashQwen3Attention.__init__ (SWA {mode}, window={window}, sym={sym}, FullAttentionSpec)")


def register() -> None:
    """vLLM general-plugin entrypoint. No-op unless one of the EQC env vars is set.

    EQC_DFLASH_QUANT_PATCH=1   → enable W4A16 quantized-drafter loading (Bugs 1+2).
    EQC_DFLASH_SWA_WINDOW=<N>  → bound the drafter attention to the last N
                                 context tokens (sliding window, approach A).
    The two are independent and may be combined.
    """
    quant = os.environ.get(_ENV) == "1"
    swa_raw = os.environ.get(_SWA_ENV)
    swa = None
    if swa_raw:
        try:
            swa = int(swa_raw)
        except ValueError:
            _log(f"invalid {_SWA_ENV}={swa_raw!r}; ignoring")
    if not quant and not (swa and swa > 0):
        return
    try:
        import vllm.model_executor.models.qwen3_dflash as m
    except Exception as exc:  # pragma: no cover
        _log(f"qwen3_dflash import failed: {type(exc).__name__}: {exc}")
        return
    if quant:
        _patch_decoder_layer(m)      # Bug 1: quant_config forwarding
        _patch_fused_kv(m)           # Bug 2: dequant fused K/V (lazy)
        _patch_attention_forward(m)  # Bug 3: quant-aware forward (cudagraph fix)
        _log("DFlash W4A16 drafter patches active")
    if swa and swa > 0:
        static = os.environ.get(_SWA_STATIC_ENV) == "1"
        _patch_drafter_sliding_window(m, swa, static=static)
        mode = "static" if static else "conditional"
        _log(f"DFlash drafter sliding-window active: window={swa} mode={mode}")
