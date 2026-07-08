#!/usr/bin/env python3
"""train_cyankiwi_v2.py — QAD with the cyankiwi INT4 grid PRESERVED.

cyankiwi's pure-PTQ checkpoint is a strong W4 reasoning baseline (passes all
firm gates). We use it as the QAD student init and inject cyankiwi's saved
per-group scales as modelopt's ``_amax`` so the fake-quant grid matches
cyankiwi exactly. QAD distillation then refines the BF16 latent
weights against the BF16 teacher without disturbing the grid.

Why the naive path failed: modelopt's default ``mtq.quantize(QAD_CFG,
forward_loop)`` runs max-cal on the cyankiwi-dequant weights and picks
slightly different scales than cyankiwi's MSE-cal. The +7 % step-0 KL
delta we measured destroyed reasoning capacity.

Path-B (this script) bypasses that:
  1. Build the student via cyankiwi dequant overlay (BF16 weights on
     cyankiwi's INT4 grid).
  2. Construct QADTrainer normally.
  3. Pre-call ``trainer._quantize_model()`` BEFORE ``trainer.train()``
     to install weight_quantizer modules + initialize their ``_amax``
     buffers (we throw those away).
  4. Walk the student, overwrite each enabled weight_quantizer's
     ``_amax`` with ``cyankiwi_scale × 7`` (the lossless combo
     identified by ``verify_cyankiwi_init.py``: modelopt scale =
     amax / 7, so amax = cyk_scale × 7 → scale_mo = cyk_scale; with
     ``narrow_range=False`` the grid [-8, 7] matches cyankiwi exactly).
  5. Gate check: forward KL of injected student vs BF16 teacher MUST
     match cyankiwi face-value (~0.0372). If not, abort.
  6. ``trainer.train()`` runs distillation; QADTrainer's lazy
     ``_quantize_model`` check sees ``is_quantized(model) == True`` and
     skips re-cal, preserving our injection.
  7. After train: ``mtq.compress`` + ``save_pretrained`` (modelopt
     state co-saved → export.py / to_compressed_tensors.py downstream).

QAD_CFG is locally modified to set ``narrow_range=False`` on the
``*weight_quantizer`` rule so the fake-quant kernel uses the full
[-8, 7] grid (cyankiwi's actual int4 range).

Usage (>=2 GPUs required — single-GPU OOMs at the first train step):

    scripts/train_multi.sh 0,1,2,3,4,5,6,7 src/configs/train_config.yaml
        # == torchrun --nproc_per_node=8 src/qad/train_cyankiwi_v2.py \\
        #      --train-config src/configs/train_config.yaml \\
        #      --packed runs/qad_nemotron_regen/train_data/packed.pt \\
        #      --output-dir runs/qad_nemotron_regen/ckpts_full \\
        #      --log-dir runs/qad_nemotron_regen/logs \\
        #      --deepspeed src/configs/ds_zero2.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from safetensors import safe_open
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset

SRC_ROOT = Path(__file__).resolve().parents[1]  # src/, so `qad.` imports resolve
REPO_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))

from qad.quant_config import QAD_CFG, verify_mask  # noqa: E402
from qad.cyankiwi_init import (  # noqa: E402
    DEFAULT_CYANKIWI_PATH,
    load_cyankiwi_dequant_state_dict,
)


# ---------------------------------------------------------------------------
# Build a narrow_range=False variant of QAD_CFG
# ---------------------------------------------------------------------------


def build_qad_cfg_nr_false() -> dict:
    cfg = copy.deepcopy(QAD_CFG)
    n_patched = 0
    for entry in cfg.get("quant_cfg", []):
        if isinstance(entry, dict) and entry.get("quantizer_name") == "*weight_quantizer":
            entry_cfg = entry.setdefault("cfg", {})
            entry_cfg["narrow_range"] = False
            n_patched += 1
    if n_patched == 0:
        raise RuntimeError("Could not find '*weight_quantizer' rule in QAD_CFG to patch.")
    return cfg


def load_cyankiwi_raw_scales(cyankiwi_path: str | Path) -> dict[str, torch.Tensor]:
    from qad.cyankiwi_init import _resolve_cyankiwi_dir

    st = _resolve_cyankiwi_dir(cyankiwi_path) / "model-00001-of-00001.safetensors"
    out: dict[str, torch.Tensor] = {}
    with safe_open(str(st), framework="pt") as f:
        for k in f.keys():
            if k.endswith(".weight_scale"):
                out[k[: -len(".weight_scale")]] = f.get_tensor(k)
    return out


# ---------------------------------------------------------------------------
# Loss helper (kept here for smoke.py to re-use)
# ---------------------------------------------------------------------------


# Token-chunk size for the KL upcast. The full-vocab float32 logits tensor
# ([T, V] = 16384 × 248044 ≈ 16 GiB each for student+teacher) OOMs even on an
# 80 GB GPU once the two BF16 logits, both models, and the 8-bit optimizer
# state are already resident. We compute the per-token KL in chunks over the
# flattened token dim so peak float memory is CHUNK×V instead of T×V — the
# math (a per-token mean over the same tokens) is identical. Tune via
# EQC_KL_CHUNK_TOKENS (default 2048).
import os as _os
_KL_CHUNK_TOKENS = int(_os.environ.get("EQC_KL_CHUNK_TOKENS", "2048"))


def kl_div_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                  temperature: float = 1.0, ignore_index: int | None = None,
                  input_ids: torch.Tensor | None = None) -> torch.Tensor:
    # Flatten [B, T, V] -> [N, V] so chunking is a simple row split that keeps
    # autograd flowing into the student logits slices.
    V = student_logits.shape[-1]
    s_flat = student_logits.reshape(-1, V)
    t_flat = teacher_logits.reshape(-1, V)
    n_tok = s_flat.shape[0]

    if ignore_index is not None and input_ids is not None:
        if input_ids.device != s_flat.device:
            input_ids = input_ids.to(s_flat.device, non_blocking=True)
        mask = (input_ids.reshape(-1) != ignore_index).to(torch.float32)
    else:
        mask = None

    if mask is None:
        mask = torch.ones(n_tok, device=s_flat.device, dtype=torch.float32)
    denom = mask.sum().clamp_min(1.0)

    def _chunk_weighted_kl(s_slice, t_slice, m_slice):
        # full-vocab float upcast for one token chunk; under activation
        # checkpointing the float log_softmax is recomputed in backward
        # instead of being stored, keeping peak memory ~ CHUNK×V.
        s = s_slice.float() / temperature
        t = t_slice.float() / temperature
        s_logp = F.log_softmax(s, dim=-1)
        t_p = F.softmax(t, dim=-1)
        per_tok = (t_p * (torch.log(t_p.clamp_min(1e-12)) - s_logp)).sum(dim=-1)
        return (per_tok * m_slice).sum()

    chunk = _KL_CHUNK_TOKENS if _KL_CHUNK_TOKENS > 0 else n_tok
    weighted_sum = student_logits.new_zeros((), dtype=torch.float32)
    use_ckpt = torch.is_grad_enabled() and s_flat.requires_grad
    for i in range(0, n_tok, chunk):
        j = min(i + chunk, n_tok)
        if use_ckpt:
            contrib = checkpoint(_chunk_weighted_kl, s_flat[i:j], t_flat[i:j],
                                 mask[i:j], use_reentrant=False)
        else:
            contrib = _chunk_weighted_kl(s_flat[i:j], t_flat[i:j], mask[i:j])
        weighted_sum = weighted_sum + contrib
    return weighted_sum / denom


@dataclass
class TrainConfig:
    teacher_model: str = "Qwen/Qwen3.5-4B"
    student_init: str = "Qwen/Qwen3.5-4B"
    cyankiwi_path: str = DEFAULT_CYANKIWI_PATH
    seed: int = 42

    wandb_project: str = "edgefm-eqc-qad"
    wandb_run_name: str | None = None

    optimizer: str = "adamw_8bit"
    lr: float = 5.0e-6
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.95)
    eps: float = 1.0e-8
    grad_clip: float = 1.0

    lr_schedule: str = "cosine"
    warmup_ratio: float = 0.03
    epochs: int = 8
    max_steps: int = -1

    max_seq_len: int = 16384
    micro_batch_size: int = 1
    grad_accum_steps: int = 8

    kl_temperature: float = 1.0
    mtp_alpha: float = 0.0

    mixed_precision: str = "bf16"
    grad_checkpointing: bool = True

    # calib_size is no longer the source of amax.
    # We still need a small forward_loop (≥1 sample) to install
    # quantizer modules; their amax is immediately overwritten with
    # cyankiwi_scale × 7. Keep at 8 (fast install).
    calib_size: int = 8

    eval_strategy: str = "epoch"
    eval_steps: int = 250
    save_strategy: str = "epoch"
    save_steps: int = 1000
    eval_every_steps: int = 250
    save_every_epochs: int = 1
    keep_last_ckpts: int = 0
    log_every_steps: int = 10

    # Grid gate: forward KL after amax injection must be within this
    # absolute delta of cyankiwi face-value (~0.0372). Aborts training
    # otherwise. Conservative: 0.005 = 13% of face-value.
    inject_kl_tolerance: float = 0.005


def load_train_config(path: Path) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = TrainConfig()
    for k, v in raw.items():
        if not hasattr(cfg, k):
            print(f"[warn] train_config: unknown key '{k}' ignored", file=sys.stderr)
            continue
        if k == "betas" and isinstance(v, list):
            v = tuple(v)
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# CompositeKDLoss — main KL + α · MTP KL via attribute-cached MTP captures
# ---------------------------------------------------------------------------


def _try_get_mtp_logits(out, candidates=("mtp_logits", "speculative_logits")):
    for k in candidates:
        v = getattr(out, k, None)
        if v is not None:
            return v
    if isinstance(out, dict):
        for k in candidates:
            v = out.get(k)
            if v is not None:
                return v
    return None


from torch.nn.modules.loss import _Loss as _TorchLoss


class CompositeKDLoss(_TorchLoss):
    def __init__(self, mtp_alpha: float = 0.5, temperature: float = 1.0,
                 pad_id: int = -100):
        super().__init__(reduction="mean")
        self.mtp_alpha = mtp_alpha
        self.temperature = temperature
        self.pad_id = pad_id
        self._t_mtp: Optional[torch.Tensor] = None
        self._s_mtp: Optional[torch.Tensor] = None
        self._t_handle = None
        self._s_handle = None
        self._input_ids: Optional[torch.Tensor] = None

    def _capture_teacher(self, _mod, _inp, output):
        self._t_mtp = output[0] if isinstance(output, (tuple, list)) else output

    def _capture_student(self, _mod, _inp, output):
        self._s_mtp = output[0] if isinstance(output, (tuple, list)) else output

    def attach_mtp_hooks(self, teacher: torch.nn.Module, student: torch.nn.Module) -> None:
        for name, m in teacher.named_modules():
            if name.endswith("mtp.fc"):
                self._t_handle = m.register_forward_hook(self._capture_teacher)
                print(f"[mtp] hooked teacher.{name}")
                break
        for name, m in student.named_modules():
            if name.endswith("mtp.fc"):
                self._s_handle = m.register_forward_hook(self._capture_student)
                print(f"[mtp] hooked student.{name}")
                break

    def remove_hooks(self) -> None:
        for h in (self._t_handle, self._s_handle):
            if h is not None:
                h.remove()
        self._t_handle = self._s_handle = None

    def set_input_ids(self, input_ids: Optional[torch.Tensor]) -> None:
        self._input_ids = input_ids

    def forward(self, out_student, out_teacher) -> torch.Tensor:
        s_logits = out_student.logits if hasattr(out_student, "logits") else out_student
        t_logits = out_teacher.logits if hasattr(out_teacher, "logits") else out_teacher
        t_logits = t_logits.detach()

        loss_main = kl_div_logits(
            s_logits, t_logits, temperature=self.temperature,
            ignore_index=self.pad_id, input_ids=self._input_ids,
        )

        t_mtp = _try_get_mtp_logits(out_teacher) or self._t_mtp
        s_mtp = _try_get_mtp_logits(out_student) or self._s_mtp

        loss_total = loss_main
        if self.mtp_alpha > 0.0 and t_mtp is not None and s_mtp is not None:
            t_mtp = t_mtp.detach()
            loss_mtp = kl_div_logits(
                s_mtp, t_mtp, temperature=self.temperature,
                ignore_index=self.pad_id, input_ids=self._input_ids,
            )
            loss_total = loss_main + self.mtp_alpha * loss_mtp
            self.last_loss_main = float(loss_main.detach())
            self.last_loss_mtp = float(loss_mtp.detach())
        else:
            self.last_loss_main = float(loss_main.detach())
            self.last_loss_mtp = 0.0

        self._t_mtp = self._s_mtp = None
        return loss_total


def _capture_run_meta(args, cfg: TrainConfig, packed_meta: dict | None) -> dict:
    import hashlib
    import platform
    import subprocess

    def _sh(cmd):
        try:
            return subprocess.check_output(cmd, cwd=str(REPO_ROOT),
                                           stderr=subprocess.DEVNULL,
                                           timeout=5).decode().strip()
        except Exception:
            return ""

    pkg_versions = {}
    for mod in ("torch", "transformers", "modelopt", "bitsandbytes", "datasets",
                "accelerate", "compressed_tensors"):
        try:
            pkg_versions[mod] = __import__(mod).__version__
        except Exception:
            pkg_versions[mod] = None

    packed_hash = ""
    try:
        with open(args.packed, "rb") as f:
            packed_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        pass

    return {
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "started_at_unix": int(time.time()),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": {
            "sha": _sh(["git", "rev-parse", "HEAD"]),
            "branch": _sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(_sh(["git", "status", "--porcelain"])),
        },
        "package_versions": pkg_versions,
        "cli_args": vars(args),
        "train_config": cfg.__dict__,
        "data_packed_path": str(args.packed),
        "data_packed_sha256_16": packed_hash,
        "data_packed_meta": packed_meta or {},
        "student_source": "cyankiwi-dequant-overlay+amax-inject",
        "cyankiwi_path": cfg.cyankiwi_path,
        "qad_cfg_modifier": "narrow_range=False on *weight_quantizer",
        "amax_inject_K": 7,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-config", required=True)
    p.add_argument("--packed", required=True)
    p.add_argument("--val-packed", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--cyankiwi-path", default=None)
    p.add_argument("--inject-only-no-train", action="store_true",
                   help="Build student + inject amax + measure KL, but skip training. "
                        "Smoke test for the inject path.")
    p.add_argument("--with-compress", action="store_true",
                   help="Re-enable mtq.compress at the end of training. Default OFF "
                        "— empirically adds KL via re-quantization. "
                        "Use pack_bf16_to_ct.py instead to produce the deployment "
                        "artifact from the BF16 'final/' or 'checkpoint-N/' folder.")
    p.add_argument("--deepspeed", default=None,
                   help="Path to a DeepSpeed config JSON. Enables ZeRO sharding of "
                        "optimizer/grads across the launched ranks (multi-GPU). The "
                        "seq-16384 full-vocab logits don't fit one 80 GB GPU; ZeRO-2 "
                        "frees enough to fit. Launch via scripts/train_multi.sh.")
    return p.parse_args()


class _PadDataset(Dataset):
    def __init__(self, packed: torch.Tensor):
        self.packed = packed

    def __len__(self):
        return self.packed.shape[0]

    def __getitem__(self, idx: int):
        ids = self.packed[idx].long()
        return {"input_ids": ids, "labels": ids.clone()}


def inject_cyankiwi_amax(model, cyk_scales: dict[str, torch.Tensor],
                         K: int = 7) -> tuple[int, int]:
    """Walk model, overwrite each enabled weight_quantizer's _amax with
    cyk_scales[name] × K. Returns (n_written, n_skipped)."""
    n_written = 0
    n_skipped_no_scale = 0
    n_skipped_shape = 0
    for name, mod in model.named_modules():
        wq = getattr(mod, "weight_quantizer", None)
        if wq is None or not getattr(wq, "is_enabled", False):
            continue
        amax_t = getattr(wq, "_amax", None)
        if not isinstance(amax_t, torch.Tensor):
            continue
        # Strip optional "_teacher_model." / "_student." prefixes from
        # DistillationModel-wrapped models when looking up cyankiwi scales.
        lookup_name = name
        for pref in ("_teacher_model.", "_student."):
            if lookup_name.startswith(pref):
                lookup_name = lookup_name[len(pref):]
                break
        scale = cyk_scales.get(lookup_name)
        if scale is None:
            n_skipped_no_scale += 1
            continue
        new_amax = (scale.float() * K).to(amax_t.device, amax_t.dtype)
        if new_amax.numel() != amax_t.numel():
            n_skipped_shape += 1
            continue
        amax_t.data.copy_(new_amax.view(amax_t.shape))
        n_written += 1
    return n_written, n_skipped_no_scale + n_skipped_shape


def main() -> int:
    args = parse_args()
    cfg = load_train_config(Path(args.train_config))
    if args.cyankiwi_path:
        cfg.cyankiwi_path = args.cyankiwi_path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir or args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_ts = int(time.time())
    log_path = log_dir / f"train-{run_ts}.jsonl"
    print(f"[info] step log: {log_path}")

    cfg_snapshot_path = out_dir / "config_snapshot.yaml"
    try:
        with open(cfg_snapshot_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg.__dict__, f, sort_keys=False, allow_unicode=True)
        print(f"[info] config snapshot: {cfg_snapshot_path}")
    except Exception as e:
        print(f"[warn] failed to write config_snapshot.yaml: {e}", file=sys.stderr)

    torch.manual_seed(cfg.seed)

    print("[info] importing transformers + modelopt…")
    t0 = time.time()
    from transformers import (
        AutoTokenizer, TrainingArguments, TrainerCallback,
        Qwen3_5ForConditionalGeneration,
    )
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.plugins.transformers_trainer import (
        QADTrainer, QuantizationArgumentsWithConfig,
    )
    mto.enable_huggingface_checkpointing()
    print(f"[info] imports done in {time.time()-t0:.1f}s")

    # === Build the narrow_range=False QAD_CFG variant ===
    qad_cfg_nrf = build_qad_cfg_nr_false()
    print(f"[info] QAD_CFG patched: narrow_range=False on *weight_quantizer")

    # ---- Packed data ----
    print(f"[info] loading packed train: {args.packed}")
    blob = torch.load(args.packed, map_location="cpu")
    train_ids = blob["input_ids"] if isinstance(blob, dict) else blob
    meta = blob.get("meta", {}) if isinstance(blob, dict) else {}
    pad_id = int(meta.get("pad_id", -100))
    print(f"[info] train.shape={tuple(train_ids.shape)}  pad_id={pad_id}")

    val_ids = None
    val_path_arg = args.val_packed
    if val_path_arg is None:
        candidate = Path(args.packed).with_name(Path(args.packed).stem + "_val" + Path(args.packed).suffix)
        if candidate.exists():
            val_path_arg = str(candidate)
            print(f"[info] auto-detected val: {val_path_arg}")
    if val_path_arg:
        try:
            vb = torch.load(val_path_arg, map_location="cpu")
            val_ids = vb["input_ids"] if isinstance(vb, dict) else vb
            print(f"[info] val.shape={tuple(val_ids.shape)}")
        except Exception as e:
            print(f"[warn] failed to load val: {e}", file=sys.stderr)

    run_meta = _capture_run_meta(args, cfg, packed_meta=meta)
    meta_path = out_dir / f"run_meta-{run_ts}.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2, default=str)
        print(f"[info] run meta:       {meta_path}")
    except Exception as e:
        print(f"[warn] failed to write run_meta.json: {e}", file=sys.stderr)

    # ---- Models ----
    print(f"[info] loading teacher: {cfg.teacher_model}")
    teacher = Qwen3_5ForConditionalGeneration.from_pretrained(
        cfg.teacher_model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"[info] loading student arch from: {cfg.student_init}")
    student = Qwen3_5ForConditionalGeneration.from_pretrained(
        cfg.student_init, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    print(f"[info] overlaying cyankiwi dequant weights from: {cfg.cyankiwi_path}")
    cyankiwi_sd = load_cyankiwi_dequant_state_dict(
        cyankiwi_path=cfg.cyankiwi_path, device="cpu"
    )
    missing, unexpected = student.load_state_dict(cyankiwi_sd, strict=False)
    if missing:
        unexpected_missing = [k for k in missing if k != "lm_head.weight"]
        if unexpected_missing:
            print(f"[fatal] overlay: unexpected missing: {unexpected_missing[:10]}", file=sys.stderr)
            return 2
        print("[info] missing 'lm_head.weight' (tied to embed_tokens, OK)")
    if unexpected:
        non_mtp = [k for k in unexpected if not k.startswith("mtp.")]
        if non_mtp:
            print(f"[fatal] overlay: unexpected non-MTP: {non_mtp[:10]}", file=sys.stderr)
            return 2
        print(f"[info] ignored {len(unexpected)} cyankiwi MTP-only tensors")

    if cfg.grad_checkpointing:
        try:
            student.gradient_checkpointing_enable()
            print("[info] student gradient_checkpointing enabled")
        except Exception as e:
            print(f"[warn] gradient_checkpointing_enable failed: {e}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model, trust_remote_code=True)

    # Pre-load cyankiwi raw scales (need them for amax injection)
    cyk_scales = load_cyankiwi_raw_scales(cfg.cyankiwi_path)
    print(f"[info] loaded {len(cyk_scales)} cyankiwi raw scales for amax injection")

    # ---- CompositeKDLoss + MTP hooks ----
    criterion = CompositeKDLoss(
        mtp_alpha=cfg.mtp_alpha,
        temperature=cfg.kl_temperature,
        pad_id=pad_id,
    )
    criterion.attach_mtp_hooks(teacher, student)

    distill_cfg = {
        "teacher_model": teacher,
        "criterion": criterion,
    }

    optim_name_map = {
        "adamw_8bit": "adamw_bnb_8bit",
        "adamw_bnb_8bit": "adamw_bnb_8bit",
        "bnb.optim.adamw8bit": "adamw_bnb_8bit",
        "adamw": "adamw_torch",
        "adamw_torch": "adamw_torch",
    }
    optim = optim_name_map.get(cfg.optimizer.lower(), cfg.optimizer)

    wandb_run_name = cfg.wandb_run_name
    if wandb_run_name is None:
        lr_tag = f"{cfg.lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        cap_tag = (f"step{cfg.max_steps}" if cfg.max_steps > 0
                   else f"ep{cfg.epochs}")
        wandb_run_name = (
            f"qad-cyankiwi-init-v2-lr{lr_tag}-ga{cfg.grad_accum_steps}"
            f"-seq{cfg.max_seq_len}-{cap_tag}-mtp{cfg.mtp_alpha}"
        )
    import os
    # Multi-GPU (DeepSpeed/torchrun) topology. Keep the effective batch =
    # micro_batch × grad_accum × world_size equal to the single-GPU recipe
    # (1 × 16 = 16) by dividing grad_accum across ranks, so "8000 steps" still
    # consumes the same amount of data and the LR schedule is unchanged.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    is_main = int(os.environ.get("RANK", "0")) == 0
    if world_size > 1:
        if cfg.grad_accum_steps % world_size != 0:
            print(f"[fatal] grad_accum_steps {cfg.grad_accum_steps} not divisible by "
                  f"world_size {world_size}; pick a GPU count that divides it.",
                  file=sys.stderr)
            return 2
        per_rank_ga = cfg.grad_accum_steps // world_size
        print(f"[dist] world_size={world_size} -> grad_accum {cfg.grad_accum_steps} "
              f"/ {world_size} = {per_rank_ga} per rank (effective batch unchanged = "
              f"{cfg.micro_batch_size * cfg.grad_accum_steps})")
        cfg.grad_accum_steps = per_rank_ga

    # wandb is optional: enabled only when wandb_project is set (non-null).
    # Standalone default configs leave it null -> no wandb dependency.
    use_wandb = bool(cfg.wandb_project) and is_main
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
        os.environ["WANDB_RUN_NAME"] = wandb_run_name
        os.environ.setdefault("WANDB_DIR", str(out_dir))
        print(f"[wandb] project={cfg.wandb_project}  run={wandb_run_name}")
    else:
        print("[wandb] disabled (wandb_project is null); logging to stdout + JSONL only")

    if args.max_steps > 0:
        resolved_max_steps = args.max_steps
    elif cfg.max_steps > 0:
        resolved_max_steps = cfg.max_steps
    else:
        resolved_max_steps = -1

    resolved_eval_strategy = cfg.eval_strategy if val_ids is not None else "no"

    hf_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.micro_batch_size,
        per_device_eval_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        adam_beta1=cfg.betas[0],
        adam_beta2=cfg.betas[1],
        adam_epsilon=cfg.eps,
        max_grad_norm=cfg.grad_clip,
        lr_scheduler_type=cfg.lr_schedule,
        warmup_ratio=cfg.warmup_ratio,
        optim=optim,
        bf16=(cfg.mixed_precision == "bf16"),
        gradient_checkpointing=cfg.grad_checkpointing,
        logging_steps=cfg.log_every_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=(cfg.keep_last_ckpts or None),
        eval_strategy=resolved_eval_strategy,
        eval_steps=cfg.eval_steps,
        report_to=(["wandb"] if use_wandb else []),
        run_name=wandb_run_name,
        remove_unused_columns=False,
        max_steps=resolved_max_steps,
        seed=cfg.seed,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        deepspeed=args.deepspeed,
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
    )

    # ---- Datasets ----
    train_ds = _PadDataset(train_ids)
    eval_ds = _PadDataset(val_ids) if val_ids is not None else None

    def _collator(features):
        ids = torch.stack([f["input_ids"] for f in features], dim=0)
        criterion.set_input_ids(ids)
        return {"input_ids": ids, "labels": ids.clone()}

    # Only rank 0 writes the JSONL step log (avoid multi-rank truncation races).
    log_fh = open(log_path, "w", encoding="utf-8") if is_main else None
    if log_fh is not None:
        log_fh.write(json.dumps({"type": "header", "run_meta": run_meta}, default=str) + "\n")
        log_fh.flush()

    class _JsonlLogCallback(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kwargs):
            if logs is None or log_fh is None:
                return control
            rec = dict(logs)
            rec["step"] = state.global_step
            rec["epoch"] = state.epoch
            rec["last_loss_main"] = float(getattr(criterion, "last_loss_main", float("nan")))
            rec["last_loss_mtp"] = float(getattr(criterion, "last_loss_mtp", float("nan")))
            try:
                log_fh.write(json.dumps(rec, default=str) + "\n")
                log_fh.flush()
            except Exception:
                pass
            return control

        def on_train_end(self, args_, state, control, **kwargs):
            try:
                log_fh.close()
            except Exception:
                pass
            criterion.remove_hooks()
            return control

    class _MaskAuditCallback(TrainerCallback):
        def on_train_begin(self, args_, state, control, **kwargs):
            inner = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
            audit = verify_mask(inner)
            print(f"[audit] n_quantized={audit.n_quantized}  n_skipped={audit.n_skipped}  "
                  f"misquant={len(audit.misquantized)}  misignore={len(audit.misignored)}")
            return control

    class _DistillEvalQADTrainer(QADTrainer):
        def prediction_step(self, model, inputs, prediction_loss_only,
                            ignore_keys=None):
            try:
                target_device = model.device
            except AttributeError:
                try:
                    target_device = next(model.parameters()).device
                except StopIteration:
                    target_device = torch.device("cuda:0")
            input_ids = inputs.get("input_ids")
            if input_ids is not None and input_ids.device != target_device:
                input_ids = input_ids.to(target_device, non_blocking=True)
            criterion.set_input_ids(input_ids)
            inner = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                if hasattr(inner, "compute_kd_loss"):
                    _ = inner(input_ids=input_ids)
                    loss = inner.compute_kd_loss(
                        loss_reduction_fn=lambda l: l.mean()
                    )
                else:
                    teacher_mod = getattr(inner, "_teacher_model", None)
                    student_mod = getattr(inner, "_student", inner)
                    t_out = teacher_mod(input_ids=input_ids)
                    s_out = student_mod(input_ids=input_ids)
                    loss = criterion(s_out, t_out)
            return (loss.detach(), None, None)

    quant_args = QuantizationArgumentsWithConfig(
        quant_cfg=qad_cfg_nrf,    # narrow_range=False variant
        calib_size=cfg.calib_size,
        compress=False,
    )

    trainer = _DistillEvalQADTrainer(
        model=student,
        args=hf_args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=_collator,
        quant_args=quant_args,
        distill_config=distill_cfg,
        callbacks=[_JsonlLogCallback(), _MaskAuditCallback()],
    )

    # Under DeepSpeed the HF Trainer defers model placement until
    # deepspeed.initialize() (inside trainer.train()), leaving the model on CPU
    # — at ANY world size, including a single-GPU run. But we run the
    # calibration/inject/gate forwards BEFORE that, on the local GPU, so move
    # the student + teacher to this rank's device first (otherwise the
    # calibration index-on-cuda vs weights-on-cpu mismatch crashes).
    if torch.cuda.is_available():
        _dev = torch.device(f"cuda:{max(local_rank, 0)}")
        torch.cuda.set_device(_dev)
        trainer.model.to(_dev)
        teacher.to(_dev).eval()
        print(f"[dist] moved student+teacher to {_dev} for inject/gate")

    # ========================================================================
    # KEY STEP: pre-quantize + inject cyankiwi amax + gate-check KL
    # ========================================================================
    print()
    print("=" * 70)
    print("[inject] pre-calling trainer._quantize_model() to install quantizers")
    print("=" * 70)
    trainer._quantize_model()
    print("[inject] quantizer install done. Now overwriting _amax with "
          "cyankiwi_scale × 7…")
    n_written, n_skipped = inject_cyankiwi_amax(trainer.model, cyk_scales, K=7)
    print(f"[inject] written={n_written}, skipped={n_skipped}")
    if n_written < 180:
        print(f"[fatal] only {n_written} amax buffers written; expected 200 "
              "(MTP fc + linear_attn.in_proj_a/b + lm_head excluded). Aborting.",
              file=sys.stderr)
        return 2

    # Quick gate: forward KL on a single train batch must be near face-value
    print("[gate] measuring forward KL on 4 random sequences from train set")
    device = next(trainer.model.parameters()).device
    probe_ids = train_ids[:4, : min(4096, train_ids.shape[1])].long().to(device)
    teacher_device = next(teacher.parameters()).device
    if teacher_device.type == "cpu":
        teacher = teacher.to(device).eval()
    # Inject input_ids into criterion's mask so kl_div_logits applies pad mask
    kls = []
    with torch.no_grad():
        for i in range(probe_ids.shape[0]):
            ids = probe_ids[i:i + 1]
            t_logits = teacher(input_ids=ids).logits
            inner = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
            student_mod = getattr(inner, "_student", inner)
            s_logits = student_mod(input_ids=ids).logits
            kl = kl_div_logits(s_logits, t_logits, ignore_index=pad_id, input_ids=ids)
            kls.append(float(kl))
            print(f"  sample {i}: KL = {kls[-1]:.5f}")
    kl_mean = sum(kls) / len(kls)
    print(f"[gate] post-inject KL = {kl_mean:.5f}  (target ≈ 0.037 = cyankiwi face-value)")
    cyankiwi_facevalue_kl = 0.0372  # from verify_cyankiwi_init.py T0
    if abs(kl_mean - cyankiwi_facevalue_kl) > cfg.inject_kl_tolerance:
        print(f"[fatal] post-inject KL {kl_mean:.5f} deviates from face-value "
              f"{cyankiwi_facevalue_kl:.5f} by more than tolerance "
              f"{cfg.inject_kl_tolerance:.5f}. Aborting before training.",
              file=sys.stderr)
        return 3
    print(f"[gate] PASS: |Δ KL| = {abs(kl_mean - cyankiwi_facevalue_kl):.5f} "
          f"< tolerance {cfg.inject_kl_tolerance:.5f}")
    print("=" * 70)

    if args.inject_only_no_train:
        print("[info] --inject-only-no-train set; exiting before trainer.train()")
        return 0

    # ---- Train ----
    print(f"[info] starting QADTrainer.train()")
    trainer.train(resume_from_checkpoint=args.resume)

    # SKIP mtq.compress. We measured that mtq.compress + reload introduces
    # extra KL on top of in-memory
    # training state (ckpt-N reload preserved face-value, final/ reload
    # degraded). The HF-auto-saved `checkpoint-N` folder already contains
    # BF16 latents + modelopt_state.pth, which is the right format for
    # our custom `pack_bf16_to_ct.py` downstream — modelopt's compress
    # path is BYPASSED for the deployment artifact.
    if args.with_compress:
        print("[info] --with-compress was set; running mtq.compress(trainer.model)")
        try:
            mtq.compress(trainer.model)
            print("[info] mtq.compress done — weights packed INT4, amax frozen")
        except Exception as e:
            print(f"[fatal] mtq.compress failed: {e}", file=sys.stderr)
            raise
        print(f"[info] saving final model -> {out_dir / 'final'}")
        trainer.save_model(str(out_dir / "final"))
    else:
        print("[info] skipping mtq.compress (use checkpoint-N for pack_bf16_to_ct)")
        # Re-save the latest BF16 latents as "final" via the patched
        # save_pretrained (mto.enable_huggingface_checkpointing co-saves
        # modelopt_state.pth so the model is dequant-aware on reload).
        print(f"[info] saving uncompressed final -> {out_dir / 'final'}")
        trainer.save_model(str(out_dir / "final"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
