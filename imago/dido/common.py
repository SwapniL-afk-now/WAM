"""Shared plumbing for the DIDO-style trainers (single node, torchrun, 2–3 GPUs).

Design choices for 2–3 × 96 GB (see research_plan/dido_implementation_guide.md):

* One frozen bf16 replica of FastWAM per GPU (~13 GB). Everything that trains
  (LoRA adapters, the action expert in Stage II) is small enough to replicate,
  so gradients are simply all-reduced; no FSDP needed.
* Data comes from FastWAM's own LeRobot datasets and ``build_inputs`` (same
  normalisation and text-embedding cache as FastWAM training).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from imago.builder import _compose, _instantiate
from imago.lora import unwrap_lora
from imago.policy import ImagoPolicyConfig


def init_distributed() -> tuple[int, int, torch.device]:
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, world, device


def load_fastwam(cfg: DictConfig, device: torch.device):
    """Compose the FastWAM config, instantiate Optional-IDM and load the checkpoint."""
    os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
    fcfg = _compose(
        cfg.fastwam.config_dir or _fastwam_config_dir(),
        cfg.fastwam.config_name,
        list(cfg.fastwam.get("overrides", []) or []),
    )
    fcfg.model.load_text_encoder = False  # datasets provide cached text embeddings
    model = _instantiate(fcfg.model, torch.bfloat16, str(device))
    model.load_checkpoint(os.path.expanduser(str(cfg.model_path)))
    model.requires_grad_(False)
    model.policy_cfg = ImagoPolicyConfig(token_refine=_refine_cfg(cfg))
    return model, fcfg


def _refine_cfg(cfg):
    from imago.dido.token_refine import TokenRefineConfig

    return TokenRefineConfig(**dict(cfg.get("token_refine", {}) or {}))


def _fastwam_config_dir() -> str:
    from rlinf.models.embodiment.fastwam import _default_fastwam_config_dir

    return _default_fastwam_config_dir()


def build_loader(fcfg: DictConfig, batch_size: int, rank: int, world: int, seed: int):
    from fastwam.runtime import build_datasets

    train_ds, _ = build_datasets(fcfg.data)
    sampler = torch.utils.data.DistributedSampler(
        train_ds, num_replicas=world, rank=rank, shuffle=True, seed=seed, drop_last=True
    )
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(fcfg.get("num_workers", 8)),
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler


def infinite(loader, sampler):
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return batch


def allreduce_grads(params, world: int) -> None:
    """Average gradients of replicated trainable parameters across ranks."""
    if world <= 1:
        return
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.reshape(-1).float() for g in grads])
    dist.all_reduce(flat)
    flat /= world
    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[offset : offset + n].view_as(g).to(g.dtype))
        offset += n


def lr_lambda(kind: str, warmup: int, total: int):
    """``const``: linear warm-up then constant (DIDO teacher prep / Stage I).
    ``cosine``: warm-up then cosine to 0 (DIDO Stage II)."""

    def fn(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if kind == "const":
            return 1.0
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def fp32_(params) -> None:
    """Keep trainable adapters in fp32 (AdamW on bf16 params is lossy)."""
    for p in params:
        p.data = p.data.float()


@torch.no_grad()
def export_fastwam_checkpoint(model, path: str, meta: dict) -> None:
    """Save a FastWAM-format checkpoint (``{"mot", "proprio_encoder"}``) that
    ``imago.builder.get_model`` / ``FastWAM.load_checkpoint`` can load directly.
    LoRA adapters must already be merged; wrappers are removed here."""
    unwrap_lora(model)
    payload = {"mot": {k: v.detach().to(torch.bfloat16).cpu() for k, v in model.mot.state_dict().items()}}
    if getattr(model, "proprio_encoder", None) is not None:
        payload["proprio_encoder"] = model.proprio_encoder.state_dict()
    payload["imago_meta"] = meta
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_yaml(path: str, overrides: list[str]) -> DictConfig:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg
