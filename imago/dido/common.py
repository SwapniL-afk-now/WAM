"""Shared plumbing for the DIDO-style trainers (single node, torchrun, 2–3 GPUs).

Design choices for 2–3 × 96 GB (see research_plan/dido_implementation_guide.md):

* Full-parameter training: every trained module is an FSDP2 unit
  (``imago/fsdp.py``); checkpoints are gathered with ``full_state`` and
  written in FastWAM's own format by ``export_full_fastwam``.
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
from imago.ckpt import load_checkpoint_checked
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


def load_fastwam(cfg: DictConfig, device: torch.device, attach_interaction: bool = False):
    """Compose the FastWAM config, instantiate Optional-IDM and load the checkpoint.

    ``attach_interaction`` adds the DIDO interaction tokens *before* loading, so
    a Stage I/II checkpoint's tokens and heads are restored.
    """
    os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
    fcfg = _compose(
        cfg.fastwam.config_dir or _fastwam_config_dir(),
        cfg.fastwam.config_name,
        list(cfg.fastwam.get("overrides", []) or []),
    )
    fcfg.model.load_text_encoder = False  # datasets provide cached text embeddings
    model = _instantiate(fcfg.model, torch.bfloat16, str(device))
    if attach_interaction:
        from imago.dido import interaction as inter

        inter.attach(model.video_expert, inter.InteractionConfig(**dict(cfg.interaction)))
    load_checkpoint_checked(model, os.path.expanduser(str(cfg.model_path)))
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


def full_state(runner) -> dict:
    """Gather a (possibly FSDP2-sharded) Runner's full state dict on CPU; collective."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        sd = get_model_state_dict(runner, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    else:
        sd = {k: v.detach().cpu() for k, v in runner.state_dict().items()}
    return {k[len("target."):] if k.startswith("target.") else k: v for k, v in sd.items()}


def export_full_fastwam(model, video_runner, mot_keys, frozen_state, path, meta, rank,
                        action_runner=None, proprio_runner=None) -> None:
    """Write a FastWAM-format checkpoint from sharded training modules (collective call).

    ``mot_keys`` / ``frozen_state`` are captured before sharding: keys of
    ``model.mot.state_dict()`` and the tensors of every module that is not
    being trained, so the checkpoint is complete and loadable with
    ``FastWAM.load_checkpoint`` (``{"mot", "proprio_encoder"}``).
    """
    video = full_state(video_runner)
    action = full_state(action_runner) if action_runner is not None else None
    proprio = full_state(proprio_runner) if proprio_runner is not None else None
    if rank != 0:
        return
    mot = {}
    for key in mot_keys:
        if key.startswith("mixtures.video."):
            mot[key] = video[key[len("mixtures.video."):]].to(torch.bfloat16)
        elif action is not None and key.startswith("mixtures.action."):
            mot[key] = action[key[len("mixtures.action."):]].to(torch.bfloat16)
        else:
            mot[key] = frozen_state[key]
    payload = {"mot": mot, "imago_meta": meta}
    if proprio is not None:
        payload["proprio_encoder"] = {k: v.to(torch.bfloat16) for k, v in proprio.items()}
    elif getattr(model, "proprio_encoder", None) is not None:
        payload["proprio_encoder"] = {k: v.detach().cpu() for k, v in model.proprio_encoder.state_dict().items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"[dido] exported {path}", flush=True)


def load_yaml(path: str, overrides: list[str]) -> DictConfig:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg
