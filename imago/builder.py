"""Build the IMAGO policy for RLinf (``model_type: fastwam_imago``).

Reuses RLinf's FastWAM config composer (``_load_fastwam_config``) and dataset
stats resolution; adds the Optional-IDM target, LoRA injection and freezing.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from rlinf.models.embodiment.fastwam import (
    _default_fastwam_config_dir,
    _load_fastwam_config,
    _resolve_dataset_stats_path,
)
from rlinf.utils.logging import get_logger

from imago.dido import interaction as inter
from imago.ckpt import load_checkpoint_checked
from imago.dido.token_refine import TokenRefineConfig
from imago.lora import inject_lora, tail_blocks
from imago.policy import FastWAMImaginePolicy, ImagoPolicyConfig

logger = get_logger()


def _compose(config_dir: str, config_name: str, overrides) -> DictConfig:
    """Compose a FastWAM config.

    ``config_name`` may be a FastWAM config name (e.g. ``sim_libero``) or a path
    to an external YAML (e.g. ``configs/fastwam/sim_libero_optional_idm.yaml``)
    whose ``defaults`` are resolved against FastWAM's config dir.
    """
    external = Path(os.path.expanduser(config_name))
    if external.suffix == ".yaml" and external.is_file():
        source = OmegaConf.load(external)
        defaults = list(source.pop("defaults", []))
        merged = OmegaConf.create()
        for entry in defaults:
            if entry == "_self_":
                merged = OmegaConf.merge(merged, source)
                continue
            if isinstance(entry, str):
                child, _ = _load_fastwam_config(Path(config_dir), entry)
                merged = OmegaConf.merge(merged, child)
                continue
            group, choice = next(iter(dict(entry).items()))
            group = str(group).removeprefix("override ").lstrip("/")
            child, is_global = _load_fastwam_config(Path(config_dir), f"{group}/{choice}")
            if not is_global:
                child = OmegaConf.create({group: child})
            merged = OmegaConf.merge(merged, child)
        if "_self_" not in defaults:
            merged = OmegaConf.merge(merged, source)
        cfg = merged
    else:
        cfg, _ = _load_fastwam_config(Path(config_dir), config_name)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def _instantiate(model_cfg: DictConfig, torch_dtype, device, cls=FastWAMImaginePolicy) -> FastWAMImaginePolicy:
    values = OmegaConf.to_container(model_cfg, resolve=True)
    target = values.pop("_target_", "")
    if target != "fastwam.runtime.create_fastwam_optional_idm":
        raise ValueError(
            "IMAGO needs the imagine-then-act model "
            f"(fastwam.runtime.create_fastwam_optional_idm), got {target!r}."
        )
    values["mot_checkpoint_mixed_attn"] = False
    for key in ("video_dit_config", "action_dit_config"):
        if isinstance(values.get(key), dict):
            values[key]["use_gradient_checkpointing"] = False
    video_scheduler = values.pop("video_scheduler", {}) or {}
    action_scheduler = values.pop("action_scheduler")
    loss = values.pop("loss", {}) or {}
    values.pop("compile_training_denoise", None)
    return cls.from_wan22_pretrained(
        **values,
        device=device,
        torch_dtype=torch_dtype,
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


def _apply_trainable(model: FastWAMImaginePolicy, cfg: DictConfig) -> dict:
    """Freeze everything, then open each branch per ``imago_trainable.<branch>.mode``.

    ``full`` (default, accuracy-first): the whole expert, including the DIDO
    interaction tokens and heads for the video expert. ``full_tail``: the last
    N blocks. ``lora``: ablation only. ``frozen``.
    Separate learning rates use RLinf's ``model.lr_multipliers``
    (pattern -> multiplier on ``actor.optim.lr``).
    """
    lcfg = cfg.get("imago_trainable", {}) or {}
    model.requires_grad_(False)
    stats = {}
    multipliers = {}
    for branch, expert in (("video", model.video_expert), ("action", model.action_expert)):
        bcfg = lcfg.get(branch, {}) or {}
        mode = str(bcfg.get("mode", "full"))
        num_tail = int(bcfg.get("num_tail_blocks", 0))
        blocks = tail_blocks(expert.blocks, num_tail)
        if "lr_mult" in bcfg:
            multipliers[f"{branch}_expert."] = float(bcfg.lr_mult)
        if mode == "frozen":
            stats[branch] = "frozen"
        elif mode == "full":
            expert.requires_grad_(True)
            stats[branch] = "full"
        elif mode == "full_tail":
            for block in blocks:
                block.requires_grad_(True)
            stats[branch] = f"full_tail({len(blocks)} blocks)"
        elif mode == "lora":
            wrapped = inject_lora(blocks, rank=int(bcfg.get("rank", 32)), alpha=float(bcfg.get("alpha", 64)))
            for name, param in expert.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    param.requires_grad_(True)
            stats[branch] = f"lora(r={bcfg.get('rank', 32)}, {len(blocks)} blocks, {wrapped} linears)"
        else:
            raise ValueError(f"Unknown imago_trainable.{branch}.mode: {mode}")
    if bool(lcfg.get("proprio", True)) and getattr(model, "proprio_encoder", None) is not None:
        model.proprio_encoder.requires_grad_(True)
    if multipliers:
        model.lr_multipliers = multipliers  # read by RLinf's FSDP build_optimizer
    stats["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stats["lr_multipliers"] = multipliers
    return stats


def _policy_class(cfg: DictConfig):
    """``imago.policy``: ``imago`` (default) or ``flow_grpo`` (C2 baseline)."""
    name = str((cfg.get("imago", {}) or {}).get("policy", "imago"))
    if name == "imago":
        return FastWAMImaginePolicy
    if name == "flow_grpo":
        from imago.baselines.flow_grpo import FlowGRPOPolicy

        return FlowGRPOPolicy
    raise ValueError(f"Unknown imago.policy: {name!r}")


def get_model(cfg: DictConfig, torch_dtype=None):
    os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(Path.cwd() / "checkpoints"))

    fw = cfg.get("fastwam", {}) or {}
    config_dir = fw.get("config_dir", None) or _default_fastwam_config_dir()
    fcfg = _compose(config_dir, fw.get("config_name"), fw.get("overrides", None))
    # The prompt bank replaces the text encoder.
    fcfg.model.load_text_encoder = False

    torch_dtype = torch_dtype or torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _instantiate(fcfg.model, torch_dtype, device, cls=_policy_class(cfg))

    ckpt_path = cfg.get("model_path", None)
    if not ckpt_path:
        raise ValueError("IMAGO requires model.model_path (FastWAM Optional-IDM checkpoint).")
    ckpt_path = os.path.expanduser(os.path.expandvars(str(ckpt_path)))
    logger.info("Loading FastWAM Optional-IDM checkpoint: %s", ckpt_path)
    icfg_inter = (cfg.get("imago", {}) or {}).get("interaction", None)
    if icfg_inter is not None and bool(icfg_inter.get("enabled", False)):
        # DIDO interaction tokens must exist before loading a DIDO checkpoint.
        inter.attach(model.video_expert, inter.InteractionConfig(**dict(icfg_inter)))
    load_checkpoint_checked(model, ckpt_path)

    stats = _apply_trainable(model, cfg)
    # RLinf actor checkpoints of IMAGO runs (LoRA / tails) are loaded on top.
    lora_ckpt = cfg.get("imago_ckpt_path", None)
    if lora_ckpt:
        state = torch.load(os.path.expanduser(lora_ckpt), map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info("IMAGO ckpt loaded (%d unexpected keys).", len(unexpected))
    logger.info("IMAGO trainable: %s", stats)

    from hydra.utils import instantiate
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    processor = instantiate(fcfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(
        load_dataset_stats_from_json(_resolve_dataset_stats_path(cfg, ckpt_path))
    )

    num_frames = int(fcfg.data.train.num_frames)
    ratio = int(fcfg.data.train.get("action_video_freq_ratio", 1))
    icfg = cfg.get("imago", {}) or {}
    policy_cfg = ImagoPolicyConfig(
        action_horizon=int(cfg.get("action_horizon", num_frames - 1)),
        num_action_chunks=int(cfg.get("num_action_chunks", 8)),
        num_video_frames=(num_frames - 1) // ratio + 1,
        action_video_freq_ratio=ratio,
        train_video_steps=int(icfg.get("train_video_steps", 4)),
        train_action_steps=int(icfg.get("train_action_steps", 4)),
        eval_video_steps=int(icfg.get("eval_video_steps", 10)),
        eval_action_steps=int(icfg.get("eval_action_steps", 10)),
        video_sigma_shift=icfg.get("video_sigma_shift", None),
        action_sigma_shift=icfg.get("action_sigma_shift", None),
        imagination_mode=str(icfg.get("imagination_mode", "idm")),
        nft_action_executed_only=bool(icfg.get("nft_action_executed_only", True)),
        action_grad_through_video=bool(icfg.get("action_grad_through_video", False)),
        binarize_gripper=bool(cfg.get("binarize_gripper", True)),
        concat_multi_camera=str(fcfg.data.train.get("concat_multi_camera", "horizontal")),
        prompt_bank_path=icfg.get("prompt_bank_path", None),
        record_dir=icfg.get("record_dir", None),
        group_size=int(icfg.get("group_size", 8)),
        rollout_precision=str(icfg.get("rollout_precision", "bf16")),
        token_refine=TokenRefineConfig(**dict(icfg.get("token_refine", {}) or {})),
        flow_sde_noise_level=float(icfg.get("flow_sde_noise_level", 0.5)),
        realism_frames=bool(icfg.get("realism_frames", False)),
        realism_frame_scale=float(icfg.get("realism_frame_scale", 0.5)),
    )
    return model.configure_imago(processor=processor, policy_cfg=policy_cfg)
