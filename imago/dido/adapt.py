"""DIDO Stage II, full parameters: joint policy learning on one-step imagination.

arXiv 2609.15570, Sec. 3.3–3.4, App. C.2 (Eq. 10):

    L = λ_video·L_video + λ_act·L_act + L_inter + λ_align·L_align
      (λ_video 0.5, λ_act 1.0, boxes 0.03 / 0.03 / GIoU 1.0, λ_align 0.02)

* Trainable, as in DIDO: the video transformer, interaction tokens and heads,
  action expert and proprio encoder. The VAE and text are frozen, and DINOv3
  is only used offline. Everything trainable is one FSDP2 unit (``imago/fsdp.py``).
* ``L_video``: Fast-WAM video flow matching (Eq. 22) on ground-truth future
  latents, with frame 0 clean and excluded.
* The one-step generator pass ``x_g = z - v(z, t=1)`` is the inference path.
  Its interaction-token states give ``L_inter`` / ``L_align``, so the tokens
  keep being supervised where they are used.
* ``L_act``: FastWAM action flow matching (Eq. 8). The action expert reads
  the K/V cache of ``sg(x_g)`` plus the interaction tokens, the whole
  world-model stream as in DIDO, after dynamics-based token refinement.
  Gradients flow from ``L_act`` into the video expert through the cache
  (DIDO's MoT is trained jointly); only the sampled latent is detached.
* Paper schedule: lr 1e-4, betas (0.9, 0.95), 5% warm-up then cosine,
  14,480 steps at global batch 384 (LIBERO).

    torchrun --nproc_per_node 3 -m imago.dido.adapt configs/dido/stage2_libero.yaml
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from imago.dido import interaction as inter
from imago.dido.common import (
    build_loader,
    full_state,
    infinite,
    init_distributed,
    load_fastwam,
    load_yaml,
    lr_lambda,
    to_device,
)
from imago.dido.video_fn import x0_from_velocity
from imago.fsdp import num_params, report_memory, shard


def video_loss(model, inputs):
    x0 = inputs["input_latents"]
    first = inputs["first_frame_latents"]
    sched = model.train_video_scheduler
    timestep = sched.sample_training_t(x0.shape[0], x0.device, x0.dtype)
    noise = torch.randn_like(x0)
    x_t = sched.add_noise(x0, noise, timestep)
    target = sched.training_target(x0, noise, timestep)
    sigma = timestep.float() / sched.num_train_timesteps
    pred, _tok, _hid = model.video_forward_full(x_t, sigma, first, inputs["context"], inputs["context_mask"])
    per_sample = model._compute_video_loss_per_sample(
        pred_video=pred[:, :, 1:], target_video=target[:, :, 1:],
        image_is_pad=inputs["image_is_pad"], include_initial_video_step=False,
    )
    return (per_sample * sched.training_weight(timestep).to(per_sample)).mean()


def action_loss(model, inputs, video_cond):
    action = inputs["action"]
    cache = model.build_video_cache(video_cond, inputs["context"], inputs["context_mask"], action.shape[1])
    sched = model.train_action_scheduler
    timestep = sched.sample_training_t(action.shape[0], action.device, action.dtype)
    noise = torch.randn_like(action)
    x_t = sched.add_noise(action, noise, timestep)
    target = sched.training_target(action, noise, timestep)
    sigma = timestep.float() / sched.num_train_timesteps
    pred = model.action_velocity(x_t, sigma, inputs["context"], inputs["context_mask"], cache)
    tok = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=2)
    pad = inputs.get("action_is_pad")
    if pad is not None:
        valid = (~pad).float()
        per_sample = (tok * valid).sum(1) / valid.sum(1).clamp_min(1.0)
    else:
        per_sample = tok.mean(1)
    return (per_sample * sched.training_weight(timestep).to(per_sample)).mean()


def train_step(_unit, model, batch, cfg, device):
    """Runs inside the FSDP2 Runner, so all trainable parameters are gathered."""
    inputs = model.build_inputs(batch)  # proprio token (trainable) appended to context
    first = inputs["first_frame_latents"]
    lat = inputs["input_latents"]
    logs = {}
    loss = torch.zeros((), device=device)
    lam = cfg.loss
    if float(lam.lambda_video) > 0:
        lv = video_loss(model, inputs)
        loss = loss + float(lam.lambda_video) * lv
        logs["video"] = lv.item()

    # One-step imagination (the inference path) + interaction supervision on it.
    z = torch.randn_like(lat)
    z[:, :, 0:1] = first
    one = torch.ones(lat.shape[0], device=device)
    tokens = inter.get(model.video_expert)
    collect = tuple(tokens.align_layers) if tokens is not None else ()
    v_g, tok_out, hidden = model.video_forward_full(z, one, first, inputs["context"], inputs["context_mask"],
                                                    collect_layers=collect)
    x_g = x0_from_velocity(z, v_g, one, first)
    if tokens is not None:
        stage2 = inter.InteractionConfig(**dict(cfg.interaction))
        l_inter, _ = tokens.interaction_loss(tok_out, batch["boxes_obj"], batch["boxes_grip"],
                                             batch["boxes_obj_valid"], batch["boxes_grip_valid"], lam=stage2)
        l_align = tokens.align_loss(hidden, batch["dino_target"], batch["dino_valid"])
        loss = loss + l_inter + stage2.lambda_align * l_align
        logs["inter"], logs["align"] = l_inter.item(), l_align.item()

    cond = x_g.detach()
    p_gt = float(lam.p_gt_video)
    if p_gt > 0:
        use_gt = (torch.rand(cond.shape[0], device=device) < p_gt).view(-1, 1, 1, 1, 1)
        cond = torch.where(use_gt, lat.to(cond.dtype), cond)
    la = action_loss(model, inputs, cond)
    loss = loss + float(lam.lambda_act) * la
    logs["act"] = la.item()
    return loss, logs


def main() -> None:
    cfg = load_yaml(sys.argv[1], sys.argv[2:])
    rank, world, device = init_distributed()
    torch.manual_seed(int(cfg.seed) + rank)
    model, fcfg = load_fastwam(cfg, device, attach_interaction=bool(cfg.interaction.enabled))
    unit = nn.ModuleDict({"mot": model.mot})
    if getattr(model, "proprio_encoder", None) is not None:
        unit["proprio"] = model.proprio_encoder
    unit.requires_grad_(True)
    if cfg.gradient_checkpointing:
        model.video_expert.use_gradient_checkpointing = True
    report_memory({"mot+proprio": (num_params(unit), True, world == 2 and bool(cfg.fsdp.offload_when_2gpu))},
                  world, rank)
    runner = shard(unit, trainable=True, offload=world == 2 and bool(cfg.fsdp.offload_when_2gpu))

    opt = cfg.optim
    optim = torch.optim.AdamW(runner.parameters(), lr=float(opt.lr), betas=tuple(opt.betas),
                              weight_decay=float(opt.weight_decay))
    total = int(cfg.train.steps)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda("cosine", int(float(opt.warmup_frac) * total), total))
    micro = int(cfg.train.micro_batch_size)
    accum = max(1, int(cfg.train.global_batch_size) // (micro * world))
    loader, sampler = build_loader(fcfg, micro, rank, world, int(cfg.seed))
    data = infinite(loader, sampler)

    t0 = time.time()
    for step in range(total):
        agg: dict = {}
        for _ in range(accum):
            batch = to_device(next(data), device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, logs = runner(train_step, model, batch, cfg, device)
            (loss / accum).backward()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v / accum
        torch.nn.utils.clip_grad_norm_(runner.parameters(), float(opt.clip_grad))
        optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
        if rank == 0 and step % int(cfg.train.log_every) == 0:
            print(f"[dido-s2] step {step}/{total} " + " ".join(f"{k} {v:.4f}" for k, v in agg.items())
                  + f" lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / (step + 1):.2f}s/it", flush=True)
        if (step + 1) % int(cfg.train.save_every) == 0 or step + 1 == total:
            name = "fastwam_optional_idm_dido.pt" if step + 1 == total else f"stage2_step{step + 1}.pt"
            state = full_state(runner)  # collective
            if rank == 0:
                payload = {
                    "mot": {k[len("mot."):]: v.to(torch.bfloat16) for k, v in state.items() if k.startswith("mot.")},
                    "imago_meta": {"dido_stage": 2, "video_steps": 1, "step": step + 1,
                                   "token_refine": dict(cfg.get("token_refine", {}) or {}),
                                   "interaction": dict(cfg.interaction)},
                }
                proprio = {k[len("proprio."):]: v.to(torch.bfloat16) for k, v in state.items() if k.startswith("proprio.")}
                if proprio:
                    payload["proprio_encoder"] = proprio
                Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
                torch.save(payload, os.path.join(cfg.output_dir, name))
                print(f"[dido-s2] exported {name}", flush=True)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
