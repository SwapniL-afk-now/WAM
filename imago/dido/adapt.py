"""DIDO Stage II: adapt the action expert to one-step imagination.

Follows arXiv 2609.15570, Sec. 3.4 and App. C.2 (Eq. 10 without the
interaction terms):

    L = λ_video · L_video + λ_act · L_act          (λ_video = 0.5, λ_act = 1.0)

* ``L_video``: Fast-WAM video flow matching (Eq. 22) on ground-truth future
  latents. Frame 0 is clean and excluded; FastWAM's time weighting is used.
  It keeps the video expert a good predictor while the policy adapts.
* ``L_act``: FastWAM action flow matching (Eq. 8). The action expert reads
  the video K/V cache of the **one-step imagined** future ``x_g = G(z)``,
  exactly as at inference, with probability ``1 - p_gt_video``, and the
  ground-truth future otherwise (FastWAM IDM teacher forcing). DIDO token
  refinement is applied when ``token_refine.enabled``.
* DIDO trains the whole backbone at lr 1e-4. Here the action expert and the
  proprio encoder train fully (about 1B params, replicated), and the video
  expert through a LoRA adapter merged at export. Same schedule: betas
  (0.9, 0.95), 5% warm-up, cosine decay.

Output: a FastWAM-format checkpoint for ``imago`` with
``imago.train_video_steps = eval_video_steps = 1``.

    torchrun --nproc_per_node 3 -m imago.dido.adapt configs/dido/stage2_libero.yaml
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

from imago.dido.common import (
    allreduce_grads,
    build_loader,
    export_fastwam_checkpoint,
    fp32_,
    infinite,
    init_distributed,
    load_fastwam,
    load_yaml,
    lr_lambda,
    to_device,
)
from imago.lora import adapter_parameters, inject_lora, merge_adapter, tail_blocks


def video_loss(model, inputs):
    """Fast-WAM video flow matching with a clean first frame (Eq. 22)."""
    x0 = inputs["input_latents"]
    first = inputs["first_frame_latents"]
    bsz = x0.shape[0]
    timestep = model.train_video_scheduler.sample_training_t(bsz, x0.device, x0.dtype)
    noise = torch.randn_like(x0)
    x_t = model.train_video_scheduler.add_noise(x0, noise, timestep)
    target = model.train_video_scheduler.training_target(x0, noise, timestep)
    sigma = timestep.float() / model.train_video_scheduler.num_train_timesteps
    pred = model.video_velocity(x_t, sigma, first, inputs["context"], inputs["context_mask"])
    per_sample = model._compute_video_loss_per_sample(
        pred_video=pred[:, :, 1:],
        target_video=target[:, :, 1:],
        image_is_pad=inputs["image_is_pad"],
        include_initial_video_step=False,
    )
    weight = model.train_video_scheduler.training_weight(timestep).to(per_sample)
    return (per_sample * weight).mean()


def action_loss(model, inputs, video_cond):
    """FastWAM action flow matching conditioned on a (clean) video latent."""
    action = inputs["action"]
    bsz = action.shape[0]
    cache = model.build_video_cache(
        video_cond, inputs["context"], inputs["context_mask"], action.shape[1]
    )
    timestep = model.train_action_scheduler.sample_training_t(bsz, action.device, action.dtype)
    noise = torch.randn_like(action)
    x_t = model.train_action_scheduler.add_noise(action, noise, timestep)
    target = model.train_action_scheduler.training_target(action, noise, timestep)
    sigma = timestep.float() / model.train_action_scheduler.num_train_timesteps
    pred = model.action_velocity(x_t, sigma, inputs["context"], inputs["context_mask"], cache)
    tok = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=2)
    pad = inputs.get("action_is_pad")
    if pad is not None:
        valid = (~pad).float()
        per_sample = (tok * valid).sum(1) / valid.sum(1).clamp_min(1.0)
    else:
        per_sample = tok.mean(1)
    weight = model.train_action_scheduler.training_weight(timestep).to(per_sample)
    return (per_sample * weight).mean()


@torch.no_grad()
def one_step_video(model, inputs):
    """``x_g = z - v(z, t=1)``: the distilled one-step imagination."""
    lat = inputs["input_latents"]
    first = inputs["first_frame_latents"]
    z = torch.randn_like(lat)
    z[:, :, 0:1] = first
    one = torch.ones(lat.shape[0], device=lat.device)
    x_g = z - model.video_velocity(z, one, first, inputs["context"], inputs["context_mask"])
    x_g[:, :, 0:1] = first
    return x_g


def main() -> None:
    cfg = load_yaml(sys.argv[1], sys.argv[2:])
    rank, world, device = init_distributed()
    torch.manual_seed(int(cfg.seed) + rank)
    model, fcfg = load_fastwam(cfg, device)  # model_path = Stage I export

    # Trainables: full action expert + proprio encoder; video expert via LoRA.
    trainable = list(model.action_expert.parameters())
    if getattr(model, "proprio_encoder", None) is not None:
        trainable += list(model.proprio_encoder.parameters())
    for p in trainable:
        p.requires_grad_(True)
    fp32_(trainable)
    video_lora = []
    if int(cfg.video_lora.rank) > 0:
        blocks = tail_blocks(model.video_expert.blocks, int(cfg.video_lora.num_tail_blocks))
        inject_lora(blocks, rank=int(cfg.video_lora.rank), alpha=float(cfg.video_lora.alpha))
        video_lora = adapter_parameters(model.video_expert, "default")
        for p in video_lora:
            p.requires_grad_(True)
        fp32_(video_lora)
    params = trainable + video_lora
    if cfg.gradient_checkpointing:
        model.video_expert.use_gradient_checkpointing = True
        model.action_expert.use_gradient_checkpointing = True

    opt = cfg.optim
    optim = torch.optim.AdamW(params, lr=float(opt.lr), betas=tuple(opt.betas),
                              weight_decay=float(opt.weight_decay))
    total = int(cfg.train.steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda("cosine", int(float(opt.warmup_frac) * total), total)
    )
    micro = int(cfg.train.micro_batch_size)
    accum = max(1, int(cfg.train.global_batch_size) // (micro * world))
    loader, sampler = build_loader(fcfg, micro, rank, world, int(cfg.seed))
    data = infinite(loader, sampler)
    lam_v, lam_a = float(cfg.loss.lambda_video), float(cfg.loss.lambda_act)
    p_gt = float(cfg.loss.p_gt_video)
    if rank == 0:
        print(f"[dido-s2] trainable {sum(p.numel() for p in params)/1e6:.0f}M accum={accum} "
              f"refine={model.policy_cfg.token_refine.enabled}", flush=True)

    start = time.time()
    for step in range(total):
        logs = {"video": 0.0, "act": 0.0}
        for _ in range(accum):
            batch = to_device(next(data), device)
            # With grad: build_inputs appends the (trainable) proprio token to the
            # context. The VAE / text parts are frozen, so no graph is kept for them.
            inputs = model.build_inputs(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lv = video_loss(model, inputs) if lam_v > 0 else torch.zeros((), device=device)
                cond = one_step_video(model, inputs)
                if p_gt > 0:  # teacher forcing on a random subset
                    use_gt = (torch.rand(cond.shape[0], device=device) < p_gt).view(-1, 1, 1, 1, 1)
                    cond = torch.where(use_gt, inputs["input_latents"].to(cond.dtype), cond)
                la = action_loss(model, inputs, cond)
                loss = lam_v * lv + lam_a * la
            (loss / accum).backward()
            logs["video"] += lv.item() / accum
            logs["act"] += la.item() / accum
        allreduce_grads(params, world)
        torch.nn.utils.clip_grad_norm_(params, float(opt.clip_grad))
        optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
        if rank == 0 and step % int(cfg.train.log_every) == 0:
            print(f"[dido-s2] step {step}/{total} L_video {logs['video']:.4f} L_act {logs['act']:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - start) / (step + 1):.2f}s/it", flush=True)
        if rank == 0 and (step + 1) % int(cfg.train.save_every) == 0:
            os.makedirs(cfg.output_dir, exist_ok=True)
            torch.save({"step": step + 1, "params": [p.detach().cpu() for p in params]},
                       os.path.join(cfg.output_dir, f"stage2_step{step + 1}.pt"))

    if rank == 0:
        if video_lora:
            merge_adapter(model.video_expert, "default")
        for p in trainable:
            p.data = p.data.to(torch.bfloat16)
        out = os.path.join(cfg.output_dir, "fastwam_optional_idm_dido.pt")
        export_fastwam_checkpoint(
            model, out,
            {"dido_stage": 2, "video_steps": 1, "token_refine": dict(cfg.get("token_refine", {}) or {})},
        )
        print(f"[dido-s2] exported -> {out}", flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
