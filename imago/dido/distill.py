"""DIDO Stage I, full parameters: one-step distillation of FastWAM's video expert.

arXiv 2609.15570, Sec. 3.1–3.2, App. B.1 and C.2:

    L_stage1 = L_DMD + L_inter + λ_align · L_align

* **Three separate video experts** (DIDO: all initialised from the teacher):
  - teacher ``f_T``: the frozen FastWAM Optional-IDM video expert;
  - generator ``G_θ``: a full-parameter copy that also carries the interaction tokens;
  - fake score ``f_F``: a full-parameter copy.

  Each is its own FSDP2 unit; see ``imago/fsdp.py``.
* ``G_θ`` generates once at timestep 1000: ``x_g = z - v(z, t=1)``, with frame 0
  clamped to the observation.
* Normalised DMD direction (Eq. 19) ``g = (x̂0_F - x̂0_T) / mean|x_g - x̂0_T|``,
  applied through the surrogate ``0.5·||x_g - sg(x_g - g)||²``. No GAN.
* The fake score regresses ``sg(x_g)`` at an independent noise level (Eq. 20).
  It steps every iteration; ``G_θ`` steps every ``gen_update_interval`` (5).
* **Score noise levels.** DIDO samples its 4-step teacher's grid {1000, 750,
  500, 250} because that teacher is itself 4-step distilled. FastWAM is a full
  continuous-time model, so the default is continuous sampling from FastWAM's
  training-time distribution clipped to [0.02, 0.98] (the DMD2 range).
  ``teacher_timesteps: grid`` reproduces the paper.
* **Teacher guidance:** 1.0. FastWAM is trained without prompt dropout, so it has
  no unconditional branch (DIDO uses 7).
* **Interaction supervision** (``imago/dido/interaction.py``): box losses and
  DINOv3 alignment on ``G_θ``'s one-step pass, with the Table 4 Stage I weights.

Memory (bf16 compute, fp32 master and optimizer states):
* 3 GPUs: about 57 GB/GPU persistent, plus one gathered 10 GB model at a time.
* 2 GPUs: the generator is CPU-offloaded (it steps every 5th iteration).

    torchrun --nproc_per_node 3 -m imago.dido.distill configs/dido/stage1_libero.yaml
"""

from __future__ import annotations

import copy
import os
import sys
import time

import torch

from imago.dido import interaction as inter
from imago.dido.common import (
    build_loader,
    export_full_fastwam,
    infinite,
    init_distributed,
    load_fastwam,
    load_yaml,
    lr_lambda,
    to_device,
)
from imago.dido.video_fn import video_forward, x0_from_velocity
from imago.fsdp import num_params, report_memory, shard
from imago.nft_math import masked_sample_mean, shifted_sigma


def sample_sigma(kind: str, bsz: int, device, *, steps: int, shift: float, scheduler, lo=0.02, hi=0.98):
    if kind == "grid":
        u = torch.linspace(1.0, 0.0, steps + 1, device=device)[:-1]
        grid = shifted_sigma(u, shift)
        return grid[torch.randint(0, len(grid), (bsz,), device=device)]
    t = scheduler.sample_training_t(bsz, device, torch.float32) / scheduler.num_train_timesteps
    return t.clamp(lo, hi)


def perturb(x0, sigma, first):
    s = sigma.view(-1, 1, 1, 1, 1).to(x0.dtype)
    x_t = (1 - s) * x0 + s * torch.randn_like(x0)
    x_t[:, :, 0:1] = first.to(x_t.dtype)
    return x_t


def main() -> None:
    cfg = load_yaml(sys.argv[1], sys.argv[2:])
    rank, world, device = init_distributed()
    torch.manual_seed(int(cfg.seed) + rank)
    if float(cfg.dmd.teacher_cfg_scale) != 1.0:
        raise ValueError("FastWAM has no unconditional branch; teacher_cfg_scale must be 1.0.")

    model, fcfg = load_fastwam(cfg, device)
    model.action_expert.to("cpu")  # unused in Stage I; exported unchanged
    teacher = copy.deepcopy(model.video_expert).requires_grad_(False)
    fake = copy.deepcopy(model.video_expert).requires_grad_(True)
    gen = model.video_expert
    icfg = inter.InteractionConfig(**dict(cfg.interaction))
    tokens = inter.attach(gen, icfg) if icfg.enabled else None
    gen.requires_grad_(True)
    if cfg.gradient_checkpointing:
        gen.use_gradient_checkpointing = True
        fake.use_gradient_checkpointing = True
    mot_keys = list(model.mot.state_dict().keys())
    frozen_state = {k: v.detach().cpu() for k, v in model.mot.state_dict().items()
                    if not k.startswith("mixtures.video.")}

    offload_gen = world == 2 and bool(cfg.fsdp.offload_generator_when_2gpu)
    report_memory(
        {"teacher": (num_params(teacher), False, False),
         "generator": (num_params(gen), True, offload_gen),
         "fake_score": (num_params(fake), True, False)},
        world, rank,
    )
    t_run = shard(teacher, trainable=False)
    g_run = shard(gen, trainable=True, offload=offload_gen)
    f_run = shard(fake, trainable=True)

    opt = cfg.optim
    opt_gen = torch.optim.AdamW(g_run.parameters(), lr=float(opt.gen_lr), betas=tuple(opt.betas),
                                weight_decay=float(opt.weight_decay))
    opt_fake = torch.optim.AdamW(f_run.parameters(), lr=float(opt.fake_lr), betas=tuple(opt.betas),
                                 weight_decay=float(opt.weight_decay))
    total = int(cfg.train.steps)
    sch_gen = torch.optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda("const", int(opt.warmup), total))
    sch_fake = torch.optim.lr_scheduler.LambdaLR(opt_fake, lr_lambda("const", int(opt.warmup), total))

    micro = int(cfg.train.micro_batch_size)
    accum = max(1, int(cfg.train.global_batch_size) // (micro * world))
    loader, sampler = build_loader(fcfg, micro, rank, world, int(cfg.seed))
    data = infinite(loader, sampler)
    interval = int(cfg.dmd.gen_update_interval)
    kind = str(cfg.dmd.teacher_timesteps)
    steps, shift = int(cfg.dmd.teacher_steps), float(cfg.dmd.teacher_shift)
    sched = model.train_video_scheduler
    align_layers = tuple(tokens.align_layers) if tokens is not None else ()
    fwd = lambda run, *a, **k: run(video_forward, *a, **k)  # noqa: E731

    t0 = time.time()
    for step in range(total):
        update_gen = step % interval == 0
        logs = {"fake": 0.0, "dmd": 0.0, "inter": 0.0, "align": 0.0}
        for _ in range(accum):
            batch = to_device(next(data), device)
            with torch.no_grad():
                inputs = model.build_inputs(batch)
            ctx, mask = inputs["context"], inputs["context_mask"]
            first = inputs["first_frame_latents"]
            shape = inputs["input_latents"].shape
            elem = torch.ones((1, 1, shape[2], 1, 1), device=device, dtype=torch.bool)
            elem[:, :, 0] = False
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = torch.randn(shape, device=device, dtype=torch.bfloat16)
                z[:, :, 0:1] = first
                one = torch.ones(shape[0], device=device)
                with torch.set_grad_enabled(update_gen):
                    v_g, tok_out, hidden = fwd(g_run, z, one, first, ctx, mask, collect_layers=align_layers)
                    x_g = x0_from_velocity(z, v_g, one, first)

                if update_gen:
                    sigma = sample_sigma(kind, shape[0], device, steps=steps, shift=shift, scheduler=sched)
                    with torch.no_grad():
                        x_t = perturb(x_g.detach(), sigma, first)
                        x0_t = x0_from_velocity(x_t, fwd(t_run, x_t, sigma, first, ctx, mask)[0], sigma, first)
                        x0_f = x0_from_velocity(x_t, fwd(f_run, x_t, sigma, first, ctx, mask)[0], sigma, first)
                        denom = masked_sample_mean((x_g.detach() - x0_t).abs().float(), elem)
                        g = (x0_f - x0_t).float() / denom.clamp_min(1e-6).view(-1, 1, 1, 1, 1)
                        target = x_g.detach().float() - g
                    loss = 0.5 * masked_sample_mean((x_g.float() - target) ** 2, elem).mean()
                    logs["dmd"] += loss.item() / accum
                    if tokens is not None:
                        l_inter, _ = tokens.interaction_loss(
                            tok_out, batch["boxes_obj"], batch["boxes_grip"],
                            batch["boxes_obj_valid"], batch["boxes_grip_valid"],
                        )
                        l_align = tokens.align_loss(hidden, batch["dino_target"], batch["dino_valid"])
                        loss = loss + l_inter + icfg.lambda_align * l_align
                        logs["inter"] += l_inter.item() / accum
                        logs["align"] += l_align.item() / accum
                    (loss / accum).backward()

                sigma_f = sample_sigma(kind, shape[0], device, steps=steps, shift=shift, scheduler=sched)
                x_t_f = perturb(x_g.detach(), sigma_f, first)
                x0_pred = x0_from_velocity(x_t_f, fwd(f_run, x_t_f, sigma_f, first, ctx, mask)[0], sigma_f, first)
                loss_fake = masked_sample_mean((x0_pred.float() - x_g.detach().float()) ** 2, elem).mean()
            (loss_fake / accum).backward()
            logs["fake"] += loss_fake.item() / accum

        torch.nn.utils.clip_grad_norm_(f_run.parameters(), float(opt.clip_grad))
        opt_fake.step(); sch_fake.step(); opt_fake.zero_grad(set_to_none=True)
        if update_gen:
            torch.nn.utils.clip_grad_norm_(g_run.parameters(), float(opt.clip_grad))
            opt_gen.step(); opt_gen.zero_grad(set_to_none=True)
        sch_gen.step()

        if rank == 0 and step % int(cfg.train.log_every) == 0:
            print(f"[dido-s1] step {step}/{total} " + " ".join(f"{k} {v:.4f}" for k, v in logs.items())
                  + f" {(time.time() - t0) / (step + 1):.2f}s/it", flush=True)
        if (step + 1) % int(cfg.train.save_every) == 0 or step + 1 == total:
            name = "fastwam_optional_idm_onestep_video.pt" if step + 1 == total else f"stage1_step{step + 1}.pt"
            export_full_fastwam(model, g_run, mot_keys, frozen_state, os.path.join(cfg.output_dir, name),
                                {"dido_stage": 1, "video_steps": 1, "step": step + 1,
                                 "interaction": dict(cfg.interaction)}, rank)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
