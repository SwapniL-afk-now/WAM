"""DIDO Stage I: one-step distillation of FastWAM's video expert (DMD2 recipe).

Follows arXiv 2609.15570, App. B.1 and C.2, with the *interaction tokens
omitted* (they are an accuracy add-on needing box/DINOv3 labels; see
research_plan/dido_implementation_guide.md):

* The teacher ``f_T`` is FastWAM Optional-IDM's own video expert, which is
  already robot-domain, so DIDO's teacher-preparation phase is not needed.
  It is evaluated on its 4-step inference grid, ``u ∈ {1, .75, .5, .25}``
  mapped through FastWAM's time shift (DIDO: ``{1000, 750, 500, 250}``
  through the scheduler).
* The student ``G_θ`` generates once at timestep 1000:
  ``x_g = z - v(z, t=1)``, with frame 0 clamped to the observation.
* The fake score ``f_F`` regresses ``x0 = sg(x_g)`` at an independently
  sampled grid timestep (Eq. 20). It updates every iteration, the student
  every ``gen_update_interval`` (5) iterations.
* The student gradient is the normalised DMD direction (Eq. 19)
  ``g = (x̂0_F - x̂0_T) / mean|x_g - x̂0_T|``, injected via the surrogate
  ``0.5 * ||x_g - sg(x_g - g)||^2``. No GAN or discriminator.
* Teacher CFG: DIDO uses guidance 7. FastWAM is trained without prompt
  dropout (no unconditional branch), so ``teacher_cfg_scale`` must be 1.0.

Memory: ``G_θ`` and ``f_F`` are two LoRA adapters on *one* frozen base, and
the teacher is that base with adapters off. That is one 5B replica per GPU
(instead of three full models) and is the documented deviation from the
paper's full fine-tuning; LoRA learning rates are therefore higher than
DIDO's 1e-6 / 1e-7.

    torchrun --nproc_per_node 3 -m imago.dido.distill configs/dido/stage1_libero.yaml
"""

from __future__ import annotations

import os
import sys
import time

import torch

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
from imago.lora import (
    adapter_parameters,
    add_adapter,
    inject_lora,
    merge_adapter,
    tail_blocks,
    use_adapter,
)
from imago.nft_math import masked_sample_mean, shifted_sigma


def teacher_grid(num_steps: int, shift: float, device) -> torch.Tensor:
    """Noise levels of the teacher's ``num_steps``-step inference schedule."""
    u = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)[:-1]
    return shifted_sigma(u, shift)


def x0_from(model, x_t, sigma, first, ctx, mask):
    v = model.video_velocity(x_t, sigma, first, ctx, mask)
    x0 = x_t - sigma.view(-1, 1, 1, 1, 1).to(x_t.dtype) * v
    x0 = x0.clone()
    x0[:, :, 0:1] = first.to(x0.dtype)
    return x0


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
    blocks = tail_blocks(model.video_expert.blocks, int(cfg.lora.num_tail_blocks))
    inject_lora(blocks, rank=int(cfg.lora.rank), alpha=float(cfg.lora.alpha), with_default=False)
    add_adapter(model.video_expert, "gen", int(cfg.lora.rank), float(cfg.lora.alpha))
    add_adapter(model.video_expert, "fake", int(cfg.lora.rank), float(cfg.lora.alpha))
    gen_params = adapter_parameters(model.video_expert, "gen")
    fake_params = adapter_parameters(model.video_expert, "fake")
    for p in gen_params + fake_params:
        p.requires_grad_(True)
    fp32_(gen_params + fake_params)
    if cfg.video_expert_gradient_checkpointing:
        model.video_expert.use_gradient_checkpointing = True

    opt = cfg.optim
    opt_gen = torch.optim.AdamW(gen_params, lr=float(opt.gen_lr), betas=tuple(opt.betas),
                                weight_decay=float(opt.weight_decay))
    opt_fake = torch.optim.AdamW(fake_params, lr=float(opt.fake_lr), betas=tuple(opt.betas),
                                 weight_decay=float(opt.weight_decay))
    total = int(cfg.train.steps)
    sch_gen = torch.optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda("const", int(opt.warmup), total))
    sch_fake = torch.optim.lr_scheduler.LambdaLR(opt_fake, lr_lambda("const", int(opt.warmup), total))

    micro = int(cfg.train.micro_batch_size)
    accum = max(1, int(cfg.train.global_batch_size) // (micro * world))
    loader, sampler = build_loader(fcfg, micro, rank, world, int(cfg.seed))
    data = infinite(loader, sampler)
    shift = float(cfg.dmd.teacher_shift)
    grid = teacher_grid(int(cfg.dmd.teacher_steps), shift, device)
    interval = int(cfg.dmd.gen_update_interval)
    if rank == 0:
        print(f"[dido-s1] LoRA params gen={sum(p.numel() for p in gen_params)/1e6:.1f}M "
              f"accum={accum} world={world} micro={micro} grid={grid.tolist()}", flush=True)

    model.eval()  # no dropout; adapters still train
    start = time.time()
    for step in range(total):
        update_gen = step % interval == 0
        logs = {"fake": 0.0, "gen": 0.0}
        for _ in range(accum):
            batch = to_device(next(data), device)
            with torch.no_grad():
                inputs = model.build_inputs(batch)
            ctx, mask = inputs["context"], inputs["context_mask"]
            first = inputs["first_frame_latents"]
            lat_shape = inputs["input_latents"].shape
            elem = torch.ones((1, 1, lat_shape[2], 1, 1), device=device, dtype=torch.bool)
            elem[:, :, 0] = False
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # ---- student one-step sample (timestep 1000)
                z = torch.randn(lat_shape, device=device, dtype=torch.bfloat16)
                z[:, :, 0:1] = first
                one = torch.ones(lat_shape[0], device=device)
                with use_adapter(model.video_expert, "gen"), torch.set_grad_enabled(update_gen):
                    x_g = x0_from(model, z, one, first, ctx, mask)

                # ---- student update: normalised DMD direction (Eqs. 18-19)
                if update_gen:
                    sigma = grid[torch.randint(0, len(grid), (lat_shape[0],), device=device)]
                    with torch.no_grad():
                        x_t = perturb(x_g.detach(), sigma, first)
                        with use_adapter(model.video_expert, None):  # frozen teacher
                            x0_t = x0_from(model, x_t, sigma, first, ctx, mask)
                        with use_adapter(model.video_expert, "fake"):
                            x0_f = x0_from(model, x_t, sigma, first, ctx, mask)
                        denom = masked_sample_mean((x_g.detach() - x0_t).abs().float(), elem)
                        g = (x0_f - x0_t).float() / denom.clamp_min(1e-6).view(-1, 1, 1, 1, 1)
                        target = (x_g.detach().float() - g)
                    loss_gen = 0.5 * masked_sample_mean((x_g.float() - target) ** 2, elem).mean()
                    (loss_gen / accum).backward()
                    logs["gen"] += loss_gen.item() / accum

                # ---- fake-score update: x0 regression on student samples (Eq. 20)
                sigma_f = grid[torch.randint(0, len(grid), (lat_shape[0],), device=device)]
                x_t_f = perturb(x_g.detach(), sigma_f, first)
                with use_adapter(model.video_expert, "fake"):
                    x0_pred = x0_from(model, x_t_f, sigma_f, first, ctx, mask)
                loss_fake = masked_sample_mean((x0_pred.float() - x_g.detach().float()) ** 2, elem).mean()
            (loss_fake / accum).backward()
            logs["fake"] += loss_fake.item() / accum

        allreduce_grads(fake_params, world)
        torch.nn.utils.clip_grad_norm_(fake_params, float(opt.clip_grad))
        opt_fake.step(); sch_fake.step(); opt_fake.zero_grad(set_to_none=True)
        if update_gen:
            allreduce_grads(gen_params, world)
            torch.nn.utils.clip_grad_norm_(gen_params, float(opt.clip_grad))
            opt_gen.step(); opt_gen.zero_grad(set_to_none=True)
        sch_gen.step()  # step-based schedule for both

        if rank == 0 and step % int(cfg.train.log_every) == 0:
            print(f"[dido-s1] step {step}/{total} fake {logs['fake']:.4f} gen {logs['gen']:.4f} "
                  f"{(time.time() - start) / (step + 1):.2f}s/it", flush=True)
        if rank == 0 and (step + 1) % int(cfg.train.save_every) == 0:
            os.makedirs(cfg.output_dir, exist_ok=True)
            torch.save({"gen": [p.detach().cpu() for p in gen_params],
                        "fake": [p.detach().cpu() for p in fake_params], "step": step + 1},
                       os.path.join(cfg.output_dir, f"adapters_step{step + 1}.pt"))

    if rank == 0:
        merge_adapter(model.video_expert, "gen")
        out = os.path.join(cfg.output_dir, "fastwam_optional_idm_onestep_video.pt")
        export_fastwam_checkpoint(model, out, {"dido_stage": 1, "video_steps": 1})
        print(f"[dido-s1] exported one-step video expert -> {out}", flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
