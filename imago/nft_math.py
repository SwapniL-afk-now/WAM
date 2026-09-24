"""Pure-torch math for joint-branch Diffusion-NFT (no RLinf / FastWAM imports).

Conventions follow FastWAM's flow matching and RLinf's ``EmbodiedNFTFSDPPolicy``:

* ``x_t = (1 - t) * x0 + t * noise`` with ``t`` in [0, 1] (``sigma`` in FastWAM),
* velocity ``v = noise - x0`` so ``x0_pred = x_t - t * v``,
* the model timestep input is ``t * num_train_timesteps``.

With ODE rollouts (noise level 0) and ``nft_target_space='x0'`` RLinf's NFT
prediction reduces to ``x0_pred``; this module implements exactly that case,
with the ``mse`` loss form:

    v_pos = v_old + beta * (v_theta - v_old),   v_neg = v_old - beta * (v_theta - v_old)
    E_pos = w * ||x0_pred(v_pos) - x0||^2,       E_neg = w * ||x0_pred(v_neg) - x0||^2
    L     = (r * E_pos + (1 - r) * E_neg) * adv_clip_max / beta

where ``r`` in [0, 1] is the rescaled group-relative advantage.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def shifted_sigma(u: torch.Tensor, shift: float) -> torch.Tensor:
    """FastWAM / Wan time shift ``phi(u) = s u / (1 + (s - 1) u)``."""
    shift = float(shift)
    return shift * u / (1.0 + (shift - 1.0) * u)


def sample_nft_timesteps(
    batch_size: int,
    num_steps: int,
    shift: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a denoising-grid index per sample and return ``(index, t)``.

    ``t`` lives on the same shifted grid that the rollout ODE used, so the NFT
    energy is evaluated where the policy actually acts.
    """
    index = torch.randint(0, num_steps, (batch_size,), generator=generator, device="cpu")
    index = index.to(device)
    u = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)
    sigma = shifted_sigma(u, shift)
    return index, sigma[index].to(dtype)


def noise_to(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    t_bc = t.view(-1, *([1] * (x0.ndim - 1))).to(x0.dtype)
    return (1.0 - t_bc) * x0 + t_bc * noise


def x0_from_velocity(x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t_bc = t.view(-1, *([1] * (x_t.ndim - 1))).to(x_t.dtype)
    return x_t - t_bc * v


def grpo_to_unit(advantages: torch.Tensor, adv_clip_max: float) -> torch.Tensor:
    """Clip GRPO advantages to ``[-m, m]`` and map to ``[0, 1]`` (RLinf NFT)."""
    m = float(adv_clip_max)
    return (advantages.clamp(-m, m) + m) / (2.0 * m)


def masked_sample_mean(x: torch.Tensor, elem_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean over all non-batch dims, optionally restricted by a broadcastable mask."""
    dims = tuple(range(1, x.ndim))
    if elem_mask is None:
        return x.mean(dim=dims)
    mask = elem_mask.to(dtype=x.dtype).expand_as(x)
    return (x * mask).sum(dim=dims) / mask.sum(dim=dims).clamp_min(1.0)


@dataclass
class NFTTerms:
    loss: torch.Tensor  # scalar
    e_pos: torch.Tensor  # [B]
    e_neg: torch.Tensor  # [B]
    delta_v_norm: torch.Tensor  # [B]
    clip_frac: float


def nft_branch_loss(
    *,
    v_theta: torch.Tensor,
    v_old: torch.Tensor,
    x_t: torch.Tensor,
    x0: torch.Tensor,
    t: torch.Tensor,
    reward01: torch.Tensor,
    sample_mask: torch.Tensor,
    beta: float = 1.0,
    adv_clip_max: float = 1.0,
    weight_mode: str = "adaptive",
    weight_scale: tuple[float, float] = (1.0, 1.0),
    clip_ratio: float | None = None,
    elem_mask: torch.Tensor | None = None,
    time_weight: torch.Tensor | None = None,
) -> NFTTerms:
    """NFT ``mse`` objective for one branch (video or action).

    Args:
        v_theta: current-policy velocity at ``(x_t, t)`` (requires grad).
        v_old: reference (rollout / EMA) velocity at ``(x_t, t)`` (no grad).
        x_t, x0, t: noised input, rollout sample and per-sample noise level.
        reward01: per-sample advantage mapped into [0, 1].
        sample_mask: per-sample validity (e.g. filtered groups, done envs).
        elem_mask: broadcastable element mask (e.g. drop the clean first frame).
        time_weight: optional per-sample multiplier (see ``imago.efficiency``).
    """
    v_old = v_old.detach()
    delta_v = v_theta - v_old
    v_pos = v_old + beta * delta_v
    v_neg = v_old - beta * delta_v
    pred_pos = x0_from_velocity(x_t, v_pos, t)
    pred_neg = x0_from_velocity(x_t, v_neg, t)
    target = x0.detach()

    def weight(pred: torch.Tensor, scale: float) -> torch.Tensor:
        if weight_mode == "constant":
            return torch.full_like(t, scale, dtype=torch.float32)
        if weight_mode == "adaptive":
            with torch.no_grad():
                err = masked_sample_mean((pred.float() - target.float()).abs(), elem_mask)
            return scale / err.clamp_min(1e-5)
        raise ValueError(f"Unsupported weight_mode: {weight_mode}")

    se_pos = (pred_pos.float() - target.float()) ** 2
    se_neg = (pred_neg.float() - target.float()) ** 2
    e_pos = masked_sample_mean(se_pos, elem_mask) * weight(pred_pos, weight_scale[0])
    e_neg = masked_sample_mean(se_neg, elem_mask) * weight(pred_neg, weight_scale[1])

    r = reward01.float()
    loss_scale = float(adv_clip_max) / float(beta)
    loss_elem = (r * e_pos + (1.0 - r) * e_neg) * loss_scale

    clip_frac = 0.0
    if clip_ratio is not None:
        # Trust region on ||delta_v|| relative to ||v_old|| (RLinf nft_clip_ratio).
        dims = tuple(range(1, delta_v.ndim))
        delta_norm = torch.linalg.vector_norm(delta_v.float(), dim=dims).clamp_min(1e-8)
        old_norm = torch.linalg.vector_norm(v_old.float(), dim=dims).clamp_min(1e-8)
        coef = (float(clip_ratio) * old_norm / delta_norm).clamp(max=1.0)
        clip_frac = float((coef < 1.0).float().mean().item())
        coef_bc = coef.view(-1, *([1] * (delta_v.ndim - 1))).to(delta_v.dtype)
        delta_clip = torch.where(coef_bc < 1.0, (delta_v * coef_bc).detach(), delta_v)
        pos_c = x0_from_velocity(x_t, v_old + beta * delta_clip, t)
        neg_c = x0_from_velocity(x_t, v_old - beta * delta_clip, t)
        e_pos_c = masked_sample_mean((pos_c.float() - target.float()) ** 2, elem_mask) * weight(
            pos_c, weight_scale[0]
        )
        e_neg_c = masked_sample_mean((neg_c.float() - target.float()) ** 2, elem_mask) * weight(
            neg_c, weight_scale[1]
        )
        loss_elem = torch.maximum(loss_elem, (r * e_pos_c + (1.0 - r) * e_neg_c) * loss_scale)

    if time_weight is not None:
        loss_elem = loss_elem * time_weight.float().detach()
    mask = sample_mask.float()
    loss = (loss_elem * mask).sum() / mask.sum().clamp_min(1.0)
    with torch.no_grad():
        dims = tuple(range(1, delta_v.ndim))
        delta_v_norm = torch.linalg.vector_norm(delta_v.float(), dim=dims)
    return NFTTerms(loss, e_pos.detach(), e_neg.detach(), delta_v_norm, clip_frac)


def realism_anchor(
    v_theta: torch.Tensor,
    v_pretrained: torch.Tensor,
    sample_mask: torch.Tensor,
    elem_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``||v_theta - v_pretrained||^2``: how far imagination may drift from the
    pretrained (reality-trained) video predictor. Weighted by ``beta_real``."""
    se = (v_theta.float() - v_pretrained.detach().float()) ** 2
    per_sample = masked_sample_mean(se, elem_mask)
    mask = sample_mask.float()
    return (per_sample * mask).sum() / mask.sum().clamp_min(1.0)


def video_elem_mask(x: torch.Tensor) -> torch.Tensor:
    """Mask ``[1, 1, T, 1, 1]`` that excludes the clean conditioning frame 0."""
    if x.ndim != 5:
        raise ValueError(f"Expected video latents [B,C,T,H,W], got {tuple(x.shape)}")
    mask = torch.ones((1, 1, x.shape[2], 1, 1), device=x.device, dtype=torch.bool)
    mask[:, :, 0] = False
    return mask
