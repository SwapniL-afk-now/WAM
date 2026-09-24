"""Video-expert forward with an explicit expert, interaction tokens and layer taps.

This is ``FastWAMIDM._denoise_video`` generalised in three ways:

* The expert is an argument, so DIDO Stage I can run three different experts
  (frozen teacher, generator, fake score).
* It appends the DIDO interaction tokens when the expert has them
  (``imago.dido.interaction``).
* It returns token states at requested layers (for the alignment loss).
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from imago.dido import interaction as inter


def video_forward(
    expert,
    x_t: torch.Tensor,  # [B, C, T, h, w]
    sigma: torch.Tensor,  # [B] noise level in [0, 1]
    first: torch.Tensor,  # [B, C, 1, h, w] clean observation latent
    context: torch.Tensor,
    context_mask: torch.Tensor,
    *,
    num_train_timesteps: int = 1000,
    collect_layers: tuple[int, ...] = (),
    fuse_vae_embedding_in_latents: bool = True,
):
    """Returns ``(velocity [B,C,T,h,w], token_states [B,64,D] | None, {layer: token_states})``."""
    x_t = x_t.clone()
    x_t[:, :, 0:1] = first.to(x_t.dtype)
    timestep = (sigma.reshape(-1).float() * num_train_timesteps).to(x_t.dtype)
    (x_tok, t, t_mod, ctx, cmask, freqs, f, h, w, tpf) = expert.prepare(
        x=x_t,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
    )
    length = int(x_tok.shape[1])
    self_mask = expert.build_video_to_video_mask(
        video_seq_len=length, video_tokens_per_frame=tpf, device=x_tok.device
    )
    tokens = inter.get(expert)
    if tokens is not None:
        x_tok, t_mod, cmask, freqs, self_mask = tokens.extend_sequence(
            x_tok, t_mod, cmask, freqs, self_mask, tpf
        )
    use_ckpt = bool(getattr(expert, "use_gradient_checkpointing", False)) and torch.is_grad_enabled()
    hidden = {}
    for idx, block in enumerate(expert.blocks):
        if use_ckpt:
            x_tok = checkpoint(block, x_tok, ctx, t_mod, freqs, cmask, self_mask, use_reentrant=False)
        else:
            x_tok = block(x_tok, ctx, t_mod, freqs, context_mask=cmask, self_attn_mask=self_mask)
        if tokens is not None and idx in collect_layers:
            hidden[idx] = x_tok[:, length:]
    video_out = x_tok[:, :length]
    velocity = expert.unpatchify(expert.head(video_out, t), (f, h, w))
    tok_out = x_tok[:, length:] if tokens is not None else None
    return velocity, tok_out, hidden


def x0_from_velocity(x_t, velocity, sigma, first):
    """Flow convention ``x_t = (1-σ)x0 + σ·noise``, ``v = noise - x0``, so ``x0 = x_t - σ v``."""
    x0 = x_t - sigma.view(-1, 1, 1, 1, 1).to(x_t.dtype) * velocity
    x0 = x0.clone()
    x0[:, :, 0:1] = first.to(x0.dtype)
    return x0


def video_cache_inputs(expert, video_latents, context, context_mask, fuse=True):
    """Clean (t=0) video tokens + interaction tokens for the MoT K/V prefill."""
    batch = video_latents.shape[0]
    timestep = torch.zeros((batch,), dtype=video_latents.dtype, device=video_latents.device)
    (x_tok, _t, t_mod, ctx, cmask, freqs, f, h, w, tpf) = expert.prepare(
        x=video_latents,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=fuse,
    )
    length = int(x_tok.shape[1])
    self_mask = expert.build_video_to_video_mask(
        video_seq_len=length, video_tokens_per_frame=tpf, device=x_tok.device
    )
    tokens = inter.get(expert)
    if tokens is not None:
        x_tok, t_mod, cmask, freqs, self_mask = tokens.extend_sequence(
            x_tok, t_mod, cmask, freqs, self_mask, tpf
        )
    return x_tok, t_mod, ctx, cmask, freqs, self_mask, (int(f), int(h), int(w)), length
