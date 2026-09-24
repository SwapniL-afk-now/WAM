"""C2 baseline: action-only Flow-SDE GRPO on FastWAM imagine-then-act.

The standard way to RL-tune a flow policy (πRL / OpenPI Flow-SDE, as ported to
FastWAM in ``Yutenji-Nyamu/rlinf_fastwam``,
``rlinf/models/embodiment/fastwam/fastwam_rl.py``, Apache-2.0). It is the
"likelihood-based" counterpart of IMAGO's likelihood-free NFT:

* Rollout: imagine the future with the (frozen) video expert exactly as IMAGO
  does, then denoise the action chain with the ODE except at **one** randomly
  chosen step (shared across the batch, OpenPI-style), where a Flow-SDE
  transition with ``noise_level`` is sampled. The chain and the step index are
  stored; ``prev_logprobs`` is the Gaussian log-density of that transition.
* Training (RLinf ``loss_type: actor``, PPO-clip GRPO): replay the stored step
  with the current action expert and return its log-density (and entropy).
* Only the executed prefix (``num_action_chunks``) enters the log-prob, like
  IMAGO's ``nft_action_executed_only``.

Selected with ``actor.model.imago.policy: flow_grpo`` (``imago/builder.py``).
"""

from __future__ import annotations

import math

import torch

from imago.policy import FastWAMImaginePolicy


# --------------------------------------------------------- Flow-SDE primitives
# Ported from rlinf_fastwam/fastwam_rl.py (flow_step_mean_std, gaussian_logprob,
# gaussian_entropy). Validation checks are kept; shapes are [B, H, A].
def _bcast(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if value.ndim == 0:
        value = value.reshape(1).expand(ref.shape[0])
    while value.ndim < ref.ndim:
        value = value.unsqueeze(-1)
    return value.to(ref.device)


def flow_step_mean_std(x, velocity, t, next_t, delta, noise_level: float):
    """OpenPI Flow-SDE mean/std on FastWAM's shifted grid (``x_t = (1-t)x0 + t·noise``)."""
    if noise_level <= 0:
        raise ValueError("flow_sde_noise_level must be positive")
    x, velocity = x.float(), velocity.float()
    t, next_t, delta = _bcast(t.float(), x), _bcast(next_t.float(), x), _bcast(delta.float(), x)
    if not bool((delta < 0).all()):
        raise ValueError("signed delta must be strictly negative")
    dt = -delta
    first = torch.isclose(t, torch.ones_like(t), rtol=0.0, atol=1e-6)
    denom_t = torch.where(first, next_t, t)
    if not bool((1.0 - denom_t > 0).all()):
        raise ValueError("Flow-SDE sigma denominator must be strictly positive")
    x0 = x - t * velocity
    x1 = x + (1.0 - t) * velocity
    sigma = float(noise_level) * torch.sqrt(t / (1.0 - denom_t))
    mean = x0 * (1.0 - (t - dt)) + x1 * (t - dt - sigma.square() * dt / (2.0 * t))
    std = (torch.sqrt(dt) * sigma).expand_as(mean)
    mean_ode = x + delta * velocity
    return mean, std, mean_ode


def gaussian_logprob(sample, mean, std):
    sample, mean, std = sample.float(), mean.float(), std.float()
    return -0.5 * ((sample - mean) / std).square() - torch.log(std) - 0.5 * math.log(2.0 * math.pi)


def gaussian_entropy(std):
    return torch.log(std.float()) + 0.5 * math.log(2.0 * math.pi * math.e)


# ------------------------------------------------------------------ policy
class FlowGRPOPolicy(FastWAMImaginePolicy):
    """FastWAM imagine-then-act with a Flow-SDE action head for PPO/GRPO."""

    def _action_schedule(self, num_steps: int):
        sched = self.infer_action_scheduler
        timesteps, deltas = sched.build_inference_schedule(
            num_inference_steps=num_steps,
            device=self.device,
            dtype=self.torch_dtype,
            shift_override=self.policy_cfg.action_sigma_shift,
        )
        t = timesteps.float() / float(sched.num_train_timesteps)
        next_t = (t + deltas.float()).clamp_min(0.0)
        return timesteps, deltas, t, next_t

    def _imagine(self, first_latents, context, context_mask, video_steps: int):
        """Imagined video from the frozen video expert (same path as IMAGO)."""
        cfg = self.policy_cfg
        if cfg.imagination_mode == "first_frame":
            return first_latents
        latent_t = (cfg.num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        shape = (first_latents.shape[0], first_latents.shape[1], latent_t, *first_latents.shape[3:])
        x_video = torch.randn(shape, device=self.device, dtype=torch.float32).to(self.torch_dtype)
        x_video[:, :, 0:1] = first_latents

        def clamp(x):
            x[:, :, 0:1] = first_latents
            return x

        return self._ode(
            lambda x, s: self.video_velocity(x, s, first_latents, context, context_mask),
            x_video, self.infer_video_scheduler, video_steps, cfg.video_sigma_shift, clamp_fn=clamp,
        )

    @torch.no_grad()
    def predict_action_batch(self, env_obs: dict, mode: str = "train", **kwargs):
        if mode == "eval":
            return super().predict_action_batch(env_obs, mode=mode, **kwargs)
        cfg = self.policy_cfg
        images = self._build_input_images(env_obs["main_images"], env_obs.get("wrist_images"))
        proprio = self._normalize_proprio(env_obs["states"]).to(self.device, self.torch_dtype)
        prompt_ids = self._prompt_ids(env_obs["task_descriptions"])
        context, context_mask = self._context(prompt_ids, proprio)
        first_latents = self.encode_first_frame(images)
        x_video = self._imagine(first_latents, context, context_mask, cfg.train_video_steps)
        cache = self.build_video_cache(x_video, context, context_mask, cfg.action_horizon)

        steps = cfg.train_action_steps
        timesteps, deltas, t_norm, next_t = self._action_schedule(steps)
        batch = first_latents.shape[0]
        shape = (batch, cfg.action_horizon, self.action_expert.action_dim)
        x = torch.randn(shape, device=self.device, dtype=torch.float32)
        eps = torch.randn(shape, device=self.device, dtype=torch.float32)
        k = int(torch.randint(0, steps, (1,)).item())  # one shared stochastic step (OpenPI)
        chains = [x]
        logprob = None
        for i in range(steps):
            sigma = t_norm[i].expand(batch)
            v = self.action_velocity(x.to(self.torch_dtype), sigma, context, context_mask, cache)
            mean, std, mean_ode = flow_step_mean_std(x, v, t_norm[i], next_t[i], deltas[i], cfg.flow_sde_noise_level)
            if i == k:
                x = mean + std * eps
                logprob = gaussian_logprob(x, mean, std)
            else:
                x = mean_ode
            chains.append(x)
        actions = self._to_env_actions(x)
        n = cfg.num_action_chunks
        forward_inputs = {
            "imago_first_latents": first_latents.detach(),
            "imago_prompt_id": prompt_ids.to(self.device),
            "imago_proprio": proprio.detach(),
            "nft_x0_video": x_video.detach(),
            "flow_chains": torch.stack(chains, dim=1).detach(),  # [B, S+1, H, A] fp32
            "flow_denoise_inds": torch.full((batch,), k, device=self.device, dtype=torch.long),
        }
        return actions, {
            "prev_logprobs": logprob[:, :n].float(),
            "prev_values": torch.zeros((batch, 1), device=self.device, dtype=torch.float32),
            "forward_inputs": forward_inputs,
        }

    def default_forward(self, forward_inputs: dict, compute_logprobs: bool = True,
                        compute_entropy: bool = False, compute_values: bool = False, **kwargs):
        """Replay the stored stochastic transition with the current action expert."""
        cfg = self.policy_cfg
        proprio = forward_inputs["imago_proprio"].to(self.device, self.torch_dtype)
        context, context_mask = self._context(forward_inputs["imago_prompt_id"], proprio)
        video = forward_inputs["nft_x0_video"].to(self.device, self.torch_dtype)
        with torch.no_grad():  # C2 trains the action expert only
            cache = self.build_video_cache(video, context, context_mask, cfg.action_horizon)
        chains = forward_inputs["flow_chains"].to(self.device, torch.float32)
        inds = forward_inputs["flow_denoise_inds"].to(self.device, torch.long)
        steps = chains.shape[1] - 1
        _, deltas, t_norm, next_t = self._action_schedule(steps)
        rows = torch.arange(chains.shape[0], device=self.device)
        x = chains[rows, inds]
        x_next = chains[rows, inds + 1]
        v = self.action_velocity(x.to(self.torch_dtype), t_norm[inds], context, context_mask, cache)
        mean, std, _ = flow_step_mean_std(x, v, t_norm[inds], next_t[inds], deltas[inds], cfg.flow_sde_noise_level)
        n = cfg.num_action_chunks
        out = {"logprobs": gaussian_logprob(x_next, mean, std)[:, :n], "values": None}
        if compute_entropy:
            out["entropy"] = gaussian_entropy(std)[:, :n]
        return out
