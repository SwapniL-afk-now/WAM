"""FastWAM imagine-then-act policy for RLinf (IMAGO).

Wraps upstream ``FastWAMOptionalIDM`` (FastWAM repo, commit ``7faa711``+) and
exposes the two things RLinf's embodied NFT pipeline needs:

1. ``predict_action_batch``: batched imagine-then-act rollout. For every env it
   denoises a future video from the current frame (ODE), then denoises an
   action chunk that attends to the *imagined* video through the video K/V
   cache (``FastWAMIDM.infer_action`` logic, batched). The per-sample initial
   noises make the imagined futures -- and therefore the actions -- differ
   across a GRPO group, which is where exploration comes from.
2. ``forward(ForwardType.NFT, branch=...)``: the velocity of either branch at an
   arbitrary ``(x_t, t)``, which the joint-NFT actor worker turns into the
   Diffusion-NFT loss for the video (imagination) and the action branch.

Only the rollout samples (``x0`` of video latents and actions) plus compact
conditioning are stored per step: the first-frame latent, a prompt-bank index
and the normalised proprio. Text embeddings come from a precomputed prompt bank
(``scripts/build_prompt_bank.py``) so the 5.7B umT5 encoder never sits on GPU.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch
from fastwam.models.wan22.fastwam_optional_idm import FastWAMOptionalIDM
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.fastwam.fastwam_policy import (
    DEFAULT_PROMPT,
    FastWAMPolicy,
    _invert_gripper_action,
)

from imago.dido.token_refine import TokenRefineConfig, refine_cache
from imago.dido.video_fn import video_cache_inputs, video_forward


@dataclass
class ImagoPolicyConfig:
    action_horizon: int = 32
    num_action_chunks: int = 8  # executed actions per step; 8 = 2 video frames
    num_video_frames: int = 9
    action_video_freq_ratio: int = 4
    # Denoising steps: few during RL rollouts, more for evaluation.
    train_video_steps: int = 4
    train_action_steps: int = 4
    eval_video_steps: int = 10
    eval_action_steps: int = 10
    video_sigma_shift: Optional[float] = None  # None -> scheduler infer shift
    action_sigma_shift: Optional[float] = None
    # "idm": imagine the future and act on it; "first_frame": Fast-WAM mode.
    imagination_mode: str = "idm"
    # Train the action NFT only on executed steps (like pi0 NFT).
    nft_action_executed_only: bool = True
    # Let action-branch gradients flow into the video expert via the cache.
    action_grad_through_video: bool = False
    binarize_gripper: bool = True
    concat_multi_camera: str = "horizontal"
    prompt_bank_path: Optional[str] = None
    record_dir: Optional[str] = None  # eval-time recording for analysis/
    max_record_calls: int = 2000
    # GRPO group size; envs are laid out as contiguous groups
    # (LiberoEnv: reset_state_ids.repeat(group_size)). Used for iso-temporal
    # NFT timestep sharing (Flash-GRPO).
    group_size: int = 8
    # Rollout precision: "bf16" or "nvfp4" (Sol-RL, needs transformer-engine).
    rollout_precision: str = "bf16"
    # DIDO dynamics-based token refinement of the imagined-video cache.
    token_refine: TokenRefineConfig = field(default_factory=TokenRefineConfig)
    # C2 baseline (imago/baselines/flow_grpo.py): Flow-SDE noise level.
    flow_sde_noise_level: float = 0.5
    # C3 baseline: store imagined / real frames for the realism reward.
    realism_frames: bool = False
    realism_frame_scale: float = 0.5


class FastWAMImaginePolicy(FastWAMOptionalIDM, BasePolicy):
    """RLinf policy around FastWAM Optional-IDM. Built by ``imago.builder``."""

    # Pure helpers reused from RLinf's FastWAM wrapper (they only touch the
    # processor / VAE-agnostic tensors, not the FastWAM denoising API).
    # RLinf FSDP2 (``wrap_policy.no_split_names: ["__none__"]``): one root unit.
    # MoT calls blocks directly and reads some weights outside ``forward``, so
    # no per-block or per-embedding units.
    _fsdp_wrap_embeddings = False
    _center_crop_resize_batch = staticmethod(FastWAMPolicy._center_crop_resize_batch)
    _build_input_images = FastWAMPolicy._build_input_images
    _normalize_proprio = FastWAMPolicy._normalize_proprio
    _denormalize_action = FastWAMPolicy._denormalize_action
    _append_proprio_to_context = FastWAMPolicy._append_proprio_to_context
    gradient_checkpointing_enable = FastWAMPolicy.gradient_checkpointing_enable
    gradient_checkpointing_disable = FastWAMPolicy.gradient_checkpointing_disable

    # ------------------------------------------------------------------ setup
    def configure_imago(self, processor: Any, policy_cfg: ImagoPolicyConfig):
        self.processor = processor
        self.policy_cfg = policy_cfg
        self._state_key = processor.shape_meta["state"][0]["key"]
        self._action_key = processor.shape_meta["action"][0]["key"]
        self._cam_hw = [
            (int(m["shape"][1]), int(m["shape"][2])) for m in processor.shape_meta["images"]
        ]
        self._num_cameras = int(processor.num_output_cameras)
        # RLinf NFT utilities read ``model.config.num_steps``.
        self.config = SimpleNamespace(num_steps=int(policy_cfg.train_action_steps))
        self._prompt_index: dict[str, int] = {}
        self._prompt_context: Optional[torch.Tensor] = None
        self._prompt_mask: Optional[torch.Tensor] = None
        if policy_cfg.prompt_bank_path:
            self.load_prompt_bank(policy_cfg.prompt_bank_path)
        self._record_calls = 0
        self._train_calls = 0
        self._lowprec = None
        return self

    def load_prompt_bank(self, path: str) -> None:
        bank = torch.load(os.path.expanduser(path), map_location="cpu")
        self._prompt_index = {p: i for i, p in enumerate(bank["prompts"])}
        self._prompt_context = bank["context"]  # [N, L, D]
        self._prompt_mask = bank["context_mask"].bool()  # [N, L]

    def _prompt_ids(self, task_descriptions) -> torch.Tensor:
        ids = []
        for task in task_descriptions:
            prompt = DEFAULT_PROMPT.format(task=str(task))
            if prompt not in self._prompt_index:
                raise KeyError(
                    f"Prompt not in prompt bank: {prompt!r}. Rebuild it with "
                    "scripts/build_prompt_bank.py for this benchmark."
                )
            ids.append(self._prompt_index[prompt])
        return torch.tensor(ids, dtype=torch.long)

    def _context(self, prompt_ids: torch.Tensor, proprio: torch.Tensor):
        if self._prompt_context is None:
            raise RuntimeError("Prompt bank not loaded (policy.prompt_bank_path).")
        ids = prompt_ids.long().cpu()
        context = self._prompt_context[ids].to(self.device, dtype=self.torch_dtype)
        mask = self._prompt_mask[ids].to(self.device)
        return self._append_proprio_to_context(context, mask, proprio.to(self.device))

    # -------------------------------------------------------------- primitives
    @torch.no_grad()
    def encode_first_frame(self, images: torch.Tensor) -> torch.Tensor:
        """``[B,3,H,W]`` in [-1,1] -> first-frame latents ``[B,C,1,h,w]``."""
        video = images.unsqueeze(2).to(device=self.device, dtype=self.torch_dtype)
        return self.vae.model.encode(video, self.vae.scale)

    def _fuse_flag(self) -> bool:
        return bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

    def _video_self_attn_mask(self, latents: torch.Tensor) -> torch.Tensor:
        patch_t, patch_h, patch_w = (int(v) for v in self.video_expert.patch_size)
        tokens_per_frame = (latents.shape[3] // patch_h) * (latents.shape[4] // patch_w)
        seq_len = (latents.shape[2] // patch_t) * tokens_per_frame
        return self.video_expert.build_video_to_video_mask(
            video_seq_len=seq_len,
            video_tokens_per_frame=tokens_per_frame,
            device=latents.device,
        )

    def video_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        first_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Video-expert velocity ``v(x_t, t)``; frame 0 is clamped to the observation.

        Includes the DIDO interaction tokens when the expert has them.
        """
        v, _tok, _hidden = self.video_forward_full(x_t, t, first_latents, context, context_mask)
        return v

    def video_forward_full(self, x_t, t, first_latents, context, context_mask, collect_layers=()):
        """``(velocity, interaction-token states, {layer: token states})``."""
        return video_forward(
            self.video_expert,
            x_t,
            t,
            first_latents,
            context,
            context_mask,
            num_train_timesteps=self.infer_video_scheduler.num_train_timesteps,
            collect_layers=tuple(collect_layers),
            fuse_vae_embedding_in_latents=self._fuse_flag(),
        )

    def build_video_cache(
        self,
        video_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_seq_len: int,
    ):
        """Clean (t=0) video conditioning -> per-layer K/V cache + action mask.

        The cache holds the video tokens plus, when present, the 64 DIDO
        interaction tokens (the action expert reads the whole world-model
        stream, as in DIDO). In ``first_frame`` mode only the first latent
        frame is cached (FastWAM's first-frame action mask).
        """
        if self.policy_cfg.imagination_mode == "first_frame":
            video_latents = video_latents[:, :, 0:1]
        (tokens, t_mod, v_context, v_context_mask, freqs, self_mask, grid, video_len) = (
            video_cache_inputs(self.video_expert, video_latents, context, context_mask, self._fuse_flag())
        )
        cache_k, cache_v = self.mot.prefill_video_cache_tensor(
            video_tokens=tokens,
            video_freqs=freqs,
            video_t_mod=t_mod,
            video_context=v_context,
            video_context_mask=v_context_mask,
            video_attention_mask=self_mask,
        )
        refine = self.policy_cfg.token_refine
        if refine.enabled and grid[0] > 1:
            # DIDO dynamics-based token refinement on the future-video part only;
            # frame 0 and the interaction tokens are kept unchanged.
            vid_k = [k[:, :video_len] for k in cache_k]
            vid_v = [v[:, :video_len] for v in cache_v]
            vid_k, vid_v = refine_cache(vid_k, vid_v, grid, refine)
            cache_k = [torch.cat([rk, k[:, video_len:]], dim=1) for rk, k in zip(vid_k, cache_k)]
            cache_v = [torch.cat([rv, v[:, video_len:]], dim=1) for rv, v in zip(vid_v, cache_v)]
        cache_len = int(cache_k[0].shape[1])
        # Action tokens attend to every cached world-model token and to each other.
        action_mask = torch.ones(
            (action_seq_len, cache_len + action_seq_len), dtype=torch.bool, device=tokens.device
        )
        return cache_k, cache_v, action_mask

    def action_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        cache,
    ) -> torch.Tensor:
        cache_k, cache_v, action_mask = cache
        timestep = (t.reshape(-1) * self.infer_action_scheduler.num_train_timesteps).to(x_t.dtype)
        return self._denoise_action_with_video_cache(
            latents_action=x_t,
            timestep_action=timestep,
            context=context,
            context_mask=context_mask,
            video_cache_k=cache_k,
            video_cache_v=cache_v,
            action_attention_mask=action_mask,
        )

    @staticmethod
    def _ode(velocity_fn, x: torch.Tensor, scheduler, num_steps: int, shift, clamp_fn=None):
        timesteps, deltas = scheduler.build_inference_schedule(
            num_inference_steps=num_steps,
            device=x.device,
            dtype=x.dtype,
            shift_override=shift,
        )
        for step_t, step_delta in zip(timesteps, deltas):
            sigma = (step_t / scheduler.num_train_timesteps).expand(x.shape[0])
            v = velocity_fn(x, sigma)
            x = scheduler.step(v, step_delta, x)
            if clamp_fn is not None:
                x = clamp_fn(x)
        return x

    @torch.no_grad()
    def imagine_and_act(
        self,
        first_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_steps: int,
        action_steps: int,
        video_noise: Optional[torch.Tensor] = None,
        action_noise: Optional[torch.Tensor] = None,
        video_override: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(x0_video [B,C,T,h,w], x0_action [B,H,A])``.

        ``video_override`` skips imagination and conditions actions on a given
        video (used by the counterfactual swap test in ``analysis/``).
        """
        cfg = self.policy_cfg
        batch = first_latents.shape[0]
        latent_t = (cfg.num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        if video_override is not None:
            x_video = video_override.to(self.device, self.torch_dtype)
        elif cfg.imagination_mode == "first_frame":
            x_video = first_latents
        else:
            shape = (batch, first_latents.shape[1], latent_t, *first_latents.shape[3:])
            if video_noise is None:
                video_noise = torch.randn(shape, device=self.device, dtype=torch.float32)
            x_video = video_noise.to(self.device, self.torch_dtype)
            x_video[:, :, 0:1] = first_latents

            def clamp(x):
                x[:, :, 0:1] = first_latents
                return x

            x_video = self._ode(
                lambda x, s: self.video_velocity(x, s, first_latents, context, context_mask),
                x_video,
                self.infer_video_scheduler,
                video_steps,
                cfg.video_sigma_shift,
                clamp_fn=clamp,
            )

        cache = self.build_video_cache(x_video, context, context_mask, cfg.action_horizon)
        if action_noise is None:
            action_noise = torch.randn(
                (batch, cfg.action_horizon, self.action_expert.action_dim),
                device=self.device,
                dtype=torch.float32,
            )
        x_action = self._ode(
            lambda x, s: self.action_velocity(x, s, context, context_mask, cache),
            action_noise.to(self.device, self.torch_dtype),
            self.infer_action_scheduler,
            action_steps,
            cfg.action_sigma_shift,
        )
        return x_video, x_action

    # ---------------------------------------------------------------- rollout
    def _to_env_actions(self, x_action: torch.Tensor) -> np.ndarray:
        actions = self._denormalize_action(x_action.float().cpu())
        # FastWAM gripper: 0=close, 1=open  ->  LIBERO: -1=open, +1=close.
        actions[..., -1] = actions[..., -1] * 2 - 1
        actions = _invert_gripper_action(actions)
        if self.policy_cfg.binarize_gripper:
            actions[..., -1] = np.sign(actions[..., -1])
        return actions[:, : self.policy_cfg.num_action_chunks].astype(np.float32)

    @torch.no_grad()
    def predict_action_batch(self, env_obs: dict, mode: str = "train", **kwargs):
        cfg = self.policy_cfg
        images = self._build_input_images(env_obs["main_images"], env_obs.get("wrist_images"))
        proprio = self._normalize_proprio(env_obs["states"]).to(self.device, self.torch_dtype)
        prompt_ids = self._prompt_ids(env_obs["task_descriptions"])
        context, context_mask = self._context(prompt_ids, proprio)
        first_latents = self.encode_first_frame(images)

        is_eval = mode == "eval"
        precision_ctx = contextlib.nullcontext()
        if cfg.rollout_precision == "nvfp4":
            # Sol-RL: explore in NVFP4, train in BF16 (NFT re-evaluates the
            # stored x0 samples with the BF16 actor, so rollouts may be low-precision).
            from imago.lowprec import NVFP4Rollout

            if self._lowprec is None:
                self._lowprec = NVFP4Rollout(self)
            precision_ctx = self._lowprec.active()
        with precision_ctx:
            x_video, x_action = self.imagine_and_act(
                first_latents,
                context,
                context_mask,
                video_steps=cfg.eval_video_steps if is_eval else cfg.train_video_steps,
                action_steps=cfg.eval_action_steps if is_eval else cfg.train_action_steps,
            )
        actions = self._to_env_actions(x_action)
        if is_eval:
            if cfg.record_dir:
                self._record(images, x_video, x_action, env_obs, first_latents, proprio, prompt_ids)
            return actions, {"prev_values": None}

        batch = actions.shape[0]
        nft_action = x_action
        if cfg.nft_action_executed_only:
            nft_action = x_action[:, : cfg.num_action_chunks]
        # Unique id of (rollout rank, step call, GRPO group) for iso-temporal NFT.
        rank = int(os.environ.get("RANK", 0))
        slots = torch.arange(batch, device=self.device) // max(1, cfg.group_size)
        group_key = rank * 10**9 + self._train_calls * 10**4 + slots
        self._train_calls += 1
        forward_inputs = {
            "imago_group_key": group_key.long(),
            "imago_first_latents": first_latents.detach(),
            "imago_prompt_id": prompt_ids.to(self.device),
            "imago_proprio": proprio.detach(),
            "nft_x0_video": x_video.detach(),
            "nft_x0_action": nft_action.detach(),
            "nft_x0_action_full": x_action.detach(),
        }
        if cfg.realism_frames:
            forward_inputs.update(self._realism_frames(images, x_video))
        zeros = torch.zeros(
            (batch, cfg.num_action_chunks, actions.shape[-1]),
            device=self.device,
            dtype=torch.float32,
        )
        return actions, {
            "prev_logprobs": zeros,
            "prev_values": torch.zeros((batch, 1), device=self.device, dtype=torch.float32),
            "forward_inputs": forward_inputs,
        }

    @torch.no_grad()
    def _realism_frames(self, images: torch.Tensor, x_video: torch.Tensor) -> dict:
        """C3 (realism reward): the current real frame and the imagined frame at
        the end of the executed chunk, as downscaled uint8 images.

        The reward -LPIPS(imagined_t, real_{t+1}) is computed by the actor worker
        (``imago/baselines/realism.py``), where step t+1's real frame is available.
        """
        cfg = self.policy_cfg
        frame = cfg.num_action_chunks // max(1, cfg.action_video_freq_ratio)
        n_lat = (frame + self.vae.temporal_downsample_factor - 1) // self.vae.temporal_downsample_factor + 1
        n_lat = min(n_lat, x_video.shape[2])
        video = self.vae.decode(x_video[:, :, :n_lat], device=self.device, tiled=False)  # [B,3,T,H,W]
        imagined = video[:, :, min(frame, video.shape[2] - 1)]

        def to_u8(x):
            x = x.float()
            if cfg.realism_frame_scale != 1.0:
                x = torch.nn.functional.interpolate(
                    x, scale_factor=cfg.realism_frame_scale, mode="bilinear", align_corners=False
                )
            return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)

        return {"imago_real_u8": to_u8(images.to(self.device)), "imago_imag_u8": to_u8(imagined)}

    # ------------------------------------------------------------ NFT forward
    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.NFT:
            return self.nft_forward(**kwargs)
        if forward_type == ForwardType.SFT:
            loss, metrics = self.training_loss(kwargs["data"])
            return {"loss": loss, **{k: torch.as_tensor(v) for k, v in metrics.items()}}
        return self.default_forward(**kwargs)

    def nft_forward(
        self,
        forward_inputs: dict,
        branch: str,
        x_t: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Velocity of one branch at ``(x_t, t)`` for the joint NFT loss."""
        proprio = forward_inputs["imago_proprio"].to(self.device, self.torch_dtype)
        context, context_mask = self._context(forward_inputs["imago_prompt_id"], proprio)
        x_t = x_t.to(self.device, self.torch_dtype)
        t = t.to(self.device)
        if branch == "video":
            first = forward_inputs["imago_first_latents"].to(self.device, self.torch_dtype)
            v = self.video_velocity(x_t, t, first, context, context_mask)
            return {"v_theta": v}
        if branch == "action":
            video = forward_inputs["nft_x0_video"].to(self.device, self.torch_dtype)
            full_len = int(forward_inputs["nft_x0_action_full"].shape[1])
            if self.policy_cfg.action_grad_through_video:
                cache = self.build_video_cache(video, context, context_mask, full_len)
            else:
                with torch.no_grad():
                    cache = self.build_video_cache(video, context, context_mask, full_len)
            # The action expert always denoises the full horizon; pad the
            # executed prefix with the rollout's own unexecuted tail.
            x_full = forward_inputs["nft_x0_action_full"].to(self.device, self.torch_dtype).clone()
            x_full = self._pad_action_xt(x_full, x_t, t)
            v = self.action_velocity(x_full, t, context, context_mask, cache)
            return {"v_theta": v[:, : x_t.shape[1]]}
        raise ValueError(f"Unknown NFT branch: {branch!r}")

    @staticmethod
    def _pad_action_xt(x0_full: torch.Tensor, x_t_prefix: torch.Tensor, t: torch.Tensor):
        """Noise the tail with the same ``t`` so the whole horizon is at one level."""
        n = x_t_prefix.shape[1]
        if n == x0_full.shape[1]:
            return x_t_prefix
        t_bc = t.view(-1, 1, 1).to(x0_full.dtype)
        tail = x0_full[:, n:]
        tail = (1 - t_bc) * tail + t_bc * torch.randn_like(tail)
        return torch.cat([x_t_prefix, tail], dim=1)

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "IMAGO trains with loss_type=embodied_joint_nft (ForwardType.NFT)."
        )

    # ---------------------------------------------------------- recording
    @torch.no_grad()
    def _record(
        self, images, x_video, x_action, env_obs, first_latents, proprio, prompt_ids
    ) -> None:
        cfg = self.policy_cfg
        if self._record_calls >= cfg.max_record_calls:
            return
        os.makedirs(cfg.record_dir, exist_ok=True)
        video = self.vae.decode(x_video, device=self.device, tiled=False)  # [B,3,T,H,W]
        video = ((video.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy()
        real = ((images.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy()
        rank = int(os.environ.get("RANK", 0))
        path = os.path.join(cfg.record_dir, f"r{rank}_call{self._record_calls:06d}.npz")
        np.savez_compressed(
            path,
            real_frame=real,  # [B,3,H,W] observation this chunk was planned from
            imagined=video,  # [B,3,T,H,W]
            x0_video=x_video.float().cpu().numpy(),
            x0_action=x_action.float().cpu().numpy(),
            first_latents=first_latents.float().cpu().numpy(),
            proprio=proprio.float().cpu().numpy(),
            prompt_id=prompt_ids.cpu().numpy(),
            num_action_chunks=cfg.num_action_chunks,
            action_video_freq_ratio=cfg.action_video_freq_ratio,
            task=np.array([str(t) for t in env_obs["task_descriptions"]]),
        )
        self._record_calls += 1
