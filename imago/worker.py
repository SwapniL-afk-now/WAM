"""Joint-branch Diffusion-NFT actor worker (``loss_type: embodied_joint_nft``).

Extends RLinf's :class:`EmbodiedNFTFSDPPolicy` so that one outcome-only
advantage trains *both* FastWAM branches:

    L = w_video * NFT(video imagination) + w_action * NFT(action)
        + beta_real * ||v_video - v_video_pretrained||^2

Differences from the parent worker (full-parameter FastWAM, FSDP2):

* The EMA "rollout" reference and the pretrained reference (only kept when
  ``imago_beta_real > 0``) are copies of the *trainable* parameters with the
  same sharding (DTensor shards under FSDP2), about 1/N of the model per GPU.
  They are evaluated by copying them into the live parameters under
  ``no_grad`` and restoring afterwards; the parent instead rebuilds a full
  model from disk on every update.
* NFT noise levels are sampled on each branch's own shifted FastWAM grid.

Use RLinf ``fsdp_config.strategy: fsdp2`` with root-only wrapping
(``wrap_policy.no_split_names: ["__none__"]``), as in RLinf's FastWAM SFT
recipe: MoT reaches into expert blocks directly. ``imago_swap_buffer: cpu``
keeps the restore buffer on CPU (default ``auto``: CPU on 2 GPUs).

C3 baseline: ``algorithm.imago_realism_weight > 0`` adds a dense
-LPIPS(imagined, next real frame) reward before advantages
(``imago/baselines/realism.py``).
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from omegaconf import ListConfig

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy

from imago.efficiency import sample_grouped_timesteps, sliding_window, time_weights
from imago.nft_math import (
    grpo_to_unit,
    nft_branch_loss,
    noise_to,
    realism_anchor,
    video_elem_mask,
)

_FSDP_PREFIXES = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "module.")


def _clean(name: str) -> str:
    for prefix in _FSDP_PREFIXES:
        name = name.replace(prefix, "")
    return name


class EmbodiedJointNFTFSDPPolicy(EmbodiedNFTFSDPPolicy):
    # ------------------------------------------------------------ references
    def init_rollout_model(self) -> None:
        self._trainable = {
            _clean(n): p for n, p in self.model.named_parameters() if p.requires_grad
        }
        if not self._trainable:
            raise RuntimeError("No trainable parameters (check model.imago_trainable).")
        with torch.no_grad():
            # EMA rollout reference (needed whenever nft_tau < 1 at some point).
            self._ema_params = {n: p.detach().clone() for n, p in self._trainable.items()}
            # Pretrained reference for the realism anchor, only if it is used.
            self._init_params = (
                {n: p.detach().clone() for n, p in self._trainable.items()}
                if float(self.cfg.algorithm.get("imago_beta_real", 0.0)) > 0
                else None
            )
        buf = str(self.cfg.algorithm.get("imago_swap_buffer", "auto"))
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        self._swap_to_cpu = buf == "cpu" or (buf == "auto" and world <= 2)
        self.rollout_model_state_dict = {}  # parent API: unused

    @contextmanager
    def _swapped(self, source: dict[str, torch.Tensor]):
        """Temporarily load ``source`` into the trainable params (in place, shard-wise)."""
        saved = {}
        with torch.no_grad():
            for name, param in self._trainable.items():
                saved[name] = param.data.to("cpu", non_blocking=True) if self._swap_to_cpu else param.data.clone()
                param.data.copy_(source[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in self._trainable.items():
                    param.data.copy_(saved[name].to(param.device, non_blocking=True))
            if self._swap_to_cpu:
                torch.cuda.synchronize()

    def get_rollout_state_dict(self) -> dict:
        state = self.get_model_state_dict(cpu_offload=False, full_state_dict=False)
        if self._get_current_nft_tau() >= 1.0:
            return state
        state = dict(state)
        for key in list(state.keys()):
            name = _clean(key)
            if name in self._ema_params:
                state[key] = self._ema_params[name]
        return state

    def soft_update_rollout_model(self) -> None:
        tau = self._get_current_nft_tau()
        with torch.no_grad():
            for name, param in self._trainable.items():
                ema = self._ema_params[name]
                if tau >= 1.0:
                    ema.copy_(param.data)
                else:
                    ema.lerp_(param.data.to(ema.dtype), tau)

    # ---------------------------------------------------------- C3 reward
    def compute_advantages_and_returns(self):
        """Optionally add the C3 realism reward (``algorithm.imago_realism_weight``)."""
        fi = self.rollout_batch.get("forward_inputs", {}) or {}
        extra = {}
        weight = float(self.cfg.algorithm.get("imago_realism_weight", 0.0))
        if weight > 0:
            if "imago_imag_u8" not in fi:
                raise RuntimeError("imago_realism_weight > 0 needs actor.model.imago.realism_frames: true")
            from imago.baselines.realism import add_realism_reward

            extra = add_realism_reward(self.rollout_batch, weight, self.device)
        for key in ("imago_imag_u8", "imago_real_u8"):  # not needed for training
            fi.pop(key, None)
        metrics = super().compute_advantages_and_returns()
        if metrics is not None:
            metrics.update(extra)
        return metrics

    # --------------------------------------------------------------- config
    def _branch_cfg(self) -> dict:
        algo = self.cfg.algorithm
        model_cfg = self.cfg.actor.model
        icfg = model_cfg.get("imago", {}) or {}
        return {
            "video": {
                "weight": float(algo.get("imago_video_weight", 1.0)),
                "steps": int(icfg.get("train_video_steps", 4)),
                "shift": float(algo.get("imago_video_nft_shift", 5.0)),
            },
            "action": {
                "weight": float(algo.get("imago_action_weight", 1.0)),
                "steps": int(icfg.get("train_action_steps", 4)),
                "shift": float(algo.get("imago_action_nft_shift", 1.0)),
            },
        }

    def _active_branches(self) -> list[str]:
        return [b for b, c in self._branch_cfg().items() if c["weight"] > 0]

    def _weight_scale(self) -> tuple[float, float]:
        scale = self.cfg.algorithm.get("nft_weight_scale", 1.0)
        if isinstance(scale, (list, tuple, ListConfig)):
            return float(scale[0]), float(scale[1])
        return float(scale), float(scale)

    # ------------------------------------------------------ precompute refs
    def _branch_x0(self, forward_inputs: dict, branch: str) -> torch.Tensor:
        return forward_inputs["nft_x0_video" if branch == "video" else "nft_x0_action"]

    def _model_velocity(self, fi: dict, branch: str, x_t, t) -> torch.Tensor:
        out = self.model(
            forward_type=ForwardType.NFT,
            forward_inputs=fi,
            branch=branch,
            x_t=x_t,
            t=t,
        )
        return out["v_theta"]

    @torch.no_grad()
    def _reference_velocity(self, forward_inputs: dict, branch: str, source) -> torch.Tensor:
        """Velocity under swapped-in reference params (``None`` = current)."""
        micro = int(self.cfg.actor.micro_batch_size)
        x_t_all = forward_inputs[f"nft_{branch}_xt"]
        t_all = forward_inputs[f"nft_{branch}_t"]
        outs = []
        ctx = self._swapped(source) if source is not None else _nullcontext()
        with ctx, self.amp_context:
            for start in range(0, x_t_all.shape[0], micro):
                end = min(start + micro, x_t_all.shape[0])
                fi = put_tensor_device(
                    self._slice_forward_inputs(forward_inputs, start, end), self.device
                )
                v = self._model_velocity(
                    fi, branch, x_t_all[start:end].to(self.device), t_all[start:end].to(self.device)
                )
                outs.append(v.detach().to(x_t_all.device))
        return torch.cat(outs, dim=0)

    def _precompute_nft_training_inputs(self) -> None:
        fi = self.rollout_batch["forward_inputs"]
        g = torch.Generator().manual_seed(int(self.cfg.actor.seed) + int(getattr(self, "version", 0)))
        self.model.eval()
        for branch in self._active_branches():
            bcfg = self._branch_cfg()[branch]
            x0 = self._branch_x0(fi, branch)
            algo = self.cfg.algorithm
            window = None
            if int(algo.get("imago_noise_window", 0)) > 0:  # MixGRPO sliding window
                window = sliding_window(
                    bcfg["steps"],
                    int(algo.imago_noise_window),
                    int(getattr(self, "version", 0)),
                    int(algo.get("imago_window_interval", 10)),
                )
            keys = fi.get("imago_group_key", torch.arange(x0.shape[0], device=x0.device))
            _, t = sample_grouped_timesteps(
                keys,
                bcfg["steps"],
                bcfg["shift"],
                iso_temporal=bool(algo.get("imago_iso_temporal", True)),  # Flash-GRPO
                window=window,
                generator=g,
            )
            noise = torch.randn(x0.shape, generator=g, dtype=torch.float32).to(x0.device, x0.dtype)
            x_t = noise_to(x0, t, noise)
            if branch == "video":
                x_t[:, :, 0:1] = fi["imago_first_latents"].to(x_t.dtype)
            fi[f"nft_{branch}_xt"] = x_t
            fi[f"nft_{branch}_t"] = t
            on_policy = self._get_current_nft_tau() >= 1.0
            fi[f"nft_{branch}_vold"] = self._reference_velocity(
                fi, branch, None if on_policy else self._ema_params
            )
        if float(self.cfg.algorithm.get("imago_beta_real", 0.0)) > 0 and "video" in self._active_branches():
            fi["nft_video_vpre"] = self._reference_velocity(fi, "video", self._init_params)
        self.model.train()

    # ----------------------------------------------------------- the loss
    def nft_forward_and_loss(self, batch):
        algo = self.cfg.algorithm
        fi = batch["forward_inputs"]
        branches = self._branch_cfg()
        first_key = "nft_x0_video" if "video" in self._active_branches() else "nft_x0_action"
        batch_size = fi[first_key].shape[0]

        advantages = batch["advantages"].reshape(batch_size, -1)[:, 0]
        if str(algo.get("adv_type", "grpo")) != "raw":
            reward01 = grpo_to_unit(advantages, float(algo.get("adv_clip_max", 1.0)))
        else:
            reward01 = advantages.clamp(0.0, 1.0)
        loss_mask = batch.get("loss_mask", None)
        if loss_mask is None:
            sample_mask = torch.ones_like(reward01)
        else:
            sample_mask = loss_mask.reshape(batch_size, -1)[:, 0].float()

        total = 0.0
        metrics = {}
        beta = float(algo.get("nft_beta", 1.0))
        clip_ratio = algo.get("nft_clip_ratio", None)
        for branch in self._active_branches():
            x_t = fi[f"nft_{branch}_xt"]
            t = fi[f"nft_{branch}_t"]
            x0 = self._branch_x0(fi, branch)
            elem_mask = video_elem_mask(x0) if branch == "video" else None
            with self.amp_context:
                v_theta = self._model_velocity(fi, branch, x_t, t)
            terms = nft_branch_loss(
                v_theta=v_theta,
                v_old=fi[f"nft_{branch}_vold"],
                x_t=x_t,
                x0=x0,
                t=t,
                reward01=reward01,
                sample_mask=sample_mask,
                beta=beta,
                adv_clip_max=float(algo.get("adv_clip_max", 1.0)),
                weight_mode=str(algo.get("nft_weight_mode", "adaptive")),
                weight_scale=self._weight_scale(),
                clip_ratio=None if clip_ratio is None else float(clip_ratio),
                elem_mask=elem_mask,
                time_weight=time_weights(  # Flash-GRPO rectify / TempFlow weighting
                    t,
                    str(algo.get("imago_time_weight", "none")),
                    num_steps=branches[branch]["steps"],
                    shift=branches[branch]["shift"],
                ),
            )
            total = total + branches[branch]["weight"] * terms.loss
            metrics[f"actor/{branch}_nft_loss"] = terms.loss.item()
            metrics[f"actor/{branch}_delta_v_norm"] = terms.delta_v_norm.mean().item()
            metrics[f"actor/{branch}_E_pos"] = terms.e_pos.mean().item()
            metrics[f"actor/{branch}_E_neg"] = terms.e_neg.mean().item()
            metrics[f"actor/{branch}_clip_frac"] = terms.clip_frac
            if branch == "video" and "nft_video_vpre" in fi:
                anchor = realism_anchor(v_theta, fi["nft_video_vpre"], sample_mask, elem_mask)
                total = total + float(algo.get("imago_beta_real", 0.0)) * anchor
                metrics["actor/video_realism_anchor"] = anchor.item()
        metrics["actor/nft_tau"] = self._get_current_nft_tau()
        metrics["actor/reward01_mean"] = (reward01 * sample_mask).sum().item() / max(
            sample_mask.sum().item(), 1.0
        )
        metrics["actor/nft_loss"] = float(total.item())
        return total, metrics


@contextmanager
def _nullcontext():
    yield
