"""Training-efficiency techniques from recent video/flow RL papers, ported to NFT.

NFT (DiffusionNFT, NVlabs, ICLR'26 oral) already trains on a *single* re-noised
step per sample, so the main win of one-step / sliding-window GRPO methods
(no backprop through the sampling chain) is built in. What still transfers is
how the noise level of that step is chosen and weighted:

* **Iso-temporal grouping** (Flash-GRPO, ICML'26,
  github.com/Shredded-Pork/Flash-GRPO): all samples of one GRPO group share the
  same training noise level, so the within-group advantage is not confounded
  by timestep difficulty.
* **Temporal gradient rectification** (Flash-GRPO): divide out the
  time-dependent gradient scale. For the x0-space NFT energy,
  ``x0_pred = x_t - t v`` gives ``dE/dv`` of order ``t^2``; weight ``1/t^2``,
  normalised to batch mean 1 (Flash-GRPO normalises its factor the same way).
* **Noise-aware weighting** (TempFlow-GRPO, ICLR'26,
  github.com/Shredded-Pork/TempFlow-GRPO): scale the loss by the SDE noise std
  ``sqrt(t / (1 - t))`` at the training step (more weight on early, high-noise,
  high-impact steps), normalised to mean 1 over the grid.
* **Sliding noise window** (MixGRPO, ECCV'26,
  github.com/Tencent-Hunyuan/MixGRPO): only train on a window of consecutive
  grid steps that slides from high to low noise during training.

NVFP4 rollouts (Sol-RL, NVlabs) live in ``imago/lowprec.py``.
"""

from __future__ import annotations

import torch

from imago.nft_math import shifted_sigma


def _grid(num_steps: int, shift: float, device) -> torch.Tensor:
    u = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)
    return shifted_sigma(u, shift)[:-1]  # noise levels of the rollout steps


def sliding_window(num_steps: int, window: int, update_step: int, interval: int) -> tuple[int, int]:
    """MixGRPO window ``[start, start + window)`` over grid indices (0 = noisiest).

    The window starts at the noisiest steps and advances by one index every
    ``interval`` updates, wrapping around once it reaches the end.
    """
    window = max(1, min(window, num_steps))
    if interval <= 0 or window == num_steps:
        return 0, num_steps
    positions = num_steps - window + 1
    start = (update_step // interval) % positions
    return start, start + window


def sample_grouped_timesteps(
    group_keys: torch.Tensor,
    num_steps: int,
    shift: float,
    *,
    iso_temporal: bool = True,
    window: tuple[int, int] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a grid index (and noise level ``t``) per sample.

    With ``iso_temporal`` every sample sharing a ``group_keys`` value gets the
    same index (Flash-GRPO); ``window`` restricts indices (MixGRPO).
    """
    device = group_keys.device
    lo, hi = window if window is not None else (0, num_steps)
    if iso_temporal:
        unique, inverse = torch.unique(group_keys.cpu(), return_inverse=True)
        per_group = torch.randint(lo, hi, (unique.numel(),), generator=generator)
        index = per_group[inverse]
    else:
        index = torch.randint(lo, hi, (group_keys.numel(),), generator=generator)
    index = index.to(device)
    t = _grid(num_steps, shift, device)[index]
    return index, t


def time_weights(
    t: torch.Tensor,
    mode: str,
    *,
    num_steps: int,
    shift: float,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Per-sample loss multipliers with mean ~1.

    ``none``: ones. ``rectify``: Flash-GRPO-style ``1/t^2`` normalised by the
    batch mean. ``tempflow``: TempFlow-GRPO noise std ``sqrt(t/(1-t))``
    normalised by its mean over the rollout grid.
    """
    if mode == "none":
        return torch.ones_like(t, dtype=torch.float32)
    t = t.float()
    if mode == "rectify":
        w = 1.0 / (t**2 + eps)
        return w / w.mean().clamp_min(1e-8)
    if mode == "tempflow":
        grid = _grid(num_steps, shift, t.device).clamp(eps, 1 - eps)
        std = lambda x: torch.sqrt(x.clamp(eps, 1 - eps) / (1 - x.clamp(eps, 1 - eps)))
        return std(t) / std(grid).mean()
    raise ValueError(f"Unknown time weight mode: {mode}")
