"""C3 baseline: dense realism reward (VAMPO / WAM-RL style).

reward_t += w · ( -LPIPS(imagined frame at the end of chunk t, real frame at t+1) )

This is the "reward the world model for being right" alternative to IMAGO's
outcome-only objective: same rollouts, same joint NFT, only the reward
changes. Frames come from ``FastWAMImaginePolicy._realism_frames`` (enabled by
``actor.model.imago.realism_frames: true``) as downscaled uint8 images.

* The term is zero for the last step of the rollout and for steps whose
  episode ended (``dones`` after step t), since there is no next real frame.
* Added after the success-based group filter, so it only reshapes advantages
  inside groups that already have mixed outcomes (same kept data as IMAGO).
* Needs the ``lpips`` package (server install, see AGENTS.md).
"""

from __future__ import annotations

import torch

_LPIPS = {}


def _lpips(device):
    if device not in _LPIPS:
        try:
            import lpips
        except ImportError as exc:  # pragma: no cover - server dependency
            raise ImportError("C3 realism reward needs `pip install lpips` (see AGENTS.md).") from exc
        _LPIPS[device] = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)
    return _LPIPS[device]


@torch.no_grad()
def realism_scores(imag_u8: torch.Tensor, real_u8: torch.Tensor, device, chunk: int = 256) -> torch.Tensor:
    """``-LPIPS`` per pair; inputs ``[N,3,H,W]`` uint8, output ``[N]`` float (CPU)."""
    net = _lpips(device)
    out = []
    for s in range(0, imag_u8.shape[0], chunk):
        a = imag_u8[s:s + chunk].to(device).float() / 127.5 - 1.0
        b = real_u8[s:s + chunk].to(device).float() / 127.5 - 1.0
        out.append(-net(a, b).flatten().float().cpu())
    return torch.cat(out) if out else torch.zeros(0)


@torch.no_grad()
def add_realism_reward(rollout_batch: dict, weight: float, device) -> dict[str, float]:
    """In-place: ``rewards[t, :, -1] += weight · realism_t``. Returns metrics."""
    fi = rollout_batch["forward_inputs"]
    imag, real = fi["imago_imag_u8"], fi["imago_real_u8"]  # [T', B, 3, h, w]
    rewards = rollout_batch["rewards"]  # [T, B, num_action_chunks]
    steps = min(rewards.shape[0], imag.shape[0], real.shape[0] - 1 if real.shape[0] > 1 else 0)
    if steps <= 0:
        return {"c3/realism": 0.0}
    batch = rewards.shape[1]
    scores = realism_scores(
        imag[:steps].reshape(-1, *imag.shape[2:]), real[1:steps + 1].reshape(-1, *real.shape[2:]), device
    ).reshape(steps, batch)
    valid = torch.ones_like(scores, dtype=torch.bool)
    dones = rollout_batch.get("dones", None)
    if dones is not None:  # dones[t + 1]: episode ended during step t
        ended = dones[1:steps + 1].reshape(steps, batch, -1).any(dim=-1).cpu()
        valid &= ~ended
    term = torch.where(valid, scores, torch.zeros_like(scores)).to(rewards.dtype)
    rewards[:steps, :, -1] += weight * term.to(rewards.device)
    n = valid.sum().clamp_min(1)
    return {"c3/realism": float((scores * valid).sum() / n), "c3/realism_valid_frac": float(valid.float().mean())}
