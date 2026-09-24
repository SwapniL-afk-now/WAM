"""Where do action tokens look in the imagined future? (RAL-style attention maps)

FastWAM's action path calls ``fastwam.models.wan22.mot.flash_attention`` (SDPA)
with action queries against ``[video cache ; action]`` keys. We wrap that
function, recompute the attention probabilities explicitly for the action
queries only (cheap: 32 queries), and aggregate:

* ``frame_mass[layer, T]``: attention mass per latent video frame
  (frame 0 = observed, 1.. = imagined),
* ``spatial[layer, T, h, w]``: spatial heat maps per frame.

Compare base vs IMAGO checkpoints: a shift of mass from frame 0 to imagined
frames (and onto contact regions) is direct evidence that RL made the action
head use the imagination.

    python analysis/attention_maps.py --record-dir rec/ --imago-ckpt ckpt.pt --layers 10 20 29
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf

from common import load_calls


@contextmanager
def capture_action_attention(action_len: int, video_len: int, layers: set[int], store: dict):
    import fastwam.models.wan22.mot as mot_mod

    original = mot_mod.flash_attention
    state = {"layer": 0}

    def wrapped(q, k, v, num_heads, ctx_mask=None, compatibility_mode=True):
        out = original(q, k, v, num_heads, ctx_mask=ctx_mask, compatibility_mode=compatibility_mode)
        if q.shape[1] == action_len and k.shape[1] == video_len + action_len:
            layer = state["layer"]
            state["layer"] += 1
            if layer in layers:
                qh = rearrange(q.float(), "b s (n d) -> b n s d", n=num_heads)
                kh = rearrange(k.float(), "b s (n d) -> b n s d", n=num_heads)
                logits = qh @ kh.transpose(-1, -2) / qh.shape[-1] ** 0.5
                if ctx_mask is not None:
                    logits = logits.masked_fill(~ctx_mask.to(logits.device).bool(), float("-inf"))
                probs = logits.softmax(-1).mean(dim=(1, 2))  # [B, Lk]
                store.setdefault(layer, []).append(probs[:, :video_len].cpu())
        return out

    mot_mod.flash_attention = wrapped
    try:
        yield state
    finally:
        mot_mod.flash_attention = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/fastwam_imago.yaml")
    parser.add_argument("--imago-ckpt", default=None)
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--num-calls", type=int, default=10)
    parser.add_argument("--layers", type=int, nargs="+", default=[5, 15, 29])
    parser.add_argument("--out", required=True, help=".npz with frame_mass / spatial")
    args = parser.parse_args()

    from imago.builder import get_model

    cfg = OmegaConf.load(args.model_config)
    cfg.imago_ckpt_path = args.imago_ckpt
    policy = get_model(cfg).eval()
    pcfg = policy.policy_cfg
    calls = load_calls(args.record_dir)[: args.num_calls]

    if pcfg.token_refine.enabled:
        raise SystemExit("attention_maps.py expects the unrefined cache; rerun with "
                         "imago.token_refine.enabled=false (spatial maps need the full grid).")
    lat = calls[0].x0_video.shape  # [B, C, T, h, w]
    patch = [int(p) for p in policy.video_expert.patch_size]
    grid_t, grid_h, grid_w = lat[2] // patch[0], lat[3] // patch[1], lat[4] // patch[2]
    video_len = grid_t * grid_h * grid_w
    store: dict[int, list] = {}
    with torch.no_grad():
        for call in calls:
            first = torch.from_numpy(call.first_latents).to(policy.device, policy.torch_dtype)
            proprio = torch.from_numpy(call.proprio).to(policy.device, policy.torch_dtype)
            context, mask = policy._context(torch.from_numpy(call.prompt_id).long(), proprio)
            video = torch.from_numpy(call.x0_video).to(policy.device, policy.torch_dtype)
            cache = policy.build_video_cache(video, context, mask, pcfg.action_horizon)
            x = torch.randn((first.shape[0], pcfg.action_horizon, policy.action_expert.action_dim),
                            device=policy.device, dtype=policy.torch_dtype)
            t = torch.full((first.shape[0],), 0.5, device=policy.device)  # mid-denoising
            with capture_action_attention(pcfg.action_horizon, video_len, set(args.layers), store):
                policy.action_velocity(x, t, context, mask, cache)

    frame_mass, spatial = [], []
    for layer in args.layers:
        probs = torch.cat(store[layer]).mean(0).numpy()  # [video_len]
        grid = probs.reshape(grid_t, grid_h, grid_w)
        frame_mass.append(grid.sum(axis=(1, 2)))
        spatial.append(grid)
    np.savez(args.out, layers=np.array(args.layers), frame_mass=np.stack(frame_mass),
             spatial=np.stack(spatial), imago_ckpt=str(args.imago_ckpt))
    print(json.dumps({f"layer{l}": fm.round(4).tolist() for l, fm in zip(args.layers, frame_mass)}))


if __name__ == "__main__":
    main()
