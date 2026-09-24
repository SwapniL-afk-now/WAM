"""Counterfactual swap test: do actions follow the imagined future? (BadWAM test)

For recorded states we compare, with the same action noise:

* ``d_noise``: a(v_A, eps1) vs a(v_A, eps2)   -- action-noise spread (floor)
* ``d_same``:  a(v_A, eps1) vs a(v_B, eps1)   -- another imagined future of the same state
* ``d_cross``: a(v_A, eps1) vs a(v_X, eps1)   -- a future imagined for a different state

Sensitivity ratios ``d_same / d_noise`` and ``d_cross / d_noise`` near 1 mean
the action head ignores imagination; large ratios mean it acts on it.

    python analysis/swap_test.py --model-config configs/model/fastwam_imago.yaml \
        --imago-ckpt runs/x/actor_step100.pt --record-dir rec/ --num-calls 20
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from omegaconf import OmegaConf

from common import load_calls


@torch.no_grad()
def run(policy, calls, num_calls: int, seed: int) -> dict:
    cfg = policy.policy_cfg
    g = torch.Generator(device="cpu").manual_seed(seed)
    d_noise, d_same, d_cross = [], [], []
    for call in calls[:num_calls]:
        first = torch.from_numpy(call.first_latents).to(policy.device, policy.torch_dtype)
        proprio = torch.from_numpy(call.proprio).to(policy.device, policy.torch_dtype)
        ids = torch.from_numpy(call.prompt_id).long()
        context, mask = policy._context(ids, proprio)
        b = first.shape[0]
        latent_t = (cfg.num_video_frames - 1) // policy.vae.temporal_downsample_factor + 1
        vshape = (b, first.shape[1], latent_t, *first.shape[3:])
        ashape = (b, cfg.action_horizon, policy.action_expert.action_dim)
        eps1 = torch.randn(ashape, generator=g)
        eps2 = torch.randn(ashape, generator=g)
        v_a, a_a1 = policy.imagine_and_act(
            first, context, mask, cfg.eval_video_steps, cfg.eval_action_steps,
            video_noise=torch.randn(vshape, generator=g), action_noise=eps1,
        )
        _, a_a2 = policy.imagine_and_act(
            first, context, mask, cfg.eval_video_steps, cfg.eval_action_steps,
            action_noise=eps2, video_override=v_a,
        )
        v_b, _ = policy.imagine_and_act(
            first, context, mask, cfg.eval_video_steps, 1,
            video_noise=torch.randn(vshape, generator=g),
        )
        _, a_b1 = policy.imagine_and_act(
            first, context, mask, cfg.eval_video_steps, cfg.eval_action_steps,
            action_noise=eps1, video_override=v_b,
        )
        # Cross-state: roll the batch so each env gets another env's future.
        v_x = torch.roll(v_a, shifts=1, dims=0)
        v_x[:, :, 0:1] = first  # keep the true current frame
        _, a_x1 = policy.imagine_and_act(
            first, context, mask, cfg.eval_video_steps, cfg.eval_action_steps,
            action_noise=eps1, video_override=v_x,
        )
        n = cfg.num_action_chunks  # executed part only
        dist = lambda x, y: (x[:, :n] - y[:, :n]).float().flatten(1).norm(dim=1).cpu().numpy()
        d_noise.append(dist(a_a1, a_a2))
        d_same.append(dist(a_a1, a_b1))
        d_cross.append(dist(a_a1, a_x1))
    d_noise, d_same, d_cross = (np.concatenate(x) for x in (d_noise, d_same, d_cross))
    floor = np.median(d_noise) + 1e-8
    return {
        "states": int(d_noise.size),
        "d_noise_median": float(np.median(d_noise)),
        "d_same_median": float(np.median(d_same)),
        "d_cross_median": float(np.median(d_cross)),
        "sensitivity_same": float(np.median(d_same) / floor),
        "sensitivity_cross": float(np.median(d_cross) / floor),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/fastwam_imago.yaml")
    parser.add_argument("--imago-ckpt", default=None)
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--num-calls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from imago.builder import get_model

    cfg = OmegaConf.load(args.model_config)
    cfg.imago_ckpt_path = args.imago_ckpt
    policy = get_model(cfg).eval()
    result = run(policy, load_calls(args.record_dir), args.num_calls, args.seed)
    result["imago_ckpt"] = args.imago_ckpt
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
