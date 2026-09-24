"""Week-1 profiler: rollout throughput and NFT step memory on one GPU.

Builds the IMAGO policy outside RLinf's runner and times
``predict_action_batch`` (imagine-then-act) plus one joint NFT
forward/backward per branch on synthetic LIBERO-shaped observations.

    python scripts/profile_rollout.py --batch-sizes 8 16 32 --video-steps 4
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from imago.builder import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from imago.nft_math import noise_to, sample_nft_timesteps


def fake_obs(policy, batch: int, hw: int = 256) -> dict:
    prompts = list(policy._prompt_index.keys())
    prefix = "instruction: "
    tasks = [prompts[i % len(prompts)].split(prefix, 1)[-1] for i in range(batch)]
    return {
        "main_images": np.random.randint(0, 255, (batch, hw, hw, 3), dtype=np.uint8),
        "wrist_images": np.random.randint(0, 255, (batch, hw, hw, 3), dtype=np.uint8),
        "states": np.random.randn(batch, 8).astype(np.float32),
        "task_descriptions": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/fastwam_imago.yaml")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--video-steps", type=int, default=4)
    parser.add_argument("--action-steps", type=int, default=4)
    parser.add_argument("--nft-micro-batch", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.model_config)
    cfg.imago.train_video_steps = args.video_steps
    cfg.imago.train_action_steps = args.action_steps
    policy = get_model(cfg).cuda()
    policy.eval()

    print("batch | s/call | calls/s*batch (env-steps/s) | peak GB")
    for batch in args.batch_sizes:
        obs = fake_obs(policy, batch)
        policy.predict_action_batch(obs, mode="train")  # warm-up
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for _ in range(args.repeats):
            _, out = policy.predict_action_batch(obs, mode="train")
        torch.cuda.synchronize()
        per_call = (time.time() - start) / args.repeats
        peak = torch.cuda.max_memory_allocated() / 2**30
        env_steps = batch * policy.policy_cfg.num_action_chunks / per_call
        print(f"{batch:5d} | {per_call:6.2f} | {env_steps:10.1f} | {peak:6.1f}")

    # One NFT step per branch (gradients through LoRA only).
    mb = args.nft_micro_batch
    _, out = policy.predict_action_batch(fake_obs(policy, mb), mode="train")
    fi = out["forward_inputs"]
    policy.train()
    for branch, key in (("video", "nft_x0_video"), ("action", "nft_x0_action")):
        x0 = fi[key]
        _, t = sample_nft_timesteps(mb, 4, 5.0 if branch == "video" else 1.0, x0.device, torch.float32)
        x_t = noise_to(x0, t, torch.randn_like(x0))
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = policy(
                forward_type=ForwardType.NFT, forward_inputs=fi, branch=branch, x_t=x_t, t=t
            )["v_theta"]
        v.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        print(f"NFT {branch}: micro_batch={mb} fwd+bwd {time.time() - start:.2f}s "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        policy.zero_grad(set_to_none=True)


if __name__ == "__main__":
    main()
