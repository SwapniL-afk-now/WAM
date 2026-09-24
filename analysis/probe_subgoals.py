"""Does RL turn imagination into subgoals? Temporal alignment probe.

Imagined pixel frame ``j`` nominally depicts time ``j * freq_ratio`` actions
ahead. Real frames are observed every ``num_action_chunks`` actions (one per
eval call). For each imagined frame we find which real future offset it
matches best:

    best_offset[j] = argmax_m sim(imagined_j, real_{c+m}),  m = 0..M

Nominal (predictive) imagination gives ``best_offset[j] ~ j * freq_ratio /
num_action_chunks``. Subgoal-like imagination "jumps ahead": late imagined
frames match further real futures, and the match concentrates on a few
keyframes. Similarity uses DINOv2 features when available (``--dino``),
otherwise negative pixel L1.

    python analysis/probe_subgoals.py --record-dir rec/ --max-offset 4 --out probe.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from common import load_calls


def pixel_sim(a: np.ndarray, b: np.ndarray) -> float:
    return -float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def make_dino_sim(device: str = "cuda"):
    import torch

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    cache = {}

    @torch.no_grad()
    def feat(x: np.ndarray):
        key = x.ctypes.data
        if key not in cache:
            t = torch.from_numpy(x).to(device).float().unsqueeze(0) / 255.0
            t = torch.nn.functional.interpolate(t, size=(224, 448), mode="bilinear")
            f = model((t - mean) / std)
            cache[key] = torch.nn.functional.normalize(f, dim=-1)
        return cache[key]

    return lambda a, b: float((feat(a) * feat(b)).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--max-offset", type=int, default=4)
    parser.add_argument("--dino", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sim = make_dino_sim() if args.dino else pixel_sim
    calls = load_calls(args.record_dir, args.rank)
    num_frames = calls[0].imagined.shape[2]
    counts = np.zeros((num_frames, args.max_offset + 1))
    for c, call in enumerate(calls):
        for i in range(call.real_frame.shape[0]):
            futures = []
            for m in range(args.max_offset + 1):
                if c + m >= len(calls) or calls[c + m].task[i] != call.task[i]:
                    break
                futures.append(calls[c + m].real_frame[i])
            if len(futures) < 2:
                continue
            for j in range(1, num_frames):
                scores = [sim(call.imagined[i, :, j], f) for f in futures]
                counts[j, int(np.argmax(scores))] += 1
    dist = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    nominal = [
        j * calls[0].freq_ratio / calls[0].num_action_chunks for j in range(num_frames)
    ]
    expected = (dist * np.arange(args.max_offset + 1)).sum(axis=1)
    result = {
        "best_offset_distribution": dist.round(4).tolist(),
        "expected_best_offset": expected.round(3).tolist(),
        "nominal_offset": [round(x, 3) for x in nominal],
        "jump_ahead_score": float(np.mean(expected[1:] - np.array(nominal[1:]))),
        "similarity": "dinov2" if args.dino else "pixel_l1",
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
