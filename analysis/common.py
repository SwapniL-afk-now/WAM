"""Shared helpers for IMAGO analyses on eval recordings (``imago.record_dir``)."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Call:
    path: str
    real_frame: np.ndarray  # [B,3,H,W] uint8, observation the chunk was planned from
    imagined: np.ndarray  # [B,3,T,H,W] uint8 decoded imagined video
    x0_video: np.ndarray
    x0_action: np.ndarray
    first_latents: np.ndarray
    proprio: np.ndarray
    prompt_id: np.ndarray
    task: np.ndarray
    num_action_chunks: int
    freq_ratio: int


def load_calls(record_dir: str, rank: int = 0) -> list[Call]:
    paths = sorted(glob.glob(os.path.join(record_dir, f"r{rank}_call*.npz")))
    calls = []
    for path in paths:
        d = np.load(path, allow_pickle=False)
        calls.append(
            Call(
                path=path,
                real_frame=d["real_frame"],
                imagined=d["imagined"],
                x0_video=d["x0_video"],
                x0_action=d["x0_action"],
                first_latents=d["first_latents"],
                proprio=d["proprio"],
                prompt_id=d["prompt_id"],
                task=d["task"],
                num_action_chunks=int(d["num_action_chunks"]),
                freq_ratio=int(d["action_video_freq_ratio"]),
            )
        )
    return calls


def consecutive_pairs(calls: list[Call], max_offset: int = 1):
    """Yield ``(call_c, env_i, offset_m, call_{c+m})`` for the same env slot.

    Consecutive eval calls on one rank are the same env batch advanced by
    ``num_action_chunks`` actions. Pairs whose task string changes (an episode
    reset in between) are dropped; the few resets that keep the task are left
    in and handled by reporting medians.
    """
    for c in range(len(calls)):
        for m in range(1, max_offset + 1):
            if c + m >= len(calls):
                break
            a, b = calls[c], calls[c + m]
            if a.real_frame.shape[0] != b.real_frame.shape[0]:
                continue
            for i in range(a.real_frame.shape[0]):
                if all(calls[c + k].task[i] == a.task[i] for k in range(1, m + 1)):
                    yield a, i, m, b


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0**2 / max(mse, 1e-10)))


def l1(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x.astype(np.float64) - y.astype(np.float64))) / 255.0)
