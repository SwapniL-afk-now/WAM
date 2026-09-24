"""FastWAM LIBERO dataset + DIDO interaction targets.

Wraps ``fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset``, so the
video, action, proprio and text-embedding handling is exactly FastWAM's, and
attaches the targets produced by ``scripts/annotate_libero.py``:

* ``boxes_obj`` / ``boxes_grip`` ``[H_a, 4]``: normalised xyxy boxes (in the
  agentview image) for the next ``H_a`` action steps; ``*_valid`` ``[H_a]``;
* ``dino_target`` ``[16, D]``: 4×4-pooled DINOv3 tokens of the object crop at
  frame ``t0 + dino_offset``; ``dino_valid``.

The sample's (dataset, episode, frame) indices are captured by wrapping the
base dataset's ``_split_lerobot_sample``. That runs on every load, including
FastWAM's padding re-draws, so the stored indices always belong to the
sample actually returned.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


def _int(x) -> int:
    return int(x.item() if torch.is_tensor(x) else x)


class InteractionAnnotatedDataset(RobotVideoDataset):
    def __init__(self, *args, annotation_root: str, horizon: int = 32, dino_offset: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotation_root = Path(os.path.expanduser(annotation_root))
        self.horizon = int(horizon)
        self.dino_offset = int(dino_offset)
        base = self.lerobot_dataset
        self._dataset_names = [Path(d).name for d in base.dataset_dirs]
        original = base._split_lerobot_sample

        def split_and_record(lerobot_sample):
            base._imago_index = (
                _int(lerobot_sample.get("dataset_index", 0)),
                _int(lerobot_sample["episode_index"]),
                _int(lerobot_sample["frame_index"]),
            )
            return original(lerobot_sample)

        base._split_lerobot_sample = split_and_record

    @staticmethod
    @lru_cache(maxsize=256)
    def _load(path: str):
        with np.load(path) as d:
            return {k: d[k] for k in d.files}

    def _targets(self, dataset_idx: int, episode: int, frame: int) -> dict:
        path = self.annotation_root / self._dataset_names[dataset_idx] / f"episode_{episode:06d}.npz"
        h = self.horizon
        out = {
            "boxes_obj": torch.zeros(h, 4),
            "boxes_grip": torch.zeros(h, 4),
            "boxes_obj_valid": torch.zeros(h, dtype=torch.bool),
            "boxes_grip_valid": torch.zeros(h, dtype=torch.bool),
            "dino_target": None,
            "dino_valid": torch.tensor(False),
        }
        if not path.exists():
            return out
        ann = self._load(str(path))
        num = ann["object_boxes"].shape[0]
        steps = np.arange(frame + 1, frame + 1 + h)
        inside = steps < num
        idx = np.clip(steps, 0, num - 1)
        for key, src, valid_key in (
            ("boxes_obj", "object_boxes", "object_valid"),
            ("boxes_grip", "gripper_boxes", "gripper_valid"),
        ):
            out[key] = torch.from_numpy(ann[src][idx].astype(np.float32))
            out[f"{key}_valid"] = torch.from_numpy(ann[valid_key][idx].astype(bool) & inside)
        dframe = min(frame + self.dino_offset, num - 1)
        out["dino_target"] = torch.from_numpy(ann["dino"][dframe].astype(np.float32))
        out["dino_valid"] = torch.tensor(bool(ann["object_valid"][dframe]))
        return out

    def _get(self, idx):
        data = super()._get(idx)
        dataset_idx, episode, frame = self.lerobot_dataset._imago_index
        targets = self._targets(dataset_idx, episode, frame)
        if targets["dino_target"] is None:  # missing annotation file: zero, masked out
            targets["dino_target"] = torch.zeros(16, int(os.environ.get("IMAGO_DINO_DIM", 1024)))
        data.update(targets)
        return data
