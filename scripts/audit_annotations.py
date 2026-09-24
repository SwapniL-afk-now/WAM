"""Render annotated boxes for random trajectories (DIDO audits 100 on LIBERO).

Draws object (green) and gripper (red) boxes on 4 frames per episode and writes
one PNG per episode. Count correct / drift / failure by eye, as the paper did.

    python scripts/audit_annotations.py --dataset-dir $LIBERO_DATA_ROOT/libero_spatial_no_noops_lerobot \
        --ann $IMAGO_CKPT_DIR/annotations/libero --out audit/ --num 100
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from annotate_libero import load_info, read_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--video-key", default="observation.images.image")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ddir = Path(args.dataset_dir)
    info = load_info(ddir)
    files = sorted((Path(args.ann) / ddir.name).glob("episode_*.npz"))
    random.Random(args.seed).shuffle(files)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in files[: args.num]:
        eidx = int(path.stem.split("_")[1])
        ann = np.load(path)
        frames = read_frames(ddir, info, eidx, args.video_key)
        n = min(len(frames), len(ann["object_boxes"]))
        tiles = []
        for t in np.linspace(0, n - 1, 4).astype(int):
            img = Image.fromarray(frames[t])
            draw = ImageDraw.Draw(img)
            w, h = img.size
            for boxes, valid, color in (
                (ann["object_boxes"], ann["object_valid"], (0, 255, 0)),
                (ann["gripper_boxes"], ann["gripper_valid"], (255, 0, 0)),
            ):
                if valid[t]:
                    b = boxes[t] * np.array([w, h, w, h])
                    draw.rectangle(list(b), outline=color, width=2)
            tiles.append(np.asarray(img))
        Image.fromarray(np.concatenate(tiles, axis=1)).save(out / f"{ddir.name}_ep{eidx:06d}.png")
    print(f"[audit] wrote {min(args.num, len(files))} images to {out}")


if __name__ == "__main__":
    main()
