"""DIDO-style automatic interaction annotation for FastWAM's LIBERO LeRobot data.

Paper, App. A:
* gripper boxes = the simulator pose projected into the image;
* object boxes = the target parsed from the task, an open-vocabulary detector
  on a reference frame, and a video tracker in both directions (keeping the
  higher-confidence box), with gaps filled by linear interpolation;
* DINOv3 targets = the object crop (enlarged by a margin) encoded by frozen
  DINOv3.

Our choices where the paper is unspecific (verify with the audit script):
* The target object comes from the task BDDL ``(:obj_of_interest ...)``, which
  is exact for LIBERO, instead of parsing the instruction.
* Gripper: an axis-aligned 3D box around ``robot0_eef_pos`` (the first 3 dims of
  the LIBERO state) with half-extent ``--grip-extent``, projected with the
  agentview camera matrix of the task scene. OpenVLA-style LIBERO images are
  stored rotated 180°; ``--image-transform`` (default ``flip180``) maps the raw
  render to the stored orientation.
* DINOv3 ViT-L/16 patch tokens, average-pooled to 4×4 = 16 tokens per frame.

Server-only dependencies: ``libero``/``liberoplus``, ``robosuite``,
``transformers`` (Grounding DINO, DINOv3), ``sam2``, ``av``, ``pyarrow``.

    python scripts/annotate_libero.py --dataset-dir $LIBERO_DATA_ROOT/libero_spatial_no_noops_lerobot \
        --suite libero_spatial --out $IMAGO_CKPT_DIR/annotations/libero [--shard 0/3]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------- LeRobot I/O
def load_info(dataset_dir: Path) -> dict:
    return json.loads((dataset_dir / "meta" / "info.json").read_text())


def load_episodes(dataset_dir: Path) -> list[dict]:
    path = dataset_dir / "meta" / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_states(dataset_dir: Path, info: dict, episode: int, state_key: str) -> np.ndarray:
    import pyarrow.parquet as pq

    chunk = episode // int(info.get("chunks_size", 1000))
    path = dataset_dir / info["data_path"].format(episode_chunk=chunk, episode_index=episode)
    table = pq.read_table(path, columns=[state_key])
    return np.stack(table.column(state_key).to_pylist()).astype(np.float64)


def read_frames(dataset_dir: Path, info: dict, episode: int, video_key: str) -> np.ndarray:
    import av

    chunk = episode // int(info.get("chunks_size", 1000))
    path = dataset_dir / info["video_path"].format(episode_chunk=chunk, video_key=video_key, episode_index=episode)
    with av.open(str(path)) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


# -------------------------------------------------------------- LIBERO scene
def task_lookup(suite: str, libero_type: str):
    if libero_type == "plus":
        from liberoplus.liberoplus import benchmark, get_libero_path
    else:
        from libero.libero import benchmark, get_libero_path
    bench = benchmark.get_benchmark_dict()[suite]()
    tasks = {}
    for i in range(bench.n_tasks):
        task = bench.get_task(i)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        tasks[str(task.language).strip().lower()] = bddl
    return tasks


def target_object(bddl: Path) -> tuple[str, str]:
    text = bddl.read_text()
    m = re.search(r"\(:obj_of_interest\s+([^)]*)\)", text)
    name = m.group(1).split()[0] if m else ""
    phrase = re.sub(r"_\d+$", "", name).replace("_", " ").strip()
    return name, phrase


def camera_matrix(bddl: Path, height: int, width: int, libero_type: str, camera: str = "agentview"):
    if libero_type == "plus":
        from liberoplus.liberoplus.envs import OffScreenRenderEnv
    else:
        from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=height, camera_widths=width)
    env.reset()
    mat = get_camera_transform_matrix(env.sim, camera, height, width)
    env.close()
    return mat


def project_box(mat, center, extent, height, width, transform: str):
    """Project an axis-aligned 3D box to a normalised xyxy box in the *stored* image.

    ``robosuite.utils.camera_utils.project_points_from_world_to_camera`` uses
    ``K @ p``, divides by depth and takes (x, y) as (col, row) of the raw MuJoCo
    render. ``transform`` maps raw-render pixels to how the dataset stores
    frames (OpenVLA-style LIBERO data is rotated 180°). Verify with
    ``scripts/audit_annotations.py`` before training.
    """
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * extent + center
    homo = np.concatenate([corners, np.ones((8, 1))], axis=1) @ mat.T
    pix = homo[:, :2] / homo[:, 2:3]
    cols, rows = pix[:, 0], pix[:, 1]
    if transform in ("flip180", "hflip"):
        cols = width - 1 - cols
    if transform in ("flip180", "vflip"):
        rows = height - 1 - rows
    box = np.array([cols.min(), rows.min(), cols.max(), rows.max()])
    box = np.clip(box / np.array([width, height, width, height]), 0.0, 1.0)
    valid = bool(box[2] - box[0] > 1e-3 and box[3] - box[1] > 1e-3)
    return box, valid


# ------------------------------------------------------------ detect / track
class ObjectTracker:
    def __init__(self, device: str, detector: str, sam2_cfg: str, sam2_ckpt: str):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from sam2.build_sam import build_sam2_video_predictor

        self.device = device
        self.proc = AutoProcessor.from_pretrained(detector)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(detector).to(device).eval()
        self.sam = build_sam2_video_predictor(sam2_cfg, sam2_ckpt, device=device)

    @torch.no_grad()
    def detect(self, frame: np.ndarray, phrase: str):
        inputs = self.proc(images=frame, text=f"{phrase}.", return_tensors="pt").to(self.device)
        out = self.det(**inputs)
        res = self.proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=0.25, text_threshold=0.2, target_sizes=[frame.shape[:2]]
        )[0]
        if len(res["scores"]) == 0:
            return None, 0.0
        best = int(res["scores"].argmax())
        return res["boxes"][best].cpu().numpy(), float(res["scores"][best])

    @torch.no_grad()
    def track(self, frames: np.ndarray, box: np.ndarray, start: int, reverse: bool):
        """Returns per-frame (box_xyxy_pixels or None, score)."""
        import tempfile
        from PIL import Image

        n = len(frames)
        boxes = [None] * n
        scores = np.zeros(n)
        with tempfile.TemporaryDirectory() as tmp:
            for i, f in enumerate(frames):
                Image.fromarray(f).save(f"{tmp}/{i:05d}.jpg", quality=95)
            state = self.sam.init_state(video_path=tmp)
            self.sam.add_new_points_or_box(state, frame_idx=start, obj_id=1, box=box)
            for fidx, _ids, logits in self.sam.propagate_in_video(state, start_frame_idx=start, reverse=reverse):
                mask = (logits[0, 0] > 0).cpu().numpy()
                if mask.any():
                    ys, xs = np.nonzero(mask)
                    boxes[fidx] = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
                    scores[fidx] = float(torch.sigmoid(logits[0, 0][logits[0, 0] > 0]).mean())
        return boxes, scores


def interpolate(boxes: list, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(boxes)
    out = np.zeros((n, 4))
    idx = np.nonzero(valid)[0]
    if len(idx) == 0:
        return out, valid
    for k in range(4):
        out[:, k] = np.interp(np.arange(n), idx, np.array([boxes[i][k] for i in idx]))
    filled = np.zeros(n, dtype=bool)
    filled[idx.min() : idx.max() + 1] = True  # interpolate inside, never extrapolate
    return out, filled


# ------------------------------------------------------------------- DINOv3
class DinoEncoder:
    def __init__(self, device: str, model_id: str, margin: float):
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval().half()
        self.margin = margin
        self.num_registers = int(getattr(self.model.config, "num_register_tokens", 0))

    @torch.no_grad()
    def encode(self, frames: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        from PIL import Image

        h, w = frames.shape[1:3]
        crops = []
        for f, b in zip(frames, boxes):
            x0, y0, x1, y1 = b * np.array([w, h, w, h])
            mx, my = (x1 - x0) * self.margin, (y1 - y0) * self.margin
            x0, y0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
            x1, y1 = min(w, int(x1 + mx) + 1), min(h, int(y1 + my) + 1)
            crop = f[y0:y1, x0:x1] if (x1 > x0 and y1 > y0) else f
            crops.append(Image.fromarray(crop).resize((224, 224)))
        feats = []
        for s in range(0, len(crops), 64):
            inputs = self.proc(images=crops[s : s + 64], return_tensors="pt", do_resize=False, do_center_crop=False)
            out = self.model(pixel_values=inputs["pixel_values"].to(self.device).half()).last_hidden_state
            patches = out[:, 1 + self.num_registers :]  # drop CLS + registers
            side = int(round(patches.shape[1] ** 0.5))
            grid = patches.transpose(1, 2).reshape(patches.shape[0], -1, side, side).float()
            pooled = torch.nn.functional.adaptive_avg_pool2d(grid, 4).flatten(2).transpose(1, 2)  # [n,16,D]
            feats.append(pooled.half().cpu().numpy())
        return np.concatenate(feats)


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--libero-type", default="standard", choices=["standard", "plus"])
    ap.add_argument("--video-key", default="observation.images.image")  # agentview in FastWAM LIBERO data
    ap.add_argument("--state-key", default="observation.state")
    ap.add_argument("--grip-extent", type=float, nargs=3, default=[0.04, 0.04, 0.05])
    ap.add_argument("--image-transform", default="flip180", choices=["none", "flip180", "vflip", "hflip"],
                    help="raw MuJoCo render -> stored frame orientation (check with the audit)")
    ap.add_argument("--detector", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--sam2-ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--dino", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--crop-margin", type=float, default=0.2)  # "enlarged by a fixed margin" (unspecified)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ddir = Path(args.dataset_dir)
    out_dir = Path(args.out) / ddir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    info = load_info(ddir)
    episodes = load_episodes(ddir)
    shard, nshard = (int(x) for x in args.shard.split("/"))
    episodes = episodes[shard::nshard]
    tasks = task_lookup(args.suite, args.libero_type)
    tracker = ObjectTracker(args.device, args.detector, args.sam2_cfg, args.sam2_ckpt)
    dino = DinoEncoder(args.device, args.dino, args.crop_margin)
    cam_cache: dict = {}

    for ep in episodes:
        eidx = int(ep["episode_index"])
        path = out_dir / f"episode_{eidx:06d}.npz"
        if path.exists():
            continue
        instruction = str(ep["tasks"][0]).strip().lower()
        bddl = tasks.get(instruction)
        if bddl is None:
            print(f"[annotate] no BDDL for task {instruction!r}; skipping episode {eidx}")
            continue
        frames = read_frames(ddir, info, eidx, args.video_key)  # stored orientation
        states = read_states(ddir, info, eidx, args.state_key)
        n, h, w = len(frames), frames.shape[1], frames.shape[2]
        n = min(n, len(states))
        frames, states = frames[:n], states[:n]

        # Gripper: simulator state projected into the image.
        key = (str(bddl), h, w)
        if key not in cam_cache:
            cam_cache[key] = camera_matrix(bddl, h, w, args.libero_type)
        grip = [project_box(cam_cache[key], s[:3], np.array(args.grip_extent), h, w, args.image_transform)
            for s in states]
        grip_boxes = np.stack([g[0] for g in grip])
        grip_valid = np.array([g[1] for g in grip])

        # Object: detect + track on upright frames, map back to stored orientation.
        upright = frames[:, ::-1, ::-1].copy() if args.image_transform == "flip180" else frames
        _, phrase = target_object(bddl)
        box0, s0 = tracker.detect(upright[0], phrase)
        boxes_f, scores_f = ([None] * n, np.zeros(n)) if box0 is None else tracker.track(upright, box0, 0, False)
        boxl, sl = tracker.detect(upright[-1], phrase)
        boxes_b, scores_b = ([None] * n, np.zeros(n)) if boxl is None else tracker.track(upright, boxl, n - 1, True)
        best = [bf if (bf is not None and (sb <= sf or bb is None)) else bb
                for bf, bb, sf, sb in zip(boxes_f, boxes_b, scores_f, scores_b)]
        valid = np.array([b is not None for b in best])
        obj_px, obj_valid = interpolate(best, valid)
        obj = obj_px / np.array([w, h, w, h])
        if args.image_transform == "flip180":  # upright -> stored (rotate 180°)
            obj = np.stack([1 - obj[:, 2], 1 - obj[:, 3], 1 - obj[:, 0], 1 - obj[:, 1]], axis=1)
        obj = np.clip(obj, 0, 1)

        feats = dino.encode(frames, obj)  # crops in stored orientation, as the model sees them
        np.savez_compressed(
            path,
            object_boxes=obj.astype(np.float32),
            object_valid=obj_valid,
            gripper_boxes=grip_boxes.astype(np.float32),
            gripper_valid=grip_valid,
            dino=feats.astype(np.float16),
            phrase=np.array(phrase),
            detect_scores=np.array([s0, sl], dtype=np.float32),
        )
        print(f"[annotate] {ddir.name} ep {eidx}: {n} frames, object valid {obj_valid.mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
