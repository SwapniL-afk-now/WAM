"""Imagination drift: how *true* is the imagined future, per checkpoint?

For each eval call the policy imagined ``T`` frames and then executed
``num_action_chunks`` actions. The next call's observation is the real frame at
imagined index ``num_action_chunks / action_video_freq_ratio`` (= 2 with the
default 8 actions / 4 actions-per-frame). We compare them.

Plot this against success over training: IMAGO's thesis predicts success can
keep rising while realism (PSNR) falls once ``imago_beta_real`` is ~0.

    python analysis/drift.py --record-dirs run_a/rec_step0 run_a/rec_step100 --out drift.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from common import consecutive_pairs, l1, load_calls, psnr


def drift_for_dir(record_dir: str, rank: int) -> dict:
    calls = load_calls(record_dir, rank)
    psnrs, l1s, copy_psnrs = [], [], []
    for a, i, _m, b in consecutive_pairs(calls, max_offset=1):
        k = a.num_action_chunks // a.freq_ratio
        if k >= a.imagined.shape[2]:
            continue
        imagined = a.imagined[i, :, k]
        real_next = b.real_frame[i]
        psnrs.append(psnr(imagined, real_next))
        l1s.append(l1(imagined, real_next))
        # Baseline: "nothing moves" (copy current frame) -- imagination should beat it.
        copy_psnrs.append(psnr(a.real_frame[i], real_next))
    if not psnrs:
        return {"record_dir": record_dir, "pairs": 0}
    return {
        "record_dir": record_dir,
        "pairs": len(psnrs),
        "psnr_median": float(np.median(psnrs)),
        "l1_median": float(np.median(l1s)),
        "copy_baseline_psnr_median": float(np.median(copy_psnrs)),
        "psnr_gain_over_copy": float(np.median(np.array(psnrs) - np.array(copy_psnrs))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dirs", nargs="+", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    results = [drift_for_dir(d, args.rank) for d in args.record_dirs]
    for r in results:
        print(json.dumps(r))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
