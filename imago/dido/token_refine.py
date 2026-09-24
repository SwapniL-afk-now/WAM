"""DIDO dynamics-based token refinement for the action expert's video cache.

Paper (arXiv 2609.15570, Sec. 3.3, Fig. 3, Eqs. 6–7):

* score each spatial cell by ``||V_tk(r) - V_t0(r)||_2`` using value features of
  the current frame ``t0`` and a future frame ``tk``;
* group cells into 2×2 regions, score a region by the sum of its cells;
* keep the top regions at full resolution and 2×2-average-pool the rest;
* apply the same refinement to the K and V features the action expert reads;
  the current frame and all other latents are unchanged.

DIDO's 8×10 grid gives a 4×5 region grid and keeps the top 3 of 20. FastWAM's
224×448 two-camera input gives a 7×14 token grid per latent frame. We tile it
with 2×2 regions (3×7 full regions plus a 1×2 edge row) and keep
``keep_ratio`` (default 3/20, as in DIDO) of the *full* regions. Edge regions
are always pooled, so every sample yields the same number of tokens and the
batch stays rectangular.

Every output row is a real token (no masking needed). Implementation: one per-sample pooling matrix ``P [B, L_out, L_in]`` is built
once from the reference layer and applied to every layer's K and V
(``P @ K``). This is cheap because ``L_in`` is only a few hundred tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class TokenRefineConfig:
    enabled: bool = False
    keep_ratio: float = 3.0 / 20.0  # DIDO: top-3 of 20 regions
    region: int = 2  # 2×2 cells per region
    future_frame: int = -1  # latent frame t_k used for the dynamics map (-1 = last); UNSPECIFIED in paper
    ref_layer: int = -1  # layer whose value cache defines the map; UNSPECIFIED in paper


def _regions(h: int, w: int, r: int):
    """List of (cells, is_full) with cells as flat indices into an h×w grid."""
    out = []
    for i0 in range(0, h, r):
        for j0 in range(0, w, r):
            cells = [i * w + j for i in range(i0, min(i0 + r, h)) for j in range(j0, min(j0 + r, w))]
            out.append((cells, len(cells) == r * r))
    return out


def build_pooling_matrix(
    v_ref: torch.Tensor,
    grid: tuple[int, int, int],
    cfg: TokenRefineConfig,
) -> torch.Tensor:
    """``v_ref [B, T*h*w, C]`` -> pooling matrix ``[B, L_out, T*h*w]``.

    Token order is (frame, row, col), as produced by Wan's patchify.
    """
    t_frames, h, w = grid
    bsz, length, _ = v_ref.shape
    if length != t_frames * h * w:
        raise ValueError(f"cache length {length} != grid {grid}")
    per_frame = h * w
    regions = _regions(h, w, cfg.region)
    full_ids = [i for i, (_, full) in enumerate(regions) if full]
    k_keep = max(1, int(round(cfg.keep_ratio * len(full_ids))))

    tk = cfg.future_frame % t_frames
    v0 = v_ref[:, 0:per_frame].float()
    vk = v_ref[:, tk * per_frame : (tk + 1) * per_frame].float()
    cell_score = torch.linalg.vector_norm(vk - v0, dim=-1)  # [B, h*w]  (Eq. 6)
    region_score = torch.stack(
        [cell_score[:, regions[i][0]].sum(dim=1) for i in full_ids], dim=1
    )  # [B, n_full]  (Eq. 7)
    top = region_score.topk(k_keep, dim=1).indices  # indices into full_ids

    # Rows per sample: frame-0 identity, then for every future frame the pooled
    # rows of all *non-kept* regions followed by the cells of the kept regions.
    # The count of non-kept regions is constant (R - k), so L_out is fixed.
    n_reg = len(regions)
    l_out = per_frame + (t_frames - 1) * (n_reg - k_keep + k_keep * cfg.region**2)
    # Filled on CPU (thousands of tiny index writes), moved to the GPU once.
    pool = torch.zeros(bsz, l_out, length, dtype=torch.float32)
    eye = torch.arange(per_frame)
    pool[:, eye, eye] = 1.0
    top_cpu = top.cpu().tolist()
    for b in range(bsz):
        kept = sorted(full_ids[j] for j in top_cpu[b])
        kept_set = set(kept)
        row = per_frame
        for f in range(1, t_frames):
            base = f * per_frame
            for ridx, (cells, _) in enumerate(regions):
                if ridx in kept_set:
                    continue
                pool[b, row, [base + c for c in cells]] = 1.0 / len(cells)  # 2×2 avg pool
                row += 1
            for ridx in kept:
                for c in regions[ridx][0]:
                    pool[b, row, base + c] = 1.0  # full resolution
                    row += 1
        assert row == l_out
    return pool.to(device=v_ref.device, dtype=v_ref.dtype, non_blocking=True)


def refine_cache(
    cache_k: list[torch.Tensor],
    cache_v: list[torch.Tensor],
    grid: tuple[int, int, int],
    cfg: TokenRefineConfig,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Apply DIDO refinement to every layer's K and V. Returns (K', V')."""
    pool = build_pooling_matrix(cache_v[cfg.ref_layer].detach(), grid, cfg)
    new_k = [torch.bmm(pool.to(k.dtype), k) for k in cache_k]
    new_v = [torch.bmm(pool.to(v.dtype), v) for v in cache_v]
    return new_k, new_v


def refined_length(grid: tuple[int, int, int], cfg: TokenRefineConfig) -> int:
    t_frames, h, w = grid
    regions = _regions(h, w, cfg.region)
    n_full = sum(1 for _, full in regions if full)
    k_keep = max(1, int(round(cfg.keep_ratio * n_full)))
    return h * w + (t_frames - 1) * (len(regions) - k_keep + k_keep * cfg.region**2)


def _selftest() -> None:  # pragma: no cover - documentation of shapes
    grid = (3, 7, 14)
    v = torch.randn(2, math.prod(grid), 8)
    cfg = TokenRefineConfig(enabled=True)
    k2, v2 = refine_cache([v], [v], grid, cfg)
    assert k2[0].shape[1] == refined_length(grid, cfg)
