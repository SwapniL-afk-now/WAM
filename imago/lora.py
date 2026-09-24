"""Minimal LoRA for FastWAM DiT blocks.

FastWAM's MoT calls ``block.self_attn.q(x)`` etc. as modules, so wrapping the
``nn.Linear`` layers is enough for both the video-only denoiser
(``FastWAMIDM._denoise_video``) and the cached action path
(``MoT.forward_action_with_video_cache_tensor``).

Kept dependency-free (no peft) so that parameter names are stable
(``...q.lora_A`` / ``...q.lora_B``) for the EMA reference and weight sync.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterable, Iterator

import torch
import torch.nn as nn

# Linear sub-modules inside a Wan ``DiTBlock``.
DEFAULT_TARGETS = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.o",
    "ffn.0",
    "ffn.2",
)


class LoRALinear(nn.Module):
    """``y = W x + b + (alpha / r) * B A x`` with ``B`` zero-initialised."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.enabled = True
        weight = base.weight
        self.lora_A = nn.Parameter(
            torch.empty(rank, base.in_features, device=weight.device, dtype=weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank, device=weight.device, dtype=weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        lora = (x.to(self.lora_A.dtype) @ self.lora_A.t()) @ self.lora_B.t()
        return out + (self.scaling * lora).to(out.dtype)


def _get_submodule(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
    return parent, parts[-1]


def inject_lora(
    blocks: Iterable[nn.Module],
    rank: int,
    alpha: float,
    targets: Iterable[str] = DEFAULT_TARGETS,
) -> int:
    """Wrap the target linears of every block in ``blocks``. Returns #wrapped."""
    wrapped = 0
    for block in blocks:
        for target in targets:
            try:
                parent, leaf = _get_submodule(block, target)
                module = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
            except (AttributeError, IndexError):
                continue
            if isinstance(module, LoRALinear) or not isinstance(module, nn.Linear):
                continue
            lora = LoRALinear(module, rank=rank, alpha=alpha)
            if leaf.isdigit():
                parent[int(leaf)] = lora
            else:
                setattr(parent, leaf, lora)
            wrapped += 1
    return wrapped


def lora_modules(model: nn.Module) -> Iterator[LoRALinear]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module


@contextmanager
def lora_disabled(model: nn.Module):
    """Temporarily evaluate the pretrained (LoRA-free) model."""
    modules = list(lora_modules(model))
    previous = [m.enabled for m in modules]
    for m in modules:
        m.enabled = False
    try:
        yield
    finally:
        for m, flag in zip(modules, previous):
            m.enabled = flag


def tail_blocks(blocks: nn.ModuleList, num_tail: int) -> list[nn.Module]:
    """Last ``num_tail`` blocks (all blocks when ``num_tail`` <= 0)."""
    blocks = list(blocks)
    if num_tail <= 0 or num_tail >= len(blocks):
        return blocks
    return blocks[-num_tail:]
