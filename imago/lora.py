"""Minimal multi-adapter LoRA for FastWAM DiT blocks.

FastWAM's MoT calls ``block.self_attn.q(x)`` etc. as modules, so wrapping the
``nn.Linear`` layers is enough for both the video-only denoiser
(``FastWAMIDM._denoise_video``) and the cached action path
(``MoT.forward_action_with_video_cache_tensor``).

Adapters:

* ``"default"`` (params ``...lora_A`` / ``...lora_B``): the IMAGO RL adapter.
* named extra adapters (params ``...extra.<name>.A`` / ``.B``): used by the
  DIDO-style distillation, where the one-step student (``"gen"``) and the fake
  score network (``"fake"``) are two adapters on the *same* frozen base, and
  the teacher is the base itself (all adapters off). One 5B copy per GPU
  instead of three.

Kept dependency-free (no peft) so parameter names are stable for the EMA
reference, weight sync and merging.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional

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


class _Adapter(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float, ref: torch.Tensor):
        super().__init__()
        self.scaling = float(alpha) / float(rank)
        self.A = nn.Parameter(torch.empty(rank, in_features, device=ref.device, dtype=ref.dtype))
        self.B = nn.Parameter(torch.zeros(out_features, rank, device=ref.device, dtype=ref.dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))


class LoRALinear(nn.Module):
    """``y = W x + b + (alpha / r) * B A x`` for the *active* adapter (B zero-init)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, with_default: bool = True):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.enabled = True  # global switch (``lora_disabled``)
        self.active: Optional[str] = "default" if with_default else None
        self.extra = nn.ModuleDict()
        weight = base.weight
        if with_default:
            self.lora_A = nn.Parameter(
                torch.empty(rank, base.in_features, device=weight.device, dtype=weight.dtype)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(base.out_features, rank, device=weight.device, dtype=weight.dtype)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.lora_A = None
            self.lora_B = None

    # nn.Linear-like surface
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

    def add_adapter(self, name: str, rank: int, alpha: float) -> None:
        if name == "default" or name in self.extra:
            raise ValueError(f"Adapter {name!r} already exists.")
        self.extra[name] = _Adapter(self.in_features, self.out_features, rank, alpha, self.base.weight)

    def _factors(self):
        if not self.enabled or self.active is None:
            return None
        if self.active == "default":
            if self.lora_A is None:
                return None
            return self.lora_A, self.lora_B, self.scaling
        ad = self.extra[self.active]
        return ad.A, ad.B, ad.scaling

    def delta_weight(self) -> torch.Tensor | None:
        f = self._factors()
        if f is None:
            return None
        a, b, s = f
        return s * (b.float() @ a.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        f = self._factors()
        if f is None:
            return out
        a, b, s = f
        lora = (x.to(a.dtype) @ a.t()) @ b.t()
        return out + (s * lora).to(out.dtype)


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
    with_default: bool = True,
) -> int:
    """Wrap the target linears of every block. Returns the number wrapped.

    ``with_default=False`` wraps without the IMAGO ``default`` adapter (for
    distillation, which only adds named adapters).
    """
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
            lora = LoRALinear(module, rank=rank, alpha=alpha, with_default=with_default)
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


def add_adapter(model: nn.Module, name: str, rank: int, alpha: float) -> int:
    count = 0
    for m in lora_modules(model):
        m.add_adapter(name, rank, alpha)
        count += 1
    return count


def adapter_parameters(model: nn.Module, name: str) -> list[nn.Parameter]:
    params = []
    for m in lora_modules(model):
        if name == "default":
            if m.lora_A is not None:
                params += [m.lora_A, m.lora_B]
        elif name in m.extra:
            params += [m.extra[name].A, m.extra[name].B]
    return params


def set_active_adapter(model: nn.Module, name: Optional[str]) -> None:
    for m in lora_modules(model):
        m.active = name


@contextmanager
def use_adapter(model: nn.Module, name: Optional[str]):
    """Temporarily activate adapter ``name`` (``None`` = base / teacher)."""
    modules = list(lora_modules(model))
    previous = [m.active for m in modules]
    for m in modules:
        m.active = name
    try:
        yield
    finally:
        for m, prev in zip(modules, previous):
            m.active = prev


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


@torch.no_grad()
def merge_adapter(model: nn.Module, name: str) -> int:
    """Fold adapter ``name`` into the base weights (e.g. the distilled one-step student)."""
    merged = 0
    for m in lora_modules(model):
        prev = m.active
        m.active = name
        delta = m.delta_weight()
        m.active = prev
        if delta is not None:
            m.base.weight.add_(delta.to(m.base.weight.dtype))
            merged += 1
    return merged


def unwrap_lora(model: nn.Module) -> int:
    """Replace every ``LoRALinear`` by its base ``nn.Linear`` (after merging)."""
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                if child_name.isdigit():
                    module[int(child_name)] = child.base
                else:
                    setattr(module, child_name, child.base)
                count += 1
    return count


def tail_blocks(blocks: nn.ModuleList, num_tail: int) -> list[nn.Module]:
    """Last ``num_tail`` blocks (all blocks when ``num_tail`` <= 0)."""
    blocks = list(blocks)
    if num_tail <= 0 or num_tail >= len(blocks):
        return blocks
    return blocks[-num_tail:]
