"""Full-parameter FSDP2 helpers for FastWAM on 2–3 × 96 GB.

FastWAM's MoT reaches into expert blocks directly (``block.self_attn.q(...)``),
so per-block FSDP units would never be gathered. We therefore follow RLinf's
FastWAM SFT recipe (``examples/sft/config/libero_sft_fastwam.yaml``: FSDP2,
root-only sharding). Each model is one FSDP2 unit wrapped in a tiny
``Runner`` whose ``forward`` dispatches to the method we need. Calling the
runner gathers the parameters and re-shards them after forward and backward.

Precision: fp32 sharded master weights, bf16 compute (``MixedPrecisionPolicy``),
fp32 gradient reduce, fp32 AdamW states. This is the full-fine-tuning setup
FastWAM/DIDO use (bf16 mixed precision).

2-GPU fallback: ``offload=True`` (``CPUOffloadPolicy``) keeps a unit's shards,
gradients and optimizer step on CPU. Use it for rarely-stepped units (the DIDO
generator, which updates every 5th iteration).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.distributed as dist
import torch.nn as nn


class Runner(nn.Module):
    """``runner(fn_name, *args, **kw)`` → ``getattr(target, fn_name)(*args, **kw)``,
    or a free function ``fn(target, ...)`` when ``fn_name`` is callable."""

    def __init__(self, target: nn.Module):
        super().__init__()
        self.target = target

    def forward(self, fn: str | Callable, *args, **kwargs):
        if callable(fn):
            return fn(self.target, *args, **kwargs)
        return getattr(self.target, fn)(*args, **kwargs)


def shard(
    module: nn.Module,
    *,
    trainable: bool,
    offload: bool = False,
    reshard_after_forward: bool = True,
) -> Runner:
    """Wrap ``module`` in a ``Runner`` and shard it as a single FSDP2 unit."""
    runner = Runner(module)
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return runner
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    if trainable:
        runner.float()  # fp32 master weights; compute happens in bf16 below
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    kwargs = {"mp_policy": mp, "reshard_after_forward": reshard_after_forward}
    if offload:
        kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=True)
    fully_shard(runner, **kwargs)
    return runner


def estimate_gb(params: int, trainable: bool, world: int, offload: bool = False) -> float:
    """Persistent GB per GPU for one sharded unit (excluding activations and the
    transient gathered bf16 copy, which is ``2 * params`` bytes during forward/backward)."""
    if offload:
        return 0.0
    per_param = (4 + 4 + 8) if trainable else 2  # fp32 master + fp32 grad + Adam m,v | bf16 frozen
    return params * per_param / world / 1e9


def report_memory(units: dict[str, tuple[int, bool, bool]], world: int, rank: int) -> None:
    if rank != 0:
        return
    total = 0.0
    lines = []
    for name, (n, trainable, offload) in units.items():
        gb = estimate_gb(n, trainable, world, offload)
        total += gb
        lines.append(f"  {name}: {n / 1e9:.2f}B params, trainable={trainable}, offload={offload} -> {gb:.1f} GB/GPU")
    peak_gather = max(2 * n for n, _, _ in units.values()) / 1e9
    print("[fsdp] persistent memory estimate:\n" + "\n".join(lines)
          + f"\n  total {total:.1f} GB/GPU + transient gather ~{peak_gather:.1f} GB + activations", flush=True)


def num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
