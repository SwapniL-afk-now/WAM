"""NVFP4 rollouts for FastWAM (Sol-RL, "FP4 Explore, BF16 Train", NVlabs 2604.06916).

Ported from ``train_scripts/sol_rl/train_utils.py`` in github.com/NVlabs/Sana
(``sync_lora_to_inference``, ``BF16TELinear``, ``replace_linear_with_te``,
``te.fp8_autocast`` with an E2M1 recipe). Sol-RL keeps a separate low-precision
inference copy with LoRA merged into the base weights; we do the same, but as
*shadow* Transformer-Engine linears attached to the existing DiT linears, and
only switch them on inside ``active()`` during rollouts. The BF16 actor used
for the NFT update is untouched.

RTX PRO 6000 Blackwell (sm_120) has FP4 tensor cores. Requires
``transformer-engine[pytorch]`` (not installed by default; see README).

What does *not* transfer from Sol-RL: its two-stage "score 96 FP4 candidates,
regenerate the most contrastive 24 in BF16" needs a reward model that can
score a sample without acting. In embodied RL the reward comes from executing
the episode, so we only use the FP4 speed-up for the rollouts themselves.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

from imago.lora import LoRALinear

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe as te_recipe

    if hasattr(te_recipe, "NVFP4BlockScaling"):  # TE >= 2.8
        NVFP4_RECIPE = te_recipe.NVFP4BlockScaling()
    else:  # recipe used in Sol-RL's released code
        NVFP4_RECIPE = te_recipe.DelayedScaling(
            fp8_format=te_recipe.Format.E2M1, amax_history_len=16, amax_compute_algo="max"
        )
    _TE_ERROR = None
except (ImportError, OSError, RuntimeError, AttributeError) as exc:  # pragma: no cover
    te = None
    NVFP4_RECIPE = None
    _TE_ERROR = exc


def _merged_weight(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Base weight with LoRA folded in (Sol-RL ``sync_lora_to_inference``)."""
    if isinstance(module, LoRALinear):
        base = module.base
        weight = base.weight.data
        delta = module.delta_weight()  # active adapter, None when disabled
        if delta is not None:
            weight = (weight.float() + delta).to(weight.dtype)
        return weight, base.bias.data if base.bias is not None else None
    return module.weight.data, module.bias.data if module.bias is not None else None


class NVFP4Rollout:
    """Shadow NVFP4 linears for the video / action DiT blocks of a FastWAM policy."""

    def __init__(self, policy: nn.Module, min_dim: int = 512, align: int = 32):
        if te is None:
            raise RuntimeError(
                "rollout_precision=nvfp4 needs transformer-engine[pytorch] "
                f"(import failed: {_TE_ERROR})."
            )
        self.targets: list[nn.Module] = []
        for expert in (policy.video_expert, policy.action_expert):
            for block in expert.blocks:
                for module in block.modules():
                    if isinstance(module, LoRALinear):
                        linear = module.base
                    elif isinstance(module, nn.Linear) and not _inside_lora(block, module):
                        linear = module
                    else:
                        continue
                    i, o = linear.in_features, linear.out_features
                    if max(i, o) <= min_dim or i % align or o % align:
                        continue  # Sol-RL skips small layers; FP4 GEMMs need aligned dims
                    self.targets.append(module)
        self.shadow: dict[int, "te.Linear"] = {}
        self._fingerprint = None

    def _fp(self) -> tuple:
        parts = []
        for m in self.targets:
            if isinstance(m, LoRALinear):
                params = [p for p in m.parameters()]  # base + all adapters
                parts.append((m.active, m.enabled) + tuple(p._version for p in params))
            else:
                parts.append(tuple([m.weight._version]))
        return tuple(parts)

    @torch.no_grad()
    def refresh(self) -> None:
        """Re-merge weights when they changed (weight sync from the actor)."""
        fp = self._fp()
        if fp == self._fingerprint:
            return
        for module in self.targets:
            weight, bias = _merged_weight(module)
            key = id(module)
            lin = self.shadow.get(key)
            if lin is None:
                lin = te.Linear(weight.shape[1], weight.shape[0], bias=bias is not None).to(
                    device=weight.device, dtype=torch.bfloat16
                )
                self.shadow[key] = lin
            lin.weight.copy_(weight)
            if bias is not None:
                lin.bias.copy_(bias)
        self._fingerprint = self._fp()

    @contextmanager
    def active(self):
        self.refresh()
        patched = []
        for module in self.targets:
            if "forward" in module.__dict__:
                continue  # already patched (re-entrant call)
            lin = self.shadow[id(module)]
            module.forward = (lambda l: lambda x: l(x.to(torch.bfloat16)))(lin)
            patched.append(module)
        try:
            with te.fp8_autocast(enabled=True, fp8_recipe=NVFP4_RECIPE):
                yield
        finally:
            for module in patched:
                del module.forward  # back to the class forward (BF16 / LoRA)


def _inside_lora(block: nn.Module, linear: nn.Linear) -> bool:
    for module in block.modules():
        if isinstance(module, LoRALinear) and module.base is linear:
            return True
    return False
