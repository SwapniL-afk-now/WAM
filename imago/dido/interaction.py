"""DIDO interaction-centric tokens, box heads and multi-layer DINOv3 alignment.

arXiv 2609.15570, Sec. 3.2 and App. B.2:

* 64 learned tokens in 4 groups of 16: ``T_obj, T_int, T_grip, T_align``,
  inserted into the video transformer and evolving jointly with it.
* ``[T_obj; T_int]`` → object boxes and ``[T_grip; T_int]`` → gripper boxes.
  Each head is a 2-layer MLP (GELU, hidden = video width) predicting
  normalised boxes for ``H_a = 32`` future steps.
* ``L_box = mean_h [ smoothL1(β=1) + λ_giou · (1 - GIoU) ]`` (Eq. 3) and
  ``L_inter = λ_obj L_box(obj) + λ_grip L_box(grip)`` (Eq. 4).
* ``L_align = mean_{l∈S} || P_l(T_align^l) - F_DINO ||_1`` (Eq. 5).

The paper leaves these choices open, so they are ours:

* Tokens use identity RoPE and the noisy-frame timestep modulation.
* Attention is bidirectional with the video, except that frame 0 (the
  observation) does not attend to them, keeping FastWAM's first-frame-causal
  rule.
* Each head reads the mean of its two token groups and outputs sigmoid
  ``(cx, cy, w, h)``, converted to ``xyxy``.
* The 16 alignment tokens are matched to DINOv3 patch tokens average-pooled
  to 4×4 (16 targets). ``S`` defaults to layers {10, 20, 29}.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP = 16
NUM_TOKENS = 4 * GROUP
SLICES = {
    "obj": slice(0, GROUP),
    "int": slice(GROUP, 2 * GROUP),
    "grip": slice(2 * GROUP, 3 * GROUP),
    "align": slice(3 * GROUP, 4 * GROUP),
}


@dataclass
class InteractionConfig:
    enabled: bool = False
    horizon: int = 32  # H_a
    align_layers: list = field(default_factory=lambda: [10, 20, 29])  # S (unspecified in paper)
    dino_dim: int = 1024  # DINOv3 ViT-L/16
    lambda_obj: float = 0.03
    lambda_grip: float = 0.03
    lambda_giou: float = 0.5
    lambda_align: float = 0.01


class _BoxHead(nn.Module):
    def __init__(self, dim: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, horizon * 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cxcywh = torch.sigmoid(self.net(x).float()).view(x.shape[0], self.horizon, 4)
        cx, cy, w, h = cxcywh.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1).clamp(0, 1)


class InteractionTokens(nn.Module):
    """Attached as ``video_expert.imago_interaction`` so it is saved with the MoT."""

    def __init__(self, dim: int, num_layers: int, cfg: InteractionConfig):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.tokens = nn.Parameter(torch.randn(NUM_TOKENS, dim) * 0.02)
        self.box_obj = _BoxHead(dim, cfg.horizon)
        self.box_grip = _BoxHead(dim, cfg.horizon)
        self.align_layers = [l % num_layers for l in cfg.align_layers]
        self.proj = nn.ModuleDict({str(l): nn.Linear(dim, cfg.dino_dim) for l in self.align_layers})

    def forward(self, batch: int) -> torch.Tensor:
        # Called as a module so FSDP hooks gather ``tokens`` before use.
        return self.tokens.unsqueeze(0).expand(batch, -1, -1)

    def extend_sequence(
        self,
        x_tok: torch.Tensor,  # [B, L, D]
        t_mod: torch.Tensor,  # [B, L, 6, D] per-token modulation
        context_mask: torch.Tensor,  # [B, L, C]
        freqs: torch.Tensor,  # [L, 1, R] complex
        self_mask: torch.Tensor,  # [L, L] bool
        tokens_per_frame: int,
    ):
        bsz, length = x_tok.shape[:2]
        tok = self(bsz).to(x_tok.dtype)
        mod_row = min(tokens_per_frame, length - 1)  # first noisy-frame token's modulation
        t_mod_tok = t_mod[:, mod_row : mod_row + 1].expand(-1, NUM_TOKENS, -1, -1)
        cmask_tok = context_mask[:, :1].expand(-1, NUM_TOKENS, -1)
        freqs_tok = torch.ones_like(freqs[:1]).expand(NUM_TOKENS, -1, -1)  # identity RoPE
        total = length + NUM_TOKENS
        mask = torch.zeros((total, total), dtype=torch.bool, device=self_mask.device)
        mask[:length, :length] = self_mask
        mask[length:, :] = True  # tokens see everything
        mask[tokens_per_frame:length, length:] = True  # future frames see tokens; frame 0 does not
        return (
            torch.cat([x_tok, tok], dim=1),
            torch.cat([t_mod, t_mod_tok], dim=1),
            torch.cat([context_mask, cmask_tok], dim=1),
            torch.cat([freqs, freqs_tok], dim=0),
            mask,
        )

    # ---------------------------------------------------------------- losses
    def predict_boxes(self, tok_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inter = tok_out[:, SLICES["int"]]
        obj = torch.cat([tok_out[:, SLICES["obj"]], inter], dim=1).mean(dim=1)
        grip = torch.cat([tok_out[:, SLICES["grip"]], inter], dim=1).mean(dim=1)
        return self.box_obj(obj), self.box_grip(grip)

    def interaction_loss(self, tok_out, boxes_obj, boxes_grip, valid_obj, valid_grip, lam=None):
        c = lam or self.cfg
        pred_obj, pred_grip = self.predict_boxes(tok_out)
        l_obj = box_loss(pred_obj, boxes_obj, valid_obj, c.lambda_giou)
        l_grip = box_loss(pred_grip, boxes_grip, valid_grip, c.lambda_giou)
        return c.lambda_obj * l_obj + c.lambda_grip * l_grip, {"box_obj": l_obj.item(), "box_grip": l_grip.item()}

    def align_loss(self, hidden: dict, dino_target: torch.Tensor, valid: torch.Tensor):
        """``hidden[l]``: token states [B, 64, D] at layer ``l``; ``dino_target``: [B, 16, Dd]."""
        losses = []
        for layer in self.align_layers:
            pred = self.proj[str(layer)](hidden[layer][:, SLICES["align"]].to(self.proj[str(layer)].weight.dtype))
            per = (pred.float() - dino_target.float()).abs().mean(dim=(1, 2))
            losses.append(per)
        per_sample = torch.stack(losses).mean(0)
        v = valid.float()
        return (per_sample * v).sum() / v.sum().clamp_min(1.0)


def giou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Generalised IoU of xyxy boxes, elementwise over leading dims."""
    area_a = (a[..., 2] - a[..., 0]).clamp_min(0) * (a[..., 3] - a[..., 1]).clamp_min(0)
    area_b = (b[..., 2] - b[..., 0]).clamp_min(0) * (b[..., 3] - b[..., 1]).clamp_min(0)
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp_min(0).prod(-1)
    union = area_a + area_b - inter
    iou = inter / union.clamp_min(1e-7)
    lt_c = torch.minimum(a[..., :2], b[..., :2])
    rb_c = torch.maximum(a[..., 2:], b[..., 2:])
    hull = (rb_c - lt_c).clamp_min(0).prod(-1)
    return iou - (hull - union) / hull.clamp_min(1e-7)


def box_loss(pred, target, valid, lambda_giou: float) -> torch.Tensor:
    """Eq. 3, averaged over valid (sample, step) pairs."""
    target = target.to(pred.dtype)
    l1 = F.smooth_l1_loss(pred, target, beta=1.0, reduction="none").sum(-1)
    g = 1.0 - giou(pred, target)
    per = l1 + lambda_giou * g
    v = valid.float()
    return (per * v).sum() / v.sum().clamp_min(1.0)


def attach(video_expert: nn.Module, cfg: InteractionConfig) -> InteractionTokens:
    dim = int(video_expert.blocks[0].hidden_dim)
    module = InteractionTokens(dim, len(video_expert.blocks), cfg)
    module.to(device=video_expert.blocks[0].modulation.device, dtype=video_expert.blocks[0].modulation.dtype)
    video_expert.imago_interaction = module
    return module


def get(video_expert: nn.Module):
    return getattr(video_expert, "imago_interaction", None)
