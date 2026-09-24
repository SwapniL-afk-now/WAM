"""Checkpoint loading shared by the builder and the DIDO trainers (no circular imports)."""

from __future__ import annotations

import torch


def load_checkpoint_checked(model, path: str) -> None:
    """``FastWAM.load_checkpoint`` (non-strict) plus a check that no trained
    weights are silently dropped (e.g. interaction tokens without the module)."""
    payload = torch.load(path, map_location="cpu")
    result = model.mot.load_state_dict(payload["mot"], strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint has {len(result.unexpected_keys)} weights the model lacks "
            f"(e.g. {result.unexpected_keys[:3]}); enable `interaction` if they are DIDO tokens."
        )
    if getattr(model, "proprio_encoder", None) is not None and "proprio_encoder" in payload:
        model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
