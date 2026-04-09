"""Model factory for the 60M GPT-2 used in the reproduction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from transformers import GPT2Config, GPT2LMHeadModel

from xai_repro.attention.softmax1 import inject_softmax1

Variant = Literal["baseline", "softmax1", "orthoadam"]


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML experiment config."""
    with open(config_path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    return cfg


def build_model(variant: Variant, config_path: str | Path) -> GPT2LMHeadModel:
    """Instantiate a fresh ~60M GPT-2 for the given variant.

    Weights are initialized from scratch (no pre-trained checkpoint).
    For ``variant="softmax1"`` the attention modules are swapped for
    :class:`Softmax1Attention` after construction; for the other two
    variants the stock attention is kept and only the optimizer differs.
    """
    cfg = load_config(config_path)
    model_cfg = GPT2Config(**cfg["model"])
    model = GPT2LMHeadModel(model_cfg)

    if variant == "softmax1":
        model = inject_softmax1(model)

    return model


def count_parameters(model: GPT2LMHeadModel) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["Variant", "build_model", "count_parameters", "load_config"]
