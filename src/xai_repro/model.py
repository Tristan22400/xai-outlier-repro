"""Model factory for the GPT-2 variants used in the reproduction.

All three variants share the same architecture with two changes required
by the paper (arXiv 2410.17174, §5):

  1. RMSNormSingle — replaces standard LayerNorm everywhere.
     Standard LayerNorm has per-channel γ, β which re-introduces
     axis-privileged directions that OrthoAdam tries to eliminate.
     With RMSNorm-S, OrthoAdam brings kurtosis to ~3; with LayerNorm
     it stays at ~188 (Table 4 of the paper).

  2. No biases in feedforward (MLP) layers.

The only difference between variants is:
  - "baseline"   : standard Adam, standard softmax
  - "softmax1"   : standard Adam, softmax-1 (no first-token sink)
  - "orthoadam"  : OrthoAdam,     standard softmax
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import yaml
from transformers import GPT2Config, GPT2LMHeadModel

from xai_repro.attention import inject_softmax1
from xai_repro.norm import inject_rmsnorm_single

Variant = Literal["baseline", "softmax1", "orthoadam", "softmax1_ortho", "vanilla_gpt2"]


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML experiment config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _remove_mlp_biases(model: GPT2LMHeadModel) -> GPT2LMHeadModel:
    """Replace MLP Conv1D layers with bias-free nn.Linear equivalents.

    GPT-2 uses ``Conv1D`` (a transposed Linear) in MLP blocks. Its
    ``forward`` method calls ``torch.addmm(self.bias, ...)`` which crashes
    when bias is set to None. We therefore swap the entire layer for a
    standard ``nn.Linear(bias=False)`` with the transposed weight.
    """
    for block in model.transformer.h:
        mlp = block.mlp
        for attr in ("c_fc", "c_proj"):
            old = getattr(mlp, attr, None)
            if old is None:
                continue
            # Conv1D weight shape: (in_features, out_features) — transpose for Linear
            out_f, in_f = old.weight.shape[1], old.weight.shape[0]
            new_layer = nn.Linear(in_f, out_f, bias=False)
            with torch.no_grad():
                new_layer.weight.copy_(old.weight.T)
            setattr(mlp, attr, new_layer)
    return model


def build_model(variant: Variant, config_path: str | Path) -> GPT2LMHeadModel:
    """Instantiate a fresh GPT-2 for the given variant.

    Applied to ALL variants (paper §5):
      - RMSNormSingle instead of LayerNorm
      - No biases in MLP layers

    Applied only to 'softmax1':
      - softmax-1 attention
    """
    cfg = load_config(config_path)
    model_cfg = GPT2Config(**cfg["model"])
    model = GPT2LMHeadModel(model_cfg)

    # Paper §5: the "modified base" used by every row of Table 2 applies
    # RMSNormSingle and drops the FFN biases. This is the default for the
    # four Table-2 variants (baseline / softmax1 / orthoadam / softmax1_ortho).
    # The ``vanilla_gpt2`` variant preserves LayerNorm + FFN biases to
    # reproduce row 1 of the Table 5 ablation (Absolute / LayerNorm / Adam).
    if variant != "vanilla_gpt2":
        model = inject_rmsnorm_single(model)
        model = _remove_mlp_biases(model)

    # Variant-specific attention
    if variant in ("softmax1", "softmax1_ortho"):
        model = inject_softmax1(model)

    return model


def count_parameters(model: GPT2LMHeadModel) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["Variant", "build_model", "count_parameters", "load_config"]
