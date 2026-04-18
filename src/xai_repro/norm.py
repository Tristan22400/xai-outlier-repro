"""RMSNormSingle — the normalization layer required by the paper.

The paper (arXiv 2410.17174, §5) explicitly states that OrthoAdam only
drives kurtosis to ~3 when **RMSNormSingle** is used instead of standard
LayerNorm.  Standard LayerNorm has per-channel learned scale/shift (γ, β),
which re-introduces axis-privileged directions that OrthoAdam is trying to
eliminate.  RMSNormSingle uses a single shared scalar γ and no bias — no
per-channel parameters — so it cannot re-amplify any particular axis.

From the paper (Table 4, 130M ablation):
    LayerNorm   + OrthoAdam + softmax-1 → kurtosis 188.4
    RMSNorm-S   + OrthoAdam + softmax-1 → kurtosis   3.0  ✓

Usage::

    inject_rmsnorm_single(model)  # replaces all LayerNorm in-place
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


class RMSNormSingle(nn.Module):
    """Root-mean-square normalization with a single shared scale.

    Unlike LayerNorm or standard RMSNorm, there is **no per-channel**
    learnable parameter — just one scalar ``γ`` shared across all
    feature dimensions.  This matches "RMSNorm-S" / "Simple RMSNorm"
    from Qin et al. (2023) and the paper's ablation (Table 4).

    Parameters
    ----------
    normalized_shape:
        The size of the last dimension (used only for compatibility with
        the LayerNorm interface; the actual γ is a scalar).
    eps:
        Stability epsilon. Defaults to 1e-5 to match GPT2Config default.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # Single scalar, initialised to 1.0 (identity scaling at init)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``x`` by its RMS then scale by the single shared scalar.

        Args:
            x: Input activations. Shape: ``(..., d_model)``.

        Returns:
            Normalized tensor of the same shape as ``x``.
        """
        assert self.scale.numel() == 1, (
            f"RMSNormSingle.scale must be a scalar; got shape {self.scale.shape}. "
            "A per-channel scale would reintroduce axis-privileged directions."
        )
        # RMS normalise along the last axis, then apply shared scale
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


def inject_rmsnorm_single(model: GPT2LMHeadModel) -> GPT2LMHeadModel:
    """Replace every ``nn.LayerNorm`` in *model* with ``RMSNormSingle``.

    Works by traversing named modules and swapping in-place so that the
    rest of the model (weight tying, state-dict keys, etc.) is unaffected.
    """
    # Collect replacements first to avoid mutating while iterating
    replacements: list[tuple[nn.Module, str, nn.LayerNorm]] = []
    for name, module in model.named_modules():
        for attr_name, child in module.named_children():
            if isinstance(child, nn.LayerNorm):
                replacements.append((module, attr_name, child))

    for parent, attr_name, old_ln in replacements:
        new_norm = RMSNormSingle(
            normalized_shape=old_ln.normalized_shape[0],
            eps=old_ln.eps,
        )
        setattr(parent, attr_name, new_norm)

    return model


__all__ = ["RMSNormSingle", "inject_rmsnorm_single"]
