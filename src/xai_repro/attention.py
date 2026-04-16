"""softmax-1 attention for GPT-2.

Implements the "softmax off-by-one" from Miller (2023) and the paper
arXiv 2410.17174 (§3.2):

    softmax1(x)_i = exp(x_i) / (1 + Σ_j exp(x_j))

This lets attention heads put mass "nowhere" (rows sum to ≤ 1) and
eliminates the attention-sink / first-token-dominance phenomenon.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2LMHeadModel


def softmax1(scores: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable softmax-1 along ``dim``.

    Uses the log-sum-exp trick: letting ``m = max(scores, dim)``,

        softmax1(x)_i = exp(x_i - m) / (exp(-m) + Σ_j exp(x_j - m)).

    Rows sum to a value in ``(0, 1]``.
    """
    m = scores.amax(dim=dim, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    shifted = scores - m
    exp_shifted = torch.exp(shifted)
    denom = torch.exp(-m) + exp_shifted.sum(dim=dim, keepdim=True)
    return exp_shifted / denom


class Softmax1Attention(GPT2Attention):
    """Drop-in ``GPT2Attention`` subclass that swaps softmax for softmax-1."""

    def _attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
        head_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], float(value.size(-1)) ** 0.5,
                dtype=attn_weights.dtype, device=attn_weights.device,
            )
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            query_length, key_length = query.size(-2), key.size(-2)
            bias = cast(Tensor, self.bias)
            causal_mask = bias[:, :, key_length - query_length : key_length, :key_length].bool()
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value_t = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value_t)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = softmax1(attn_weights, dim=-1).type(value.dtype)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


def inject_softmax1(model: GPT2LMHeadModel) -> GPT2LMHeadModel:
    """Replace every ``GPT2Attention`` block in ``model`` with ``Softmax1Attention``."""
    config = model.config
    for i, block in enumerate(model.transformer.h):
        old: GPT2Attention = cast(GPT2Attention, block.attn)
        new = Softmax1Attention(config=config, layer_idx=i)
        new.load_state_dict(old.state_dict())
        block.attn = new
    return model


__all__ = ["Softmax1Attention", "inject_softmax1", "softmax1"]
