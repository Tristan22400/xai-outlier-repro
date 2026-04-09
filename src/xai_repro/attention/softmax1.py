"""softmax-1 attention for GPT-2.

Implements the "softmax off-by-one" from Miller (2023):

    softmax1(x)_i = exp(x_i) / (1 + sum_j exp(x_j))

This lets attention heads put mass "nowhere" (rows sum to < 1) and has
been argued to prevent the attention-sink / outlier-feature pathology in
transformer LMs.

The implementation replaces only the softmax step inside
``GPT2Attention._attn``; every other code path (QKV projection, masking,
head pruning, cross-attention) is inherited unchanged.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2LMHeadModel


def softmax1(scores: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable softmax-1 along ``dim``.

    Uses the log-sum-exp trick: letting ``m = max(scores, dim)``,

        softmax1(x)_i = exp(x_i - m) / (exp(-m) + sum_j exp(x_j - m)).

    Rows sum to a value in ``(0, 1]`` — strictly less than 1 unless some
    entry dominates. Causal masking sets invalid positions to ``-inf``
    *before* calling this function, so their ``exp`` contribution is 0,
    exactly as in stock GPT-2 softmax.

    Parameters
    ----------
    scores:
        Raw attention scores, any shape. Must be float32 for numerical
        safety (the full training pipeline is fp32; see plan §5).
    dim:
        Reduction axis.

    Returns
    -------
    Tensor of the same shape as ``scores``, rows summing to ``<= 1``.
    """
    m = scores.amax(dim=dim, keepdim=True)
    # Guard the case where an entire row is -inf (fully masked): clamp m
    # so that ``exp(-m)`` stays finite and the row becomes all zeros.
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    shifted = scores - m
    exp_shifted = torch.exp(shifted)
    # denom = exp(-m) + sum exp(x - m); broadcast m back after the sum.
    denom = torch.exp(-m) + exp_shifted.sum(dim=dim, keepdim=True)
    return exp_shifted / denom


class Softmax1Attention(GPT2Attention):
    """Drop-in ``GPT2Attention`` subclass that swaps softmax for softmax-1.

    Only ``_attn`` is overridden; ``_upcast_and_reordered_attn`` is not
    used because we always pass through the standard path in fp32.
    """

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
                [],
                float(value.size(-1)) ** 0.5,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )

        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            query_length, key_length = query.size(-2), key.size(-2)
            bias = cast(Tensor, self.bias)
            causal_mask = bias[:, :, key_length - query_length : key_length, :key_length].bool()
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value_t = torch.full(
                [], mask_value, dtype=attn_weights.dtype, device=attn_weights.device
            )
            attn_weights = torch.where(causal_mask, attn_weights, mask_value_t)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # The only deviation from stock GPT2Attention._attn.
        attn_weights = softmax1(attn_weights, dim=-1).type(value.dtype)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


def inject_softmax1(model: GPT2LMHeadModel) -> GPT2LMHeadModel:
    """Replace every ``GPT2Attention`` block in ``model`` with ``Softmax1Attention``.

    Weights are copied from the original attention modules so pre-trained
    checkpoints (if any) remain usable; for a fresh init this is still
    correct because the state dict keys are identical.
    """
    config = model.config
    for i, block in enumerate(model.transformer.h):
        old: GPT2Attention = cast(GPT2Attention, block.attn)
        new = Softmax1Attention(config=config, layer_idx=i)
        new.load_state_dict(old.state_dict())
        block.attn = new
    return model


__all__ = ["Softmax1Attention", "inject_softmax1", "softmax1"]
