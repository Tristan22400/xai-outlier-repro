"""softmax-1 attention for GPT-2.

Implements the "softmax off-by-one" from Miller (2023) and the paper
arXiv 2410.17174 (§3.2):

    softmax1(x)_i = exp(x_i) / (1 + Σ_j exp(x_j))

This lets attention heads put mass "nowhere" (rows sum to ≤ 1) and
eliminates the attention-sink / first-token-dominance phenomenon.
"""

from __future__ import annotations

from copy import deepcopy
from typing import cast

import torch
from torch import Tensor
from transformers.models.gpt2.modeling_gpt2 import (
    ALL_ATTENTION_FUNCTIONS,
    GPT2Attention,
    GPT2LMHeadModel,
)

_SOFTMAX1_IMPL_KEY = "softmax1"


def softmax1(scores: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable softmax-1 along ``dim``.

    Uses the log-sum-exp trick: letting ``m = max(scores, dim)``,

        softmax1(x)_i = exp(x_i - m) / (exp(-m) + Σ_j exp(x_j - m)).

    The extra ``exp(-m)`` term in the denominator corresponds to the implicit
    "null" token, guaranteeing rows sum to a value in ``(0, 1]``.

    Args:
        scores: Raw attention logits. Shape: ``(..., seq_len)``.
        dim: Dimension over which to normalize (default: last).

    Returns:
        Attention weights of the same shape as ``scores``, with each row
        summing to a value in ``(0, 1]``.
    """
    m = scores.amax(dim=dim, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    shifted = scores - m
    exp_shifted = torch.exp(shifted)
    denom = torch.exp(-m) + exp_shifted.sum(dim=dim, keepdim=True)
    out = exp_shifted / denom
    # Invariant: each row sums to at most 1 because the denominator includes
    # an extra exp(-m) ≥ 0 that is never cancelled.
    assert (out.sum(dim=dim) <= 1.0 + 1e-4).all(), (
        f"softmax1 row sums exceeded 1: max={float(out.sum(dim=dim).max()):.6f}. "
        "This violates the softmax-1 invariant and indicates a numerical bug."
    )
    return out


def _softmax1_eager_attention_forward(
    module: GPT2Attention,
    query: Tensor,  # shape: (batch, n_heads, T_q, head_dim)
    key: Tensor,    # shape: (batch, n_heads, T_k, head_dim)
    value: Tensor,  # shape: (batch, n_heads, T_k, head_dim)
    attention_mask: Tensor | None,
    head_mask: Tensor | None = None,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    """Drop-in replacement for ``eager_attention_forward`` using softmax-1.

    Args:
        module: The ``GPT2Attention`` instance (used for scaling and mask buffers).
        query: Query tensor. Shape: ``(batch, n_heads, T_q, head_dim)``.
        key: Key tensor.   Shape: ``(batch, n_heads, T_k, head_dim)``.
        value: Value tensor. Shape: ``(batch, n_heads, T_k, head_dim)``.
        attention_mask: Additive mask (``0`` or ``-inf``), shape broadcastable
            to ``(batch, n_heads, T_q, T_k)``.
        head_mask: Per-head scaling tensor (optional).

    Returns:
        ``(attn_output, attn_weights)`` where ``attn_output`` has shape
        ``(batch, T_q, n_heads * head_dim)`` after the final transpose, and
        ``attn_weights`` has shape ``(batch, n_heads, T_q, T_k)`` with rows
        summing to ≤ 1.
    """
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5,
            dtype=attn_weights.dtype, device=attn_weights.device,
        )
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)

    if not module.is_cross_attention:
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value_t = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value_t)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = softmax1(attn_weights, dim=-1).type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights


# Register once at import time so ALL_ATTENTION_FUNCTIONS["softmax1"] is always available.
ALL_ATTENTION_FUNCTIONS[_SOFTMAX1_IMPL_KEY] = _softmax1_eager_attention_forward


class Softmax1Attention(GPT2Attention):
    """Drop-in ``GPT2Attention`` subclass that uses softmax-1.

    In recent Transformers versions, ``GPT2Attention.forward`` dispatches via
    ``ALL_ATTENTION_FUNCTIONS[config._attn_implementation]`` and never calls
    ``_attn()``.  We therefore give each instance an *isolated* config copy
    whose ``_attn_implementation`` is set to ``"softmax1"``, which maps to
    ``_softmax1_eager_attention_forward`` in ``ALL_ATTENTION_FUNCTIONS``.

    The ``_attn`` override is kept for backward compatibility with older
    Transformers installs that still call it directly.
    """

    def __init__(self, config, layer_idx: int | None = None) -> None:
        cfg = deepcopy(config)
        cfg._attn_implementation = _SOFTMAX1_IMPL_KEY
        super().__init__(cfg, layer_idx=layer_idx)

    # ------------------------------------------------------------------ #
    # Backward-compat path (old Transformers that call _attn directly)    #
    # ------------------------------------------------------------------ #
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
