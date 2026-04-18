"""Unit tests for softmax-1 attention."""

from __future__ import annotations

import torch

from xai_repro.attention import softmax1


def _naive_softmax1(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Reference implementation straight from the definition."""
    e = torch.exp(x)
    return e / (1.0 + e.sum(dim=dim, keepdim=True))


def test_rows_sum_below_one() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float64)
    y = softmax1(x, dim=-1)
    row_sums = y.sum(dim=-1)
    assert torch.all(row_sums < 1.0 + 1e-12)
    assert torch.all(row_sums > 0.0)


def test_matches_naive_reference() -> None:
    torch.manual_seed(1)
    x = torch.randn(3, 5, 7, dtype=torch.float64) * 2.0
    y_stable = softmax1(x, dim=-1)
    y_naive = _naive_softmax1(x, dim=-1)
    assert torch.allclose(y_stable, y_naive, atol=1e-12, rtol=1e-10)


def test_shift_invariance_to_large_scores() -> None:
    """softmax-1 is NOT shift-invariant (unlike softmax), but the stable
    implementation must not overflow for very large scores."""
    torch.manual_seed(2)
    x = torch.randn(2, 4, dtype=torch.float64) + 1000.0
    y = softmax1(x, dim=-1)
    assert torch.isfinite(y).all()
    # With huge positive scores, the +1 in the denominator is negligible
    # and softmax-1 tends to standard softmax.
    y_softmax = torch.softmax(x, dim=-1)
    assert torch.allclose(y, y_softmax, atol=1e-6)


def test_fully_masked_row_is_zero() -> None:
    """A row of -inf (fully masked) should produce all zeros, not NaN."""
    x = torch.full((2, 4), float("-inf"), dtype=torch.float64)
    x[0, 0] = 0.0  # partially unmasked row
    y = softmax1(x, dim=-1)
    assert torch.isfinite(y).all()
    assert torch.all(y[1] == 0.0)  # fully masked row -> zeros
    assert y[0, 0] > 0.0


def test_gradients_finite() -> None:
    torch.manual_seed(3)
    x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    y = softmax1(x, dim=-1)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
