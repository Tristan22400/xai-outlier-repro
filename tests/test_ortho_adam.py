"""Unit tests for OrthoAdam."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from xai_repro.optim.ortho_adam import OrthoAdam, _rotate, _sample_orthogonal


def _toy_model(seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.GELU(),
        nn.Linear(16, 4),
    )


def test_sample_orthogonal_is_orthogonal() -> None:
    gen = torch.Generator().manual_seed(0)
    q = _sample_orthogonal(32, gen, torch.device("cpu"))
    eye = torch.eye(32)
    assert torch.allclose(q @ q.t(), eye, atol=1e-5)
    assert torch.allclose(q.t() @ q, eye, atol=1e-5)


def test_rotate_round_trip_preserves_tensor() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 6, 5)
    gen = torch.Generator().manual_seed(1)
    qs: list[torch.Tensor | None] = [
        _sample_orthogonal(4, gen, torch.device("cpu")),
        None,
        _sample_orthogonal(5, gen, torch.device("cpu")),
    ]
    y = _rotate(x, qs, transpose=False)
    x_back = _rotate(y, qs, transpose=True)
    assert torch.allclose(x_back, x, atol=1e-5)


def test_matches_adamw_when_rotation_disabled() -> None:
    """With ``max_rotate_dim=0`` OrthoAdam must match torch.optim.AdamW
    step-for-step on identical inputs (same init, same data, same HPs)."""
    torch.manual_seed(42)
    model_a = _toy_model(seed=7)
    model_b = copy.deepcopy(model_a)

    opt_a = torch.optim.AdamW(
        model_a.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    opt_b = OrthoAdam(
        model_b.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        max_rotate_dim=0,
    )

    torch.manual_seed(123)
    for _ in range(50):
        x = torch.randn(32, 8)
        target = torch.randn(32, 4)
        loss_a = ((model_a(x) - target) ** 2).mean()
        loss_b = ((model_b(x) - target) ** 2).mean()
        opt_a.zero_grad()
        opt_b.zero_grad()
        loss_a.backward()
        loss_b.backward()
        opt_a.step()
        opt_b.step()

    for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.allclose(pa, pb, atol=1e-6, rtol=1e-5)


def test_rotation_preserves_loss_descent_on_fixed_batch() -> None:
    """With rotation enabled, OrthoAdam must still descend on a fixed batch."""
    torch.manual_seed(0)
    model = _toy_model(seed=11)
    opt = OrthoAdam(
        model.parameters(),
        lr=5e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        max_rotate_dim=64,
        seed=99,
    )
    torch.manual_seed(0)
    x = torch.randn(32, 8)
    target = torch.randn(32, 4)
    losses: list[float] = []
    for _ in range(200):
        loss = ((model(x) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # On a fixed batch the optimizer should reduce loss by at least an order
    # of magnitude.
    assert losses[-1] < losses[0] * 0.1
    assert all(math.isfinite(v) for v in losses)


def test_deterministic_q_generation_across_optimizer_instances() -> None:
    """Two OrthoAdam instances on identical params with the same seed must
    produce bitwise-identical rotations (and therefore identical updates)."""
    torch.manual_seed(7)
    model_a = _toy_model(seed=3)
    model_b = copy.deepcopy(model_a)
    opt_a = OrthoAdam(model_a.parameters(), lr=1e-3, max_rotate_dim=64, seed=123)
    opt_b = OrthoAdam(model_b.parameters(), lr=1e-3, max_rotate_dim=64, seed=123)
    torch.manual_seed(0)
    x = torch.randn(8, 8)
    for _ in range(5):
        for m, o in [(model_a, opt_a), (model_b, opt_b)]:
            loss = m(x).sum()
            o.zero_grad()
            loss.backward()
            o.step()
    for p1, p2 in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.allclose(p1, p2, atol=1e-7)
