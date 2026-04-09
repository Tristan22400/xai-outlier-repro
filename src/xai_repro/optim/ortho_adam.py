"""OrthoAdam: Adam with per-parameter random orthogonal rotation of gradients.

This is Algorithm 1 from the OrthoAdam paper, adapted for practical
memory on a P100. The key idea is that the per-coordinate normalization
``g / sqrt(v)`` performed by Adam privileges the axis-aligned basis of
each parameter; if that basis happens to coincide with an outlier
direction in activation space, the optimizer amplifies it. Rotating the
gradient into a random orthogonal basis before applying Adam's moment
updates — and rotating the step back before applying it — breaks this
coupling.

Naively storing a dense ``(D x D)`` rotation per parameter is
impossible: a single 512x512 weight has ``D = 262144`` and Q would be
~275 GB. We instead apply an **axis-wise** Kronecker rotation:

    g̃ = Q_0 x_0 g x_1 Q_1^T x_2 Q_2^T ...          (one Q per tensor axis)

Memory cost drops from ``(∏ d_i)²`` to ``∑ d_i²``. This matches the
Kronecker factorization ``Q = Q_0 ⊗ Q_1 ⊗ ...`` of the full ``D x D``
rotation — still a valid random orthogonal matrix, just drawn from a
structured subgroup of ``O(D)`` rather than uniformly.

For axes whose dimension exceeds ``max_rotate_dim`` (e.g. the vocab axis
of the token embedding, 50257), Q is fixed to the identity and no
rotation is performed along that axis. This is an explicit, documented
deviation from the paper.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


def _sample_orthogonal(dim: int, generator: torch.Generator, device: torch.device) -> Tensor:
    """Draw a random orthogonal matrix from the Haar measure on ``O(dim)``.

    Uses QR of a Gaussian matrix and sign-corrects the diagonal of R so
    the resulting Q is Haar-distributed (Mezzadri, 2007).
    """
    a = torch.randn(dim, dim, generator=generator, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(a)
    d = torch.diagonal(r)
    q = q * torch.sign(d).unsqueeze(0)
    return q


def _rotate(x: Tensor, qs: list[Tensor | None], transpose: bool) -> Tensor:
    """Apply axis-wise rotation ``⊗_i Q_i`` (or its transpose) to ``x``.

    For each axis ``i`` with a non-``None`` ``Q_i``, contract that axis:
    ``x <- Q_i @ x`` along axis ``i`` (or ``Q_i^T @ x`` if ``transpose``).

    Axes with ``Q_i is None`` are treated as identity rotations (no-op).
    """
    out = x
    for i, q in enumerate(qs):
        if q is None:
            continue
        mat = q.t() if transpose else q
        # Move axis i to position 0, flatten the rest, matmul, unflatten, move back.
        moved = out.movedim(i, 0)
        original_shape = moved.shape
        flat = moved.reshape(original_shape[0], -1)
        rotated = mat @ flat
        out = rotated.reshape(original_shape).movedim(0, i)
    return out


class OrthoAdam(Optimizer):
    """Adam variant that performs moment updates in a random orthogonal basis.

    Behaviourally identical to :class:`torch.optim.AdamW` when
    ``max_rotate_dim=0`` (all Q = I), which the unit tests rely on.

    Parameters
    ----------
    params:
        Iterable of parameters or parameter groups.
    lr:
        Peak learning rate.
    betas:
        Adam momentum coefficients (first and second moment).
    eps:
        Denominator stability term.
    weight_decay:
        Decoupled (AdamW-style) weight decay applied in the original basis.
    max_rotate_dim:
        Any tensor axis with dimension greater than this is not rotated
        (identity Q). Set to ``0`` to disable rotation entirely and
        recover AdamW exactly.
    seed:
        Base seed for per-parameter Q generation. Each parameter gets a
        distinct derived seed based on its position in the optimizer's
        ``param_groups`` traversal order, so rotations are deterministic
        across reloads provided the param order is stable.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        max_rotate_dim: int = 4096,
        seed: int = 0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"invalid betas {betas}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        if max_rotate_dim < 0:
            raise ValueError(f"max_rotate_dim must be non-negative, got {max_rotate_dim}")

        defaults: dict[str, Any] = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "max_rotate_dim": max_rotate_dim,
        }
        super().__init__(params, defaults)
        self._seed = seed

    def _build_qs_for(
        self, p: Tensor, param_index: int, max_rotate_dim: int
    ) -> list[Tensor | None]:
        qs: list[Tensor | None] = []
        for axis, dim in enumerate(p.shape):
            if max_rotate_dim == 0 or dim > max_rotate_dim or dim < 2:
                qs.append(None)
                continue
            gen = torch.Generator(device=p.device)
            # Mix seed with (param_index, axis, dim) for a stable, distinct stream.
            gen.manual_seed(self._seed + 1_000_003 * param_index + 1009 * axis + dim)
            qs.append(_sample_orthogonal(dim, gen, p.device))
        return qs

    @torch.no_grad()
    def step(self, closure: Any = None) -> Tensor | None:  # type: ignore[override]
        loss: Tensor | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Global param index across all groups for stable seeding.
        param_index = 0
        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]
            max_rotate_dim: int = group["max_rotate_dim"]

            for p in group["params"]:
                if p.grad is None:
                    param_index += 1
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("OrthoAdam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["qs"] = self._build_qs_for(p, param_index, max_rotate_dim)

                state["step"] += 1
                step: int = state["step"]
                m: Tensor = state["m"]
                v: Tensor = state["v"]
                qs: list[Tensor | None] = state["qs"]

                # Rotate gradient into the per-parameter orthogonal basis.
                g_tilde = _rotate(grad, qs, transpose=False)

                # Standard Adam moment updates (in the rotated basis).
                m.mul_(beta1).add_(g_tilde, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g_tilde, g_tilde, value=1.0 - beta2)

                bias1 = 1.0 - beta1**step
                bias2 = 1.0 - beta2**step
                m_hat = m / bias1
                v_hat = v / bias2

                # Adam step in the rotated basis, then rotate back.
                s_tilde = -lr * m_hat / (v_hat.sqrt() + eps)
                s = _rotate(s_tilde, qs, transpose=True)

                # Decoupled weight decay, applied in the original basis.
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                p.add_(s)
                param_index += 1

        return loss


__all__ = ["OrthoAdam"]
