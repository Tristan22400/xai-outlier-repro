"""OrthoAdam: Adam with per-parameter random orthogonal rotation of gradients.

Algorithm 1 from arXiv 2410.17174 (§4.2), adapted for practical memory via
an axis-wise Kronecker rotation instead of a full D×D matrix.

The key idea: Adam's per-coordinate normalisation g/sqrt(v) privileges the
axis-aligned basis of each parameter. Rotating the gradient into a random
orthogonal basis before the moment updates — and rotating the step back —
breaks this coupling and eliminates outlier activations.

Memory cost: O(Σ dᵢ²) instead of O((Π dᵢ)²).
Axes whose dim > max_rotate_dim are treated as identity (no rotation).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


def _sample_orthogonal(dim: int, generator: torch.Generator, device: torch.device) -> Tensor:
    """Draw a Haar-distributed random orthogonal matrix via QR decomposition."""
    a = torch.randn(dim, dim, generator=generator, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def _rotate(x: Tensor, qs: list[Tensor | None], transpose: bool) -> Tensor:
    """Apply axis-wise Kronecker rotation ⊗ᵢ Qᵢ (or its transpose) to x."""
    out = x
    for i, q in enumerate(qs):
        if q is None:
            continue
        mat = q.t() if transpose else q
        moved = out.movedim(i, 0)
        shape = moved.shape
        out = (mat @ moved.reshape(shape[0], -1)).reshape(shape).movedim(0, i)
    return out


class OrthoAdam(Optimizer):
    """Adam variant that performs moment updates in a random orthogonal basis.

    Equivalent to AdamW when ``max_rotate_dim=0`` (all Q = I).

    Parameters
    ----------
    params:      Parameters or param groups.
    lr:          Peak learning rate.
    betas:       (β₁, β₂) momentum coefficients.
    eps:         Denominator stability term.
    weight_decay: Decoupled (AdamW) weight decay in the original basis.
    max_rotate_dim: Axes with dim > this are not rotated. 0 = disable.
    seed:        Base seed for deterministic Q generation.
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
        defaults: dict[str, Any] = dict(lr=lr, betas=betas, eps=eps,
                                        weight_decay=weight_decay,
                                        max_rotate_dim=max_rotate_dim)
        super().__init__(params, defaults)
        self._seed = seed

    def _build_qs(self, p: Tensor, param_idx: int, max_dim: int) -> list[Tensor | None]:
        qs: list[Tensor | None] = []
        for axis, dim in enumerate(p.shape):
            if max_dim == 0 or dim > max_dim or dim < 2:
                qs.append(None)
            else:
                gen = torch.Generator(device=p.device)
                gen.manual_seed(self._seed + 1_000_003 * param_idx + 1009 * axis + dim)
                qs.append(_sample_orthogonal(dim, gen, p.device))
        return qs

    @torch.no_grad()
    def step(self, closure: Any = None) -> Tensor | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_idx = 0
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            max_dim = group["max_rotate_dim"]

            for p in group["params"]:
                if p.grad is None:
                    param_idx += 1
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["qs"] = self._build_qs(p, param_idx, max_dim)

                state["step"] += 1
                t = state["step"]
                m, v, qs = state["m"], state["v"], state["qs"]

                g = _rotate(p.grad, qs, transpose=False)
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                s = _rotate(-lr * m_hat / (v_hat.sqrt() + eps), qs, transpose=True)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(s)
                param_idx += 1

        return loss


__all__ = ["OrthoAdam"]
