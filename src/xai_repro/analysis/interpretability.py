"""Interpretability extraction: attention dominance + hidden-state statistics.

Implements the core analyses from Kaul et al. (2024) "From Attention to
Activation: Unravelling the Enigmas of Large Language Models":

1. **First-token attention dominance** — % of (query, head) pairs where
   argmax of the attention distribution falls on the first key token.
   Computed on-the-fly via forward hooks to avoid storing full L×L matrices.

2. **Hidden-state kurtosis** — per-layer, per-position kurtosis (Eq. 2)
   using Welford-style running accumulators.  Separates first-token vs.
   remaining positions (Table 2 in the paper).

3. **Single-sequence snapshot** — for one held-out sequence, stores the
   *full* attention weight tensors and hidden-state activations so we can
   render qualitative heatmaps (Figure 1a/b replicas).

All three analyses share a single forward pass over the validation set to
minimise GPU time.

Usage::

    python -m xai_repro.analysis.interpretability \
        --checkpoint runs/baseline/final \
        --config configs/gpt2_60m.yaml \
        --out_dir analysis_results/baseline
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from xai_repro.data import load_c4
from xai_repro.model import load_config

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class AttentionDominanceStats:
    """Running counters for first-token attention dominance per layer."""

    n_layers: int
    n_heads: int
    # (n_layers, n_heads): count of (query, head) pairs attending maximally to tok 0
    first_tok_count: Tensor = field(init=False)
    # (n_layers, n_heads): total query positions processed
    total_count: Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.first_tok_count = torch.zeros(self.n_layers, self.n_heads, dtype=torch.int64)
        self.total_count = torch.zeros(self.n_layers, self.n_heads, dtype=torch.int64)

    def update(self, layer_idx: int, attn_weights: Tensor) -> None:
        """Update from attention weights of shape (B, n_heads, T, T)."""
        # attn_weights: (B, H, Tq, Tk)
        argmax_keys = attn_weights.argmax(dim=-1)  # (B, H, Tq)
        is_first = (argmax_keys == 0)  # (B, H, Tq)
        # Sum over batch and query positions
        self.first_tok_count[layer_idx] += is_first.sum(dim=(0, 2)).cpu()
        self.total_count[layer_idx] += is_first.shape[0] * is_first.shape[2]

    def dominance_ratio(self) -> Tensor:
        """Per-layer, per-head fraction attending maximally to token 0."""
        return self.first_tok_count.float() / self.total_count.float().clamp(min=1)

    def dominance_per_layer(self) -> Tensor:
        """Per-layer fraction (averaged over heads)."""
        return self.dominance_ratio().mean(dim=-1)

    def overall(self) -> float:
        """Scalar: overall fraction across all layers and heads."""
        return float(self.first_tok_count.sum()) / float(self.total_count.sum().clamp(min=1))


@dataclass
class HiddenStateStats:
    """Running statistics for hidden-state kurtosis and activation magnitudes.

    Implements the paper's Eq. 2 exactly:

        κ_{m, l} = E_d[(X_{m,l,d} − μ_{m,l})⁴] / E_d[(X_{m,l,d} − μ_{m,l})²]²

    i.e. for each (layer m, token position l) the kurtosis is computed **over
    the D feature channels** of the residual-stream vector at that position.
    The result is a single scalar per (sample, layer, position); we then
    average over samples and positions, partitioned into
    {first token (l=0), remaining tokens (l≥1), all tokens}.

    This matches the paper's definition.  Reporting is raw kurtosis (Gaussian
    ≈ 3); no −3 is applied — see paper §4.

    Activation magnitudes:
      - ``mean_abs``: E over (sample, position, channel) of |X|, per layer.
      - ``max_abs``:  max over (sample, position, channel) of |X|, per layer.
    Both are partitioned by {first, rest, all}.
    """

    n_layers: int
    n_embd: int

    def __post_init__(self) -> None:
        for suffix in ("all", "first", "rest"):
            # Per-layer running sum of per-sample-position kurtosis scalars
            setattr(self, f"kurt_sum_{suffix}", torch.zeros(self.n_layers, dtype=torch.float64))
            # Count of (sample, position) pairs contributing to that sum
            setattr(self, f"kurt_n_{suffix}", torch.zeros(self.n_layers, dtype=torch.float64))
            # Per-layer running sum of |X| (over (sample, position, channel))
            setattr(self, f"abs_sum_{suffix}", torch.zeros(self.n_layers, dtype=torch.float64))
            # Per-layer count of (sample, position, channel) items for the abs mean
            setattr(self, f"abs_n_{suffix}", torch.zeros(self.n_layers, dtype=torch.float64))
            # Per-layer running max of |X|
            setattr(self, f"max_abs_{suffix}", torch.zeros(self.n_layers, dtype=torch.float64))

    @staticmethod
    def _kurtosis_over_channels(x: Tensor) -> Tensor:
        """Raw kurtosis of the last axis for every preceding index.

        Parameters
        ----------
        x : Tensor of shape (..., D)

        Returns
        -------
        Tensor of shape (...) containing `E[(x−μ)⁴] / E[(x−μ)²]²` along D.
        """
        mu = x.mean(dim=-1, keepdim=True)
        centered = x - mu
        m2 = (centered ** 2).mean(dim=-1)
        m4 = (centered ** 4).mean(dim=-1)
        kurt = m4 / m2.clamp(min=1e-30).pow(2)
        # Jensen's inequality guarantees raw kurtosis ≥ 1 (Gaussian = 3).
        # Any value < 0.99 signals a numerical or logic bug.
        if not (kurt >= 0.99).all():
            import warnings
            warnings.warn(
                f"Kurtosis below 1 detected (min={float(kurt.min()):.4f}). "
                "This violates Jensen's inequality and indicates a bug.",
                RuntimeWarning, stacklevel=2,
            )
        return kurt

    def _accumulate(self, layer_idx: int, x: Tensor, suffix: str) -> None:
        """Accumulate kurtosis + |x| stats for ``x`` of shape (N, D) where
        each row is one (sample, position) hidden-state vector."""
        if x.numel() == 0:
            return
        n = x.shape[0]
        d = x.shape[-1]

        # Kurtosis over the D axis → one scalar per row
        kurt_row = self._kurtosis_over_channels(x)         # (N,)
        getattr(self, f"kurt_sum_{suffix}")[layer_idx] += kurt_row.sum()
        getattr(self, f"kurt_n_{suffix}")[layer_idx] += n

        # |activation| mean & max
        abs_x = x.abs()
        getattr(self, f"abs_sum_{suffix}")[layer_idx] += abs_x.sum()
        getattr(self, f"abs_n_{suffix}")[layer_idx] += n * d
        cur_max = float(abs_x.max())
        if cur_max > float(getattr(self, f"max_abs_{suffix}")[layer_idx]):
            getattr(self, f"max_abs_{suffix}")[layer_idx] = torch.tensor(cur_max, dtype=torch.float64)

    def update(self, layer_idx: int, hidden: Tensor) -> None:
        """Update stats from hidden states of shape (B, T, D)."""
        x = hidden.to(torch.float64)
        b, t, d = x.shape

        x_first = x[:, 0, :]                               # (B, D)
        self._accumulate(layer_idx, x_first, "first")

        if t > 1:
            x_rest = x[:, 1:, :].reshape(-1, d)            # (B·(T−1), D)
            self._accumulate(layer_idx, x_rest, "rest")

        x_all = x.reshape(-1, d)                           # (B·T, D)
        self._accumulate(layer_idx, x_all, "all")

    # --- getters ----------------------------------------------------------

    def kurtosis_per_layer(self, suffix: str = "all") -> Tensor:
        """Per-layer mean of the per-position kurtosis (paper Eq. 2).

        Shape: (n_layers,).  Raw kurtosis — Gaussian ≈ 3.
        """
        s = getattr(self, f"kurt_sum_{suffix}")
        n = getattr(self, f"kurt_n_{suffix}").clamp(min=1.0)
        return s / n

    # Backwards-compatible name used by callers.
    def mean_kurtosis_per_layer(self, suffix: str = "all") -> Tensor:
        return self.kurtosis_per_layer(suffix)

    def max_activation(self, suffix: str = "all") -> Tensor:
        """Per-layer max absolute activation."""
        return getattr(self, f"max_abs_{suffix}")

    def mean_activation(self, suffix: str = "all") -> Tensor:
        """Per-layer mean absolute activation (avg over samples, positions, channels)."""
        s = getattr(self, f"abs_sum_{suffix}")
        n = getattr(self, f"abs_n_{suffix}").clamp(min=1.0)
        return s / n


@dataclass
class SingleSequenceSnapshot:
    """Stores full tensors for one sequence — for qualitative heatmaps."""

    # attention_weights[layer] = (1, n_heads, T, T)
    attention_weights: dict[int, Tensor] = field(default_factory=dict)
    # hidden_states[layer] = (1, T, D)
    hidden_states: dict[int, Tensor] = field(default_factory=dict)
    captured: bool = False


# ---------------------------------------------------------------------------
# Hook installation
# ---------------------------------------------------------------------------


HookFn = Callable[[nn.Module, tuple[Tensor, ...], Any], None]


def _install_hooks(
    model: GPT2LMHeadModel,
    attn_stats: AttentionDominanceStats,
    hidden_stats: HiddenStateStats,
    snapshot: SingleSequenceSnapshot | None,
    snapshot_batch_idx: int = 0,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register forward hooks on all transformer blocks.

    Each block hook captures:
    - The attention weights (from the attention sub-module)
    - The residual-stream hidden state (block output)

    For the attention weights, we need ``output_attentions=True`` in the
    model config so that GPT2Attention returns them.
    """
    handles: list[torch.utils.hooks.RemovableHandle] = []
    batch_counter = {"idx": 0}

    # Hook on each transformer block for hidden states
    def make_block_hook(layer_idx: int) -> HookFn:
        def hook(
            _m: nn.Module,
            _inp: tuple[Tensor, ...],
            out: Tensor | tuple[Tensor, ...],
        ) -> None:
            h = out[0] if isinstance(out, tuple) else out
            hidden_stats.update(layer_idx, h.detach())

            if snapshot is not None and batch_counter["idx"] == snapshot_batch_idx and not snapshot.captured:
                snapshot.hidden_states[layer_idx] = h[:1].detach().cpu()

        return hook

    # Hook on each attention module for attention weights
    def make_attn_hook(layer_idx: int) -> HookFn:
        def hook(
            _m: nn.Module,
            _inp: tuple[Tensor, ...],
            out: tuple[Tensor, ...],
        ) -> None:
            # GPT2Attention forward returns (attn_output, present, attn_weights)
            # when output_attentions=True, attn_weights is out[2] or out[1]
            # Actually: returns (attn_output, present) or (attn_output, present, attn_weights)
            if len(out) >= 3:
                attn_w = out[2]  # (B, n_heads, T, T)
            elif len(out) >= 2 and out[1] is not None and out[1].dim() == 4:
                attn_w = out[1]
            else:
                return

            attn_stats.update(layer_idx, attn_w.detach())

            if snapshot is not None and batch_counter["idx"] == snapshot_batch_idx and not snapshot.captured:
                snapshot.attention_weights[layer_idx] = attn_w[:1].detach().cpu()

        return hook

    for i, block in enumerate(model.transformer.h):
        handles.append(block.register_forward_hook(make_block_hook(i)))
        handles.append(block.attn.register_forward_hook(make_attn_hook(i)))

    # We also need a hook to increment the batch counter *after* each full forward
    def batch_counter_hook(
        _m: nn.Module, _inp: tuple[Tensor, ...], _out: Any
    ) -> None:
        if snapshot is not None and batch_counter["idx"] == snapshot_batch_idx:
            snapshot.captured = True
        batch_counter["idx"] += 1

    handles.append(model.register_forward_hook(batch_counter_hook))

    return handles


# ---------------------------------------------------------------------------
# Main extraction routine
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_all(
    model: GPT2LMHeadModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 64,
    snapshot_batch_idx: int = 0,
) -> tuple[AttentionDominanceStats, HiddenStateStats, SingleSequenceSnapshot]:
    """Run a single forward pass collecting all interpretability metrics.

    Parameters
    ----------
    model:
        Trained GPT2LMHeadModel (already on ``device``).
    loader:
        Validation DataLoader.
    device:
        Compute device.
    max_batches:
        Cap the number of batches to process (OOM protection).
    snapshot_batch_idx:
        Which batch to capture full tensors for (heatmap generation).

    Returns
    -------
    (attn_stats, hidden_stats, snapshot)
    """
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    n_embd = model.config.n_embd

    # SDPA does not return attention weights; force the eager implementation.
    model.config._attn_implementation = "eager"
    model.config.output_attentions = True

    attn_stats = AttentionDominanceStats(n_layers=n_layers, n_heads=n_heads)
    hidden_stats = HiddenStateStats(n_layers=n_layers, n_embd=n_embd)
    snapshot = SingleSequenceSnapshot()

    handles = _install_hooks(model, attn_stats, hidden_stats, snapshot, snapshot_batch_idx)

    try:
        model.eval()
        for step, batch in enumerate(loader):
            if step >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
    finally:
        for h in handles:
            h.remove()
        model.config.output_attentions = False

    return attn_stats, hidden_stats, snapshot


# ---------------------------------------------------------------------------
# Summary report generation
# ---------------------------------------------------------------------------


def build_report(
    attn_stats: AttentionDominanceStats,
    hidden_stats: HiddenStateStats,
    variant: str,
) -> dict[str, Any]:
    """Flat dictionary of scalar metrics suitable for W&B logging / JSON."""
    kurt_all = hidden_stats.kurtosis_per_layer("all")
    kurt_first = hidden_stats.kurtosis_per_layer("first")
    kurt_rest = hidden_stats.kurtosis_per_layer("rest")
    max_act_all = hidden_stats.max_activation("all")
    max_act_first = hidden_stats.max_activation("first")
    max_act_rest = hidden_stats.max_activation("rest")
    mean_act_all = hidden_stats.mean_activation("all")
    mean_act_first = hidden_stats.mean_activation("first")
    mean_act_rest = hidden_stats.mean_activation("rest")

    report: dict[str, Any] = {
        "variant": variant,
        # Attention dominance (paper Table 2 column "%First Attn")
        "attn/first_token_dominance_overall": attn_stats.overall(),
        "attn/dominance_per_layer": attn_stats.dominance_per_layer().tolist(),
        "attn/dominance_per_layer_per_head": attn_stats.dominance_ratio().tolist(),
        # Kurtosis — paper Table 2 columns E_m[κ_{m,1}] and E_m[κ_{m,>1}]
        # Raw (Gaussian ≈ 3). Averaged across layers.
        "kurtosis/mean_all_layers": float(kurt_all.mean()),
        "kurtosis/mean_per_layer": kurt_all.tolist(),
        "kurtosis/mean_first_token": float(kurt_first.mean()),
        "kurtosis/first_per_layer": kurt_first.tolist(),
        "kurtosis/mean_rest_tokens": float(kurt_rest.mean()),
        "kurtosis/rest_per_layer": kurt_rest.tolist(),
        # Mean absolute activation — paper Table 2 columns E_m[‖X_{m,·,d}‖]
        "activation/mean_abs_all": float(mean_act_all.mean()),
        "activation/mean_abs_first_token": float(mean_act_first.mean()),
        "activation/mean_abs_rest_tokens": float(mean_act_rest.mean()),
        "activation/mean_abs_per_layer_all": mean_act_all.tolist(),
        "activation/mean_abs_per_layer_first": mean_act_first.tolist(),
        "activation/mean_abs_per_layer_rest": mean_act_rest.tolist(),
        # Max absolute activation (useful for Fig 1b sanity)
        "activation/max_abs_all": float(max_act_all.max()),
        "activation/max_abs_first_token": float(max_act_first.max()),
        "activation/max_abs_rest_tokens": float(max_act_rest.max()),
        "activation/max_abs_per_layer_all": max_act_all.tolist(),
        "activation/max_abs_per_layer_first": max_act_first.tolist(),
        "activation/max_abs_per_layer_rest": max_act_rest.tolist(),
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract interpretability metrics from a trained GPT-2.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to HF model directory.")
    p.add_argument("--config", type=Path, required=True, help="YAML experiment config.")
    p.add_argument("--variant", type=str, required=True, choices=("baseline", "softmax1", "orthoadam"))
    p.add_argument("--out_dir", type=Path, required=True, help="Directory for output JSON + tensors.")
    p.add_argument("--max_batches", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--wandb_run", type=str, default=None, help="W&B run id to log results to.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint from {args.checkpoint} ...")
    model = GPT2LMHeadModel.from_pretrained(
        str(args.checkpoint), attn_implementation="eager"
    ).to(device)

    # If this is a softmax1 model, inject the softmax1 attention
    if args.variant == "softmax1":
        from xai_repro.attention import inject_softmax1
        model = inject_softmax1(model)

    data = load_c4(seq_len=cfg["training"]["seq_len"])
    loader = DataLoader(
        data.validation,
        batch_size=args.batch_size,
        collate_fn=data.collator,
        shuffle=False,
    )

    print(f"Running extraction (max_batches={args.max_batches}) ...")
    attn_stats, hidden_stats, snapshot = extract_all(
        model, loader, device, max_batches=args.max_batches
    )

    # Build and save report
    report = build_report(attn_stats, hidden_stats, args.variant)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")

    # Save snapshot tensors for heatmap generation
    snapshot_data = {
        "attention_weights": {
            str(k): v for k, v in snapshot.attention_weights.items()
        },
        "hidden_states": {
            str(k): v for k, v in snapshot.hidden_states.items()
        },
    }
    torch.save(snapshot_data, args.out_dir / "snapshot.pt")
    print(f"Snapshot tensors saved to {args.out_dir / 'snapshot.pt'}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Variant: {args.variant}")
    print(f"First-token attention dominance: {attn_stats.overall():.3%}")
    print(f"Mean kurtosis (all positions):   {report['kurtosis/mean_all_layers']:.1f}")
    print(f"Mean kurtosis (first token):     {report['kurtosis/mean_first_token']:.1f}")
    print(f"Mean kurtosis (rest tokens):     {report['kurtosis/mean_rest_tokens']:.1f}")
    print(f"Max absolute activation:         {report['activation/max_abs_all']:.1f}")
    print(f"{'='*60}")

    # Optionally log to W&B
    if args.wandb_run is not None:
        import wandb
        wandb.init(
            project=cfg["training"]["wandb_project"],
            id=args.wandb_run,
            resume="must",
        )
        # Log scalar summaries
        wandb.log({
            f"analysis/{k}": v
            for k, v in report.items()
            if isinstance(v, (int, float))
        })
        wandb.finish()


if __name__ == "__main__":
    main()
