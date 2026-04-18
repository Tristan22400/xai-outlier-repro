"""Visualization for the interpretability analysis.

Generates all five publication-ready figures for the report:

1. **Sanity Check – Training Stability**
   Validation loss vs. step for all three variants (W&B pull).

2. **Attention Dominance vs. Depth**
   Proportion of first-token max-attention per layer index;
   baseline vs. softmax-1 only.

3. **Outlier Severity – Hidden-State Kurtosis vs. Depth**
   First-token kurtosis per layer (log scale); baseline vs. OrthoAdam.

4. **Outlier Magnitude – Max Absolute Activation vs. Depth**
   First-token max |activation| per layer (log scale); baseline vs. OrthoAdam.

5. **XAI Bonus – Optimizer Sensitivity (β₂)**
   Mean layer kurtosis vs. training step for standard Adam baseline and
   three β₂ variants of OrthoAdam (from beta2_sensitivity.json).

Usage::

    python -m xai_repro.analysis.visualize \\
        --results_dir analysis_results \\
        --out_dir analysis_results/figures \\
        --wandb_project xai-outlier-repro \\
        --wandb_runs baseline:ceoh7scy,softmax1:q9w3rxwr,orthoadam:0ljxhaso \\
        --beta2_json  analysis_results/beta2_sensitivity/beta2_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
})

# Palette – consistent across all figures
VARIANT_COLORS = {
    "vanilla_gpt2":   "#555555",   # dark grey  – paper Table 5 row 1 (LN + biases)
    "baseline":       "#1f77b4",   # blue       – RMSNorm-S, Adam, softmax
    "softmax1":       "#d62728",   # red        – softmax-1 only
    "orthoadam":      "#2ca02c",   # green      – OrthoAdam only
    "softmax1_ortho": "#9467bd",   # purple     – softmax-1 + OrthoAdam
}
VARIANT_LABELS = {
    "vanilla_gpt2":   "Vanilla GPT-2 (LN + biases)",
    "baseline":       "baseline",
    "softmax1":       "softmax-1",
    "orthoadam":      "OrthoAdam",
    "softmax1_ortho": "softmax-1 + OrthoAdam",
}

MARKER = {
    "vanilla_gpt2":   "D",
    "baseline":       "o",
    "softmax1":       "s",
    "orthoadam":      "^",
    "softmax1_ortho": "v",
}


# ---------------------------------------------------------------------------
# Figure 1 – Training Stability (validation loss vs. step)
# ---------------------------------------------------------------------------


def plot_training_curves_from_wandb(
    run_ids: dict[str, str],
    project: str,
    out_path: Path,
    entity: str | None = None,
    metric: str = "eval/loss",
    step_key: str = "train/global_step",
) -> None:
    """Pull validation-loss curves from W&B and overlay all three variants.

    Parameters
    ----------
    run_ids:
        ``{variant_name: wandb_run_id}``
    project:
        W&B project name.
    out_path:
        Destination ``.png`` path.
    entity:
        W&B entity (optional).
    metric:
        The history column to plot on the y-axis (default: ``eval/loss``).
    step_key:
        The history column to use as the x-axis (default: ``train/global_step``).
    """
    import wandb

    api = wandb.Api()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for variant, run_id in run_ids.items():
        path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
        try:
            run = api.run(path)
        except Exception as exc:
            print(f"  [WARN] Could not fetch run {path}: {exc}")
            continue

        history = run.history(samples=5000, pandas=True)

        cols_available = history.columns.tolist()
        # Resolve step key
        sk = step_key if step_key in cols_available else "_step"
        # Resolve metric key – try the requested one, then common fallbacks
        for mk in [metric, "eval/loss", "validation/loss", "val/loss"]:
            if mk in cols_available:
                break
        else:
            print(f"  [WARN] No eval-loss column found for {variant}. Skipping.")
            continue

        curve = history[[sk, mk]].dropna()
        if curve.empty:
            print(f"  [WARN] Empty curve for {variant} ({mk}).")
            continue

        color = VARIANT_COLORS.get(variant, "#999999")
        label = VARIANT_LABELS.get(variant, variant)
        ax.plot(
            curve[sk], curve[mk],
            color=color, label=label,
            linewidth=1.6, alpha=0.9,
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Training Stability: Validation Loss vs. Training Step", fontweight="bold")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.margins(x=0.05)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved training curves → {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 – Attention Dominance vs. Layer Index
# ---------------------------------------------------------------------------


def plot_layerwise_dominance(
    reports: dict[str, dict[str, Any]],
    out_path: Path,
    variants: list[str] | None = None,
) -> None:
    """First-token attention dominance per *layer index* (not normalised depth).

    Parameters
    ----------
    reports:
        Result dict from ``build_report`` keyed by variant name.
    out_path:
        Destination path.
    variants:
        Subset of variants to include.  Defaults to ``["baseline", "softmax1"]``.
    """
    if variants is None:
        variants = ["vanilla_gpt2", "baseline", "softmax1", "orthoadam", "softmax1_ortho"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for variant in variants:
        if variant not in reports:
            print(f"  [SKIP] No report for '{variant}'")
            continue
        dom_avg = np.array(reports[variant]["attn/dominance_per_layer"]) * 100
        layers = np.arange(len(dom_avg))
        color = VARIANT_COLORS.get(variant, "#999999")
        label_base = VARIANT_LABELS.get(variant, variant)

        # Plot Average (Solid)
        ax.plot(
            layers, dom_avg,
            marker=MARKER.get(variant, "o"),
            color=color,
            label=label_base,
            linewidth=2.0,
            markersize=6,
            alpha=1.0,
        )

    n_layers = max(
        len(reports[v]["attn/dominance_per_layer"])
        for v in variants if v in reports
    )
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Avg. First-Token Attention Rate (%)")
    ax.set_title("Attention Sink: Average First-Token Attention Rate per Layer", fontweight="bold")
    ax.set_xlim(-1, n_layers)
    ax.set_xticks(np.arange(n_layers))
    ax.margins(y=0.05)  # Let it breathe, don't stick 0 to the axis
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved attention dominance → {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 – First-Token Kurtosis vs. Layer Index  (log scale)
# ---------------------------------------------------------------------------


def plot_layerwise_kurtosis(
    reports: dict[str, dict[str, Any]],
    out_path: Path,
    variants: list[str] | None = None,
) -> None:
    """First-token hidden-state kurtosis per layer (log y-scale).

    Uses the ``kurtosis/first_per_layer`` key from the report (mean excess
    kurtosis of the first-token activation across channels, per layer).

    Parameters
    ----------
    reports:
        See ``plot_layerwise_dominance``.
    out_path:
        Destination path.
    variants:
        Defaults to ``["baseline", "orthoadam"]``.
    """
    if variants is None:
        variants = ["vanilla_gpt2", "baseline", "softmax1", "orthoadam", "softmax1_ortho"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Reference line: kurtosis of a Gaussian (excess = 0  →  raw = 3)
    ax.axhline(
        3.0, color="#888888", linestyle="--", linewidth=1.1,
        label="Gaussian reference (κ = 3)",
        zorder=0,
    )

    for variant in variants:
        if variant not in reports:
            print(f"  [SKIP] No report for '{variant}'")
            continue
        # prefer first-token kurtosis; fall back to mean-all if missing
        key = "kurtosis/first_per_layer"
        if key not in reports[variant]:
            key = "kurtosis/mean_per_layer"
        kurt = np.array(reports[variant][key])
        # interpretability.py stores *raw* kurtosis (Gaussian ≈ 3) per paper Eq. 2.
        raw = np.clip(kurt, 1e-1, None)  # avoid log(0)
        layers = np.arange(len(raw))
        ax.plot(
            layers, raw,
            marker=MARKER.get(variant, "o"),
            color=VARIANT_COLORS.get(variant, "#999999"),
            label=VARIANT_LABELS.get(variant, variant),
            linewidth=1.8,
            markersize=5,
        )

    n_layers = max(
        len(reports[v].get("kurtosis/first_per_layer",
                           reports[v].get("kurtosis/mean_per_layer", [])))
        for v in variants if v in reports
    )
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("First-Token Kurtosis (log scale)")
    ax.set_title("Outlier Severity: First-Token Hidden-State Kurtosis per Layer", fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlim(-1, n_layers)
    ax.set_xticks(np.arange(n_layers))
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved kurtosis depth curve → {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 – First-Token Max |Activation| vs. Layer Index  (log scale)
# ---------------------------------------------------------------------------


def plot_layerwise_max_activation(
    reports: dict[str, dict[str, Any]],
    out_path: Path,
    variants: list[str] | None = None,
) -> None:
    """First-token maximum |activation| per layer (log y-scale).

    Uses ``activation/max_abs_first_per_layer`` if available, otherwise
    falls back to ``activation/max_abs_per_layer`` (all positions).

    Parameters
    ----------
    reports:
        See ``plot_layerwise_dominance``.
    out_path:
        Destination path.
    variants:
        Defaults to ``["baseline", "orthoadam"]``.
    """
    if variants is None:
        variants = ["vanilla_gpt2", "baseline", "softmax1", "orthoadam", "softmax1_ortho"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for variant in variants:
        if variant not in reports:
            print(f"  [SKIP] No report for '{variant}'")
            continue

        max_act = np.array(reports[variant].get(
            "activation/max_abs_per_layer_first",
            reports[variant].get("activation/max_abs_per_layer_all", []),
        ))
        layers = np.arange(len(max_act))
        color = VARIANT_COLORS.get(variant, "#999999")
        label_base = VARIANT_LABELS.get(variant, variant)

        # Plot Max (Solid)
        ax.plot(
            layers, max_act,
            marker=MARKER.get(variant, "o"),
            color=color,
            label=label_base,
            linewidth=2.2,
            markersize=6,
            alpha=0.9,
        )


    n_layers = max(
        len(reports[v].get("activation/max_abs_per_layer_first",
                           reports[v].get("activation/max_abs_per_layer_all", [])))
        for v in variants if v in reports
    )
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Max |Activation| of First Token (log scale)")
    ax.set_title("Outlier Magnitude: Max Absolute Activation of the First Token per Layer", fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlim(-1, n_layers)
    ax.set_xticks(np.arange(n_layers))
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved max-activation depth curve → {out_path}")


# ---------------------------------------------------------------------------
# Figure 5 – β₂ Sensitivity (kurtosis emergence vs. step)
# ---------------------------------------------------------------------------

# Labels for each condition in the β₂ sensitivity plot.
# Keys are the exact strings used in beta2_sensitivity.json.
BETA2_COLORS = {
    "adam_baseline": "#1f77b4",   # blue  – standard Adam
    "0.999":         "#d62728",   # red
    "0.99":          "#ff7f0e",   # orange
    "0.95":          "#2ca02c",   # green
    "0.90":          "#9467bd",   # purple (optional extra)
}
BETA2_LABELS = {
    "adam_baseline": "Standard Adam (β₂=0.999, baseline)",
    "0.999":         "OrthoAdam β₂=0.999",
    "0.99":          "OrthoAdam β₂=0.99",
    "0.95":          "OrthoAdam β₂=0.95",
    "0.90":          "OrthoAdam β₂=0.90",
}
BETA2_LINES = {
    "adam_baseline": "--",   # dashed to distinguish the "same β₂, different opt" case
    "0.999":         "-",
    "0.99":          "-",
    "0.95":          "-",
    "0.90":          "-",
}


def plot_beta2_sensitivity(
    results: dict[str, list[tuple[int, float]]],
    out_path: Path,
) -> None:
    """Kurtosis vs. training step for different β₂ values.

    The ``results`` dict may contain a special ``"adam_baseline"`` key for the
    standard Adam run (no orthogonal rotation); all other keys are interpreted
    as numeric β₂ values used with OrthoAdam.

    Parameters
    ----------
    results:
        ``{condition_key: [(step, mean_kurtosis), ...]}``

        ``condition_key`` is either ``"adam_baseline"`` or a stringified
        float like ``"0.999"``.
    out_path:
        Destination path.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Preferred order: adam_baseline first, then descending β₂
    preferred_order = ["adam_baseline", "0.999", "0.99", "0.95", "0.90"]
    keys = [k for k in preferred_order if k in results]
    # Append any unexpected keys at the end
    keys += [k for k in results if k not in keys]

    for key in keys:
        curve = results[key]
        steps, kurts = zip(*curve)
        # beta2_sensitivity.py already stores raw kurtosis (Gaussian ≈ 3).
        kurts_raw = list(kurts)
        color = BETA2_COLORS.get(key, "#999999")
        label = BETA2_LABELS.get(key, f"β₂ = {key}")
        ls = BETA2_LINES.get(key, "-")
        ax.plot(
            steps, kurts_raw,
            linestyle=ls,
            color=color,
            label=label,
            linewidth=1.8,
            marker="o",
            markersize=4,
        )

    ax.axhline(
        3.0, color="#aaaaaa", linestyle=":", linewidth=1.0,
        label="Gaussian baseline (κ = 3)", zorder=0,
    )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Layer Kurtosis (log scale)")
    ax.set_title("Optimizer Sensitivity to β₂: Mean Hidden-State Kurtosis vs. Training Step", fontweight="bold")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=8)
    ax.margins(x=0.05)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved β₂ sensitivity → {out_path}")


# ---------------------------------------------------------------------------
# Legacy helpers kept for backward-compat with run_analysis.py imports
# ---------------------------------------------------------------------------


def plot_attention_heatmaps(
    snapshots: dict[str, dict[str, Tensor]],
    out_path: Path,
    seq_len: int | None = None,
) -> None:
    """Side-by-side mean attention maps (averaged over all heads and layers)."""
    variants = list(snapshots.keys())
    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)

    for col, variant in enumerate(variants):
        attn_dict = snapshots[variant]["attention_weights"]
        layers_sorted = sorted(attn_dict.keys(), key=lambda x: int(x))
        all_attn = torch.stack([attn_dict[k].squeeze(0) for k in layers_sorted])
        mean_attn = all_attn.mean(dim=(0, 1)).numpy()
        if seq_len is not None:
            mean_attn = mean_attn[:seq_len, :seq_len]
        ax = axes[0, col]
        # Use log scale to see non-sink attention values (often very small)
        norm = mcolors.LogNorm(vmin=max(mean_attn.min(), 1e-5), vmax=max(mean_attn.max(), 1e-4))
        T = mean_attn.shape[0]
        im = ax.imshow(
            mean_attn, aspect="equal", cmap="Blues", norm=norm,
            origin="upper", extent=[0, T, T, 0], interpolation="nearest"
        )
        ax.set_xlim(0, T)
        ax.set_ylim(T, 0)
        ax.grid(False)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_title(VARIANT_LABELS.get(variant, variant))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Mean Attention Map Averaged Over All Layers and Heads", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved attention heatmaps → {out_path}")


def plot_per_layer_attention_heatmaps(
    snapshot: dict[str, Tensor],
    variant: str,
    out_path: Path,
    max_layers: int = 12,
) -> None:
    """Grid of per-layer mean-over-heads attention maps."""
    attn_dict = snapshot["attention_weights"]
    layers_sorted = sorted(attn_dict.keys(), key=lambda x: int(x))[:max_layers]
    n = len(layers_sorted)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
    for idx, layer_key in enumerate(layers_sorted):
        r, c = divmod(idx, ncols)
        attn = attn_dict[layer_key].squeeze(0).mean(dim=0).numpy()
        ax = axes[r, c]
        # Use log scale for better contrast between sink tokens and others
        norm = mcolors.LogNorm(vmin=max(attn.min(), 1e-5), vmax=max(attn.max(), 1e-4))
        T = attn.shape[0]
        ax.imshow(
            attn, aspect="equal", cmap="Blues", norm=norm,
            origin="upper", extent=[0, T, T, 0], interpolation="nearest"
        )
        ax.set_xlim(0, T)
        ax.set_ylim(T, 0)
        ax.grid(False)
        ax.set_title(f"Layer {layer_key}")
        if r == nrows - 1:
            ax.set_xlabel("Key Position")
        if c == 0:
            ax.set_ylabel("Query Position")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(f"Per-Layer Attention Maps — {VARIANT_LABELS.get(variant, variant)}", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved per-layer attention heatmaps → {out_path}")


def plot_attention_heatmaps_with_sink_profile(
    snapshots: dict[str, dict[str, Tensor]],
    out_path: Path,
    seq_len: int | None = None,
) -> None:
    """Attention heatmap + first-token sink profile strip side by side.

    For each variant:
    - Left panel: mean attention map (log scale, Blues cmap)
    - Right panel: vertical bar showing attn[q, 0] for each query q,
      making the attention-sink strength directly readable and comparable.
    """
    variants = list(snapshots.keys())
    n = len(variants)
    # 2 subplots per variant: heatmap (width 4) + profile strip (width 1)
    width_ratios = [4, 1] * n
    fig, axes = plt.subplots(
        1, 2 * n,
        figsize=(5.5 * n, 4.5),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.05},
        squeeze=False,
    )

    for col, variant in enumerate(variants):
        attn_dict = snapshots[variant]["attention_weights"]
        layers_sorted = sorted(attn_dict.keys(), key=lambda x: int(x))
        all_attn = torch.stack([attn_dict[k].squeeze(0) for k in layers_sorted])
        mean_attn = all_attn.mean(dim=(0, 1)).numpy()
        if seq_len is not None:
            mean_attn = mean_attn[:seq_len, :seq_len]

        T = mean_attn.shape[0]
        ax_map = axes[0, 2 * col]
        ax_prof = axes[0, 2 * col + 1]

        # --- Heatmap panel ---
        norm = mcolors.LogNorm(vmin=max(mean_attn.min(), 1e-5), vmax=mean_attn.max())
        im = ax_map.imshow(
            mean_attn, aspect="equal", cmap="Blues", norm=norm,
            origin="upper", extent=[0, T, T, 0], interpolation="nearest",
        )
        ax_map.set_xlim(0, T)
        ax_map.set_ylim(T, 0)
        ax_map.grid(False)
        ax_map.set_xlabel("Key Position")
        ax_map.set_ylabel("Query Position")
        ax_map.set_title(VARIANT_LABELS.get(variant, variant), fontweight="bold")
        # Highlight the first column with a vertical line
        ax_map.axvline(x=1, color="red", linewidth=1.2, linestyle="--", alpha=0.7)

        # --- Sink profile panel ---
        sink_weights = mean_attn[:, 0]   # attn[q, key=0] for each query q
        qs = np.arange(T)
        color = VARIANT_COLORS.get(variant, "#1f77b4")
        ax_prof.barh(qs, sink_weights, color=color, alpha=0.75, height=0.9)
        ax_prof.set_ylim(T, 0)
        ax_prof.set_xlabel("Attn → tok 0", fontsize=8)
        ax_prof.tick_params(axis="y", left=False, labelleft=False)
        ax_prof.tick_params(axis="x", labelsize=7)
        ax_prof.set_xscale("log")
        ax_prof.grid(True, axis="x", alpha=0.3, which="both")
        ax_prof.spines["left"].set_visible(False)

        fig.colorbar(im, ax=ax_prof, fraction=0.25, pad=0.35, label="Attn weight")

    fig.suptitle(
        "Mean Attention Map + First-Token Sink Profile (Averaged Over All Layers & Heads)",
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved attention heatmaps with sink profile → {out_path}")


def plot_hidden_state_heatmap(
    snapshots: dict[str, dict[str, Tensor]],
    out_path: Path,
) -> None:
    """Channel × token-position heatmap of hidden states (log scale)."""
    variants = list(snapshots.keys())
    n = len(variants)

    # Collect mid-layer arrays for all variants to compute a shared scale
    mid_layers: dict[str, str] = {}
    abs_arrays: dict[str, np.ndarray] = {}
    for variant in variants:
        hs_dict = snapshots[variant]["hidden_states"]
        layers_sorted = sorted(hs_dict.keys(), key=lambda x: int(x))
        mid = layers_sorted[len(layers_sorted) // 2]
        mid_layers[variant] = mid
        abs_arrays[variant] = np.abs(hs_dict[mid].squeeze(0).numpy())

    global_vmin = max(min(a.min() for a in abs_arrays.values()), 1e-2)
    global_vmax = max(a.max() for a in abs_arrays.values())
    shared_norm = mcolors.LogNorm(vmin=global_vmin, vmax=global_vmax)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)

    for col, variant in enumerate(variants):
        abs_h = abs_arrays[variant]
        mid_layer = mid_layers[variant]
        D, T_len = abs_h.shape[1], abs_h.shape[0]
        ax = axes[0, col]
        im = ax.imshow(
            abs_h.T, aspect="auto", cmap="hot",
            norm=shared_norm,
            origin="upper", extent=[0, T_len, D, 0], interpolation="nearest"
        )
        ax.set_xlim(0, T_len)
        ax.set_ylim(D, 0)
        ax.grid(False)
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Channel ID")
        ax.set_title(f"{VARIANT_LABELS.get(variant, variant)}\n(Layer {mid_layer})")

    fig.colorbar(im, ax=axes[0, -1], fraction=0.046, pad=0.04)
    fig.suptitle("Hidden-State Activation Magnitudes (|x|, log scale)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved hidden state heatmaps → {out_path}")


def plot_summary_bars(
    reports: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    """Grouped bar chart of the three key scalar metrics across variants."""
    variants = list(reports.keys())
    metrics = [
        ("attn/first_token_dominance_overall", "First Token\nAttn %", lambda x: x * 100),
        ("kurtosis/mean_all_layers", "Mean\nKurtosis", lambda x: x),
        ("activation/max_abs_all", "Max |Act|", lambda x: x),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (key, label, transform) in zip(axes, metrics):
        vals = [transform(reports[v][key]) for v in variants]
        colors = [VARIANT_COLORS.get(v, "#999999") for v in variants]
        bars = ax.bar(
            [VARIANT_LABELS.get(v, v) for v in variants],
            vals, color=colors, edgecolor="black", linewidth=0.5,
        )
        ax.set_ylabel(label)
        ax.set_title(label)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=8,
            )
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Summary of Key Outlier Metrics Across Architectural Variants", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved summary bars → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate all five analysis figures.")
    p.add_argument("--results_dir", type=Path, required=True,
                   help="Dir with per-variant subdirs containing report.json + snapshot.pt")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Output directory for figures.")
    p.add_argument("--wandb_project", type=str, default="xai-outlier-repro")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument(
        "--wandb_runs", type=str, default=None,
        help="Comma-separated variant:run_id pairs, e.g. 'baseline:abc123,softmax1:def456'",
    )
    p.add_argument(
        "--beta2_json", type=Path, default=None,
        help="Path to beta2_sensitivity.json produced by beta2_sensitivity.py",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load per-variant reports / snapshots
    # ------------------------------------------------------------------ #
    reports: dict[str, dict[str, Any]] = {}
    snapshots: dict[str, dict[str, Tensor]] = {}

    for variant_dir in sorted(args.results_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name
        report_path = variant_dir / "report.json"
        if report_path.exists():
            with open(report_path) as f:
                reports[variant] = json.load(f)

        snapshot_path = variant_dir / "snapshot.pt"
        if snapshot_path.exists():
            snapshots[variant] = torch.load(snapshot_path, weights_only=False)

    if not reports:
        print("No reports found. Run run_analysis.py first.")
        return

    print(f"Loaded reports for: {sorted(reports.keys())}")
    print(f"Loaded snapshots for: {sorted(snapshots.keys())}")

    # ------------------------------------------------------------------ #
    # Figure 1 – Training curves (W&B pull)
    # ------------------------------------------------------------------ #
    if args.wandb_runs:
        run_ids: dict[str, str] = {}
        for pair in args.wandb_runs.split(","):
            variant, run_id = pair.strip().split(":")
            run_ids[variant.strip()] = run_id.strip()
        plot_training_curves_from_wandb(
            run_ids, args.wandb_project,
            args.out_dir / "fig1_training_stability.png",
            entity=args.wandb_entity,
        )
    else:
        print("[SKIP] Figure 1: --wandb_runs not provided.")

    # ------------------------------------------------------------------ #
    # Figure 2 – Attention dominance (baseline vs. softmax1)
    # ------------------------------------------------------------------ #
    if reports:
        plot_layerwise_dominance(
            reports,
            args.out_dir / "fig2_attention_dominance.png",
            variants=["vanilla_gpt2", "baseline", "softmax1", "orthoadam", "softmax1_ortho"],
        )

    # ------------------------------------------------------------------ #
    # Figure 3 – First-token kurtosis (baseline vs. orthoadam)
    # ------------------------------------------------------------------ #
    if reports:
        plot_layerwise_kurtosis(
            reports,
            args.out_dir / "fig3_kurtosis_depth.png",
            variants=["vanilla_gpt2", "baseline", "softmax1", "orthoadam", "softmax1_ortho"],
        )

    # ------------------------------------------------------------------ #
    # Figure 4 – First-token max |activation| (baseline vs. orthoadam)
    # ------------------------------------------------------------------ #
    if reports:
        plot_layerwise_max_activation(
            reports,
            args.out_dir / "fig4_max_activation_depth.png",
            variants=["vanilla_gpt2", "baseline", "softmax1", "orthoadam", "softmax1_ortho"],
        )

    # ------------------------------------------------------------------ #
    # Figure 5 – β₂ sensitivity
    # ------------------------------------------------------------------ #
    if args.beta2_json and args.beta2_json.exists():
        with open(args.beta2_json) as f:
            beta2_data = json.load(f)
        plot_beta2_sensitivity(
            beta2_data,
            args.out_dir / "fig5_beta2_sensitivity.png",
        )
    else:
        print("[SKIP] Figure 5: --beta2_json not found.")

    # ------------------------------------------------------------------ #
    # Legacy supplementary figures
    # ------------------------------------------------------------------ #
    if snapshots:
        plot_attention_heatmaps(snapshots, args.out_dir / "supp_attention_heatmaps.png")
        plot_attention_heatmaps_with_sink_profile(
            snapshots, args.out_dir / "supp_attention_heatmaps_sink_profile.png"
        )
        plot_hidden_state_heatmap(snapshots, args.out_dir / "supp_hidden_state_heatmap.png")
        for variant, snap in snapshots.items():
            plot_per_layer_attention_heatmaps(
                snap, variant, args.out_dir / f"supp_attn_per_layer_{variant}.png"
            )

    if len(reports) >= 2:
        plot_summary_bars(reports, args.out_dir / "supp_summary_bars.png")

    print(f"\nAll figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
