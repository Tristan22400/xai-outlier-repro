"""Hyperparameter grid analysis for OrthoAdam.

Reads checkpoints from ``runs/grid/orthoadam_b2{b2}_rd{rd}_lr{lr}/``
and for each cell:
  1. Loads the model and runs interpretability extraction.
  2. Records mean first-token kurtosis and average attention dominance.
  3. Plots a 2-D heatmap (beta2 × max_rotate_dim) per learning rate.

Usage::

    python -m xai_repro.analysis.grid_analysis \\
        --grid_dir runs/grid \\
        --config configs/gpt2_60m.yaml \\
        --out_dir analysis_results/grid \\
        --max_batches 32 --batch_size 4
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from xai_repro.analysis.interpretability import build_report, extract_all
from xai_repro.analysis.ptq_int8 import _load_checkpoint
from xai_repro.data import load_c4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BETA2_VALUES  = [0.9, 0.95, 0.99, 0.999]
MAX_RD_VALUES = [128, 512, 4096]
LR_VALUES     = [3e-4, 1e-3, 3e-3]

RUN_NAME_RE = re.compile(
    r"orthoadam_b2(?P<beta2>[0-9.e+-]+)_rd(?P<rd>\d+)_lr(?P<lr>[0-9.e+-]+)"
)


def _parse_run_name(name: str) -> tuple[float, int, float] | None:
    m = RUN_NAME_RE.match(name)
    if not m:
        return None
    return float(m["beta2"]), int(m["rd"]), float(m["lr"])


def _find_checkpoint(run_dir: Path) -> Path | None:
    final = run_dir / "final"
    if final.exists() and (final / "config.json").exists():
        return final
    checkpoints = sorted(run_dir.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[-1]))
    for ckpt in reversed(checkpoints):
        if (ckpt / "config.json").exists():
            return ckpt
    return None


def extract_metrics(
    ckpt: Path,
    config_path: Path,
    device: torch.device,
    max_batches: int,
    batch_size: int,
) -> dict:
    model = _load_checkpoint(ckpt, device, variant="orthoadam", config_path=config_path)
    data = load_c4(seq_len=256)
    loader = torch.utils.data.DataLoader(
        data.validation, batch_size=batch_size,
        collate_fn=data.collator, num_workers=0,
    )
    attn_stats, hidden_stats, _ = extract_all(model, loader, device, max_batches=max_batches)
    return build_report(attn_stats, hidden_stats, variant="orthoadam")


def plot_grid_heatmaps(
    results: dict[tuple[float, int, float], dict],
    out_dir: Path,
) -> None:
    """For each LR, plot a (beta2 × max_rotate_dim) heatmap of key metrics."""
    metrics = [
        ("kurtosis/mean_all_layers",              "Mean Kurtosis (all layers)"),
        ("attn/first_token_dominance_overall",     "Avg First-Token Attention (%)"),
    ]

    for lr in LR_VALUES:
        lr_results = {(b2, rd): v for (b2, rd, l), v in results.items() if l == lr}
        if not lr_results:
            continue

        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5))
        if len(metrics) == 1:
            axes = [axes]

        for ax, (key, label) in zip(axes, metrics):
            grid = np.full((len(BETA2_VALUES), len(MAX_RD_VALUES)), np.nan)
            for i, b2 in enumerate(BETA2_VALUES):
                for j, rd in enumerate(MAX_RD_VALUES):
                    cell = lr_results.get((b2, rd))
                    if cell is None:
                        continue
                    val = cell.get(key, np.nan)
                    if key == "attn/first_token_dominance_overall":
                        val = val * 100
                    grid[i, j] = val

            im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r")
            ax.set_xticks(range(len(MAX_RD_VALUES)))
            ax.set_xticklabels([str(r) for r in MAX_RD_VALUES])
            ax.set_yticks(range(len(BETA2_VALUES)))
            ax.set_yticklabels([str(b) for b in BETA2_VALUES])
            ax.set_xlabel("max_rotate_dim")
            ax.set_ylabel("β₂")
            ax.set_title(label)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            for i in range(len(BETA2_VALUES)):
                for j in range(len(MAX_RD_VALUES)):
                    if not np.isnan(grid[i, j]):
                        ax.text(j, i, f"{grid[i, j]:.1f}",
                                ha="center", va="center", fontsize=8,
                                color="black")

        lr_str = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        fig.suptitle(
            f"OrthoAdam Hyperparameter Grid — lr = {lr_str}\n"
            f"Robustness of outlier suppression across β₂ and max_rotate_dim",
            fontweight="bold",
        )
        plt.tight_layout()
        fname = out_dir / f"grid_lr{lr_str.replace('-','m')}.png"
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved grid heatmap → {fname}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--grid_dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/gpt2_60m.yaml"))
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--max_batches", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--recompute", action="store_true",
                   help="Recompute even if cached report.json exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: dict[tuple[float, int, float], dict] = {}

    run_dirs = sorted(args.grid_dir.iterdir()) if args.grid_dir.exists() else []
    if not run_dirs:
        print(f"No runs found in {args.grid_dir}. Have the grid jobs finished?")
        return

    for run_dir in run_dirs:
        parsed = _parse_run_name(run_dir.name)
        if parsed is None:
            continue
        beta2, rd, lr = parsed

        cache = args.out_dir / run_dir.name / "report.json"
        if cache.exists() and not args.recompute:
            with open(cache) as f:
                results[(beta2, rd, lr)] = json.load(f)
            print(f"[cache] {run_dir.name}")
            continue

        ckpt = _find_checkpoint(run_dir)
        if ckpt is None:
            print(f"[SKIP] No checkpoint: {run_dir.name}")
            continue

        print(f"Extracting {run_dir.name} ...")
        report = extract_metrics(ckpt, args.config, device, args.max_batches, args.batch_size)
        results[(beta2, rd, lr)] = report

        cell_dir = args.out_dir / run_dir.name
        cell_dir.mkdir(parents=True, exist_ok=True)
        with open(cell_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)

    if not results:
        print("No results to plot.")
        return

    plot_grid_heatmaps(results, args.out_dir)
    print(f"\nDone. Results in {args.out_dir}")


if __name__ == "__main__":
    main()
