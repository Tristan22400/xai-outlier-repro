"""Verify that a training run is healthy by querying the W&B API.

Usage
-----
    python -m xai_repro.analysis.wandb_health \\
        --entity <user> --project xai-outlier-repro --run <run_id>

Fails with a non-zero exit code if any assertion fires. Intended to be
run after each Slurm job finishes — log-tailing is not a reliable
verification story (user preference).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import wandb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--entity", type=str, default=None)
    p.add_argument("--project", type=str, default="xai-outlier-repro")
    p.add_argument("--run", type=str, required=True, help="W&B run id (not name).")
    p.add_argument(
        "--max_final_loss",
        type=float,
        default=5.0,
        help="Sanity check: final train loss must be below this.",
    )
    p.add_argument(
        "--min_mfu",
        type=float,
        default=0.15,
        help="Median MFU across the run must exceed this.",
    )
    p.add_argument(
        "--rolling_window",
        type=int,
        default=2000,
        help="Rolling window (in optimizer steps) for the monotone-descent check.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    api = wandb.Api()
    path = (
        f"{args.entity}/{args.project}/{args.run}" if args.entity else f"{args.project}/{args.run}"
    )
    run = api.run(path)
    history = run.history(samples=10_000, pandas=True)

    errors: list[str] = []

    if "train/loss" in history.columns:
        loss = history["train/loss"].dropna().to_numpy()
        if len(loss) == 0:
            errors.append("no train/loss points logged")
        else:
            if not np.all(np.isfinite(loss)):
                errors.append("train/loss contains NaN or Inf")
            if loss[-1] > args.max_final_loss:
                errors.append(f"final train/loss {loss[-1]:.3f} exceeds max {args.max_final_loss}")
            # Rolling-mean monotone descent: compare first vs last window.
            w = min(args.rolling_window, len(loss) // 4)
            if w > 10:
                head = loss[:w].mean()
                tail = loss[-w:].mean()
                if tail >= head:
                    errors.append(
                        f"loss did not decrease: head_mean={head:.3f} tail_mean={tail:.3f}"
                    )
    else:
        errors.append("train/loss key missing from history")

    if "throughput/mfu" in history.columns:
        mfu = history["throughput/mfu"].dropna().to_numpy()
        if len(mfu) > 0:
            median = float(np.median(mfu))
            if median < args.min_mfu:
                errors.append(f"median MFU {median:.3f} below threshold {args.min_mfu:.3f}")
        else:
            errors.append("throughput/mfu logged no points")
    else:
        errors.append("throughput/mfu key missing — MFUCallback not active?")

    if "train/grad_norm" in history.columns:
        gn = history["train/grad_norm"].dropna().to_numpy()
        if len(gn) and not np.all(np.isfinite(gn)):
            errors.append("grad_norm contains NaN or Inf")

    if errors:
        print("HEALTH CHECK FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(f"HEALTH CHECK OK for run {args.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
