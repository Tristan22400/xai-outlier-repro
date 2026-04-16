"""Main analysis orchestrator: runs all interpretability analyses and
generates the full set of figures + W&B logging for the report.

This is the single entrypoint you run after training is complete::

    # 1. Sync checkpoints from the cluster
    rsync -avz gpu-telecom:~/...xai-outlier-repro/runs/ runs/

    # 2. Run the full analysis
    python -m xai_repro.analysis.run_analysis \
        --config configs/gpt2_60m.yaml \
        --runs_dir runs \
        --out_dir analysis_results \
        --wandb_runs baseline:ceoh7scy,softmax1:q9w3rxwr,orthoadam:0ljxhaso

It will:
- Load each variant's final checkpoint
- Extract attention dominance, kurtosis, max activations, heatmap snapshots
- Run INT8 PTQ perplexity comparison
- Generate all figures
- Log everything to W&B
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel

from xai_repro.analysis.interpretability import build_report, extract_all
from xai_repro.analysis.ptq_int8 import run_all_schemes, PTQ_SCHEMES
from xai_repro.analysis.visualize import (
    plot_attention_heatmaps,
    plot_hidden_state_heatmap,
    plot_layerwise_dominance,
    plot_layerwise_kurtosis,
    plot_layerwise_max_activation,
    plot_per_layer_attention_heatmaps,
    plot_summary_bars,
    plot_training_curves_from_wandb,
)
from xai_repro.data import load_c4
from xai_repro.model import load_config

VARIANTS = ["baseline", "softmax1", "orthoadam", "softmax1_ortho", "vanilla_gpt2"]


def _write_table2(all_reports: dict[str, dict], out_dir: Path) -> None:
    """Dump Table-2 (paper §5) replica as Markdown + LaTeX.

    Columns match paper Table 2: %FirstAttn | κ(first) | κ(rest) | |a|(first)
    | |a|(rest), optionally followed by PPL fp32 + Δ for {coarse, moderate,
    fine, zeropoint4b}. Orders variants by the paper's convention.
    """
    order = ["vanilla_gpt2", "baseline", "softmax1", "orthoadam", "softmax1_ortho"]
    rows = [(v, all_reports[v]) for v in order if v in all_reports]
    if not rows:
        return
    has_ptq = any("ptq/ppl_fp32" in r for _, r in rows)

    md_cols = ["Variant", "%1stAttn", "κ(1st)", "κ(>1)", "|a|(1st)", "|a|(>1)"]
    if has_ptq:
        md_cols += ["PPL fp32", "Δcoarse", "Δmoderate", "Δfine", "Δzp4b"]
    md_lines = ["| " + " | ".join(md_cols) + " |", "|" + "|".join(["---"] * len(md_cols)) + "|"]

    tex_lines = [
        "\\begin{tabular}{l" + "r" * (len(md_cols) - 1) + "}",
        "\\toprule",
        " & ".join(md_cols) + " \\\\",
        "\\midrule",
    ]

    for variant, r in rows:
        cells = [
            variant.replace("_", "\\_"),
            f"{r['attn/first_token_dominance_overall']*100:.1f}",
            f"{r['kurtosis/mean_first_token']:.1f}",
            f"{r['kurtosis/mean_rest_tokens']:.1f}",
            f"{r['activation/mean_abs_first_token']:.2f}",
            f"{r['activation/mean_abs_rest_tokens']:.2f}",
        ]
        if has_ptq and "ptq/ppl_fp32" in r:
            fp32 = r["ptq/ppl_fp32"]
            cells += [
                f"{fp32:.1f}",
                f"{r.get('ptq/delta_coarse', float('nan')):+.1f}",
                f"{r.get('ptq/delta_moderate', float('nan')):+.1f}",
                f"{r.get('ptq/delta_fine', float('nan')):+.1f}",
                f"{r.get('ptq/delta_zeropoint4b', float('nan')):+.1f}",
            ]
        elif has_ptq:
            cells += ["—"] * 5
        md_lines.append("| " + " | ".join(cells) + " |")
        tex_lines.append(" & ".join(cells) + " \\\\")

    tex_lines += ["\\bottomrule", "\\end{tabular}"]

    (out_dir / "table2.md").write_text("\n".join(md_lines) + "\n")
    (out_dir / "table2.tex").write_text("\n".join(tex_lines) + "\n")
    print(f"Wrote Table-2 replica → {out_dir / 'table2.md'} and table2.tex")


def find_checkpoint(runs_dir: Path, variant: str) -> Path | None:
    """Locate the best available checkpoint for a variant.

    Checks in order: runs/<variant>/final, runs/<variant>/checkpoint-*, runs/<variant>/
    """
    base = runs_dir / variant
    if not base.exists():
        return None
    final = base / "final"
    if final.exists() and (final / "config.json").exists():
        return final
    # Check for numbered checkpoints
    checkpoints = sorted(base.glob("checkpoint-*"), key=lambda p: p.name)
    if checkpoints:
        last = checkpoints[-1]
        if (last / "config.json").exists():
            return last
    # Maybe the base dir itself is a valid checkpoint
    if (base / "config.json").exists():
        return base
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all analyses across trained variants.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--runs_dir", type=Path, required=True,
                    help="Directory containing variant subdirs with checkpoints.")
    p.add_argument("--out_dir", type=Path, default=Path("analysis_results"))
    p.add_argument("--max_batches", type=int, default=64,
                    help="Max validation batches for interpretability extraction.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--skip_ptq", action="store_true", help="Skip INT8 PTQ analysis.")
    p.add_argument("--wandb_project", type=str, default="xai-outlier-repro")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_runs", type=str, default=None,
                    help="Comma-separated variant:run_id for training curve plots.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    data = load_c4(seq_len=cfg["training"]["seq_len"])
    loader = DataLoader(
        data.validation,
        batch_size=args.batch_size,
        collate_fn=data.collator,
        shuffle=False,
    )

    # ---------------------------------------------------------------
    # Phase 1: Per-variant interpretability extraction
    # ---------------------------------------------------------------
    all_reports: dict[str, dict] = {}
    all_snapshots: dict[str, dict] = {}

    for variant in VARIANTS:
        ckpt = find_checkpoint(args.runs_dir, variant)
        if ckpt is None:
            print(f"[SKIP] No checkpoint found for {variant}")
            continue

        print(f"\n{'='*60}")
        print(f"Analysing: {variant} (checkpoint: {ckpt})")
        print(f"{'='*60}")

        from xai_repro.analysis.ptq_int8 import _load_checkpoint
        from safetensors.torch import load_file as _sf_load
        # _load_checkpoint now auto-detects RMSNorm checkpoints and builds the
        # model via build_model() so all custom modules are already in place.
        model = _load_checkpoint(ckpt, device, variant=variant, config_path=args.config)
        _ckpt_sd = _sf_load(str(ckpt / "model.safetensors"), device=str(device))
        _uses_rmsnorm = any("ln_1.scale" in k for k in _ckpt_sd)
        print(f"  Architecture: {'RMSNormSingle' if _uses_rmsnorm else 'standard LayerNorm'}"
              f"{', softmax-1' if variant in ('softmax1','softmax1_ortho') else ''}"
              f"{', OrthoAdam trained' if variant in ('orthoadam','softmax1_ortho') else ''}")

        attn_stats, hidden_stats, snapshot = extract_all(
            model, loader, device,
            max_batches=args.max_batches,
        )

        report = build_report(attn_stats, hidden_stats, variant)

        # PTQ analysis — paper §5.2: Absmax 8-bit (coarse/moderate/fine) + Zeropoint 4-bit W.
        if not args.skip_ptq:
            print(f"  Running PTQ evaluation ({', '.join(PTQ_SCHEMES)}) ...")
            ptq_report = run_all_schemes(
                ckpt, loader, device,
                max_eval_batches=args.max_batches,
                n_calib_batches=min(16, args.max_batches),
            )
            report.update(ptq_report)

        # Save per-variant results
        variant_dir = args.out_dir / variant
        variant_dir.mkdir(exist_ok=True)
        with open(variant_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)

        snapshot_data = {
            "attention_weights": {str(k): v for k, v in snapshot.attention_weights.items()},
            "hidden_states": {str(k): v for k, v in snapshot.hidden_states.items()},
        }
        torch.save(snapshot_data, variant_dir / "snapshot.pt")

        all_reports[variant] = report
        all_snapshots[variant] = snapshot_data

        # Print summary
        print(f"  First-token attention dominance: {attn_stats.overall():.3%}")
        print(f"  Mean kurtosis (all):   {report['kurtosis/mean_all_layers']:.1f}")
        print(f"  Mean kurtosis (first): {report['kurtosis/mean_first_token']:.1f}")
        print(f"  Max |activation|:      {report['activation/max_abs_all']:.1f}")
        if "ptq/ppl_fp32" in report:
            print(f"  PPL fp32: {report['ptq/ppl_fp32']:.1f}")
            for scheme in PTQ_SCHEMES:
                k_ppl = f"ptq/ppl_{scheme}"
                k_d = f"ptq/delta_{scheme}"
                if k_ppl in report:
                    print(f"    {scheme:>12s}: {report[k_ppl]:.1f}  Δ={report[k_d]:+.2f}")

        del model
        torch.cuda.empty_cache()

    if not all_reports:
        print("\nNo checkpoints found. Nothing to analyse.")
        return

    # ---------------------------------------------------------------
    # Phase 2: Generate figures
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Generating figures")
    print(f"{'='*60}")

    if all_snapshots:
        plot_attention_heatmaps(all_snapshots, figures_dir / "attention_heatmaps.png")
        plot_hidden_state_heatmap(all_snapshots, figures_dir / "hidden_state_heatmap.png")
        for variant, snap in all_snapshots.items():
            plot_per_layer_attention_heatmaps(snap, variant, figures_dir / f"attn_per_layer_{variant}.png")

    plot_layerwise_dominance(all_reports, figures_dir / "layerwise_dominance.png")
    plot_layerwise_kurtosis(all_reports, figures_dir / "layerwise_kurtosis.png")
    plot_layerwise_max_activation(all_reports, figures_dir / "layerwise_max_activation.png")

    if len(all_reports) >= 2:
        plot_summary_bars(all_reports, figures_dir / "summary_bars.png")

    if args.wandb_runs:
        run_ids = {}
        for pair in args.wandb_runs.split(","):
            variant, run_id = pair.split(":")
            run_ids[variant.strip()] = run_id.strip()
        plot_training_curves_from_wandb(
            run_ids, args.wandb_project,
            figures_dir / "training_curves.png",
            entity=args.wandb_entity,
        )

    # ---------------------------------------------------------------
    # Phase 3: Print combined summary table
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    has_ptq = any("ptq/ppl_fp32" in r for r in all_reports.values())
    header = f"{'Variant':>14} | {'%1stAttn':>8} | {'Kurt1':>7} | {'Kurt>1':>7} | {'|a|_1':>7} | {'|a|_>1':>7}"
    if has_ptq:
        header += f" | {'fp32':>6} | {'coarse':>6} | {'moder.':>6} | {'fine':>6} | {'4b':>6}"
    print(header)
    print("-" * len(header))

    for variant, report in all_reports.items():
        row = (
            f"{variant:>14} | "
            f"{report['attn/first_token_dominance_overall']*100:>7.1f}% | "
            f"{report['kurtosis/mean_first_token']:>7.1f} | "
            f"{report['kurtosis/mean_rest_tokens']:>7.1f} | "
            f"{report['activation/mean_abs_first_token']:>7.1f} | "
            f"{report['activation/mean_abs_rest_tokens']:>7.1f}"
        )
        if "ptq/ppl_fp32" in report:
            row += (
                f" | {report['ptq/ppl_fp32']:>6.1f}"
                f" | {report.get('ptq/ppl_coarse', float('nan')):>6.1f}"
                f" | {report.get('ptq/ppl_moderate', float('nan')):>6.1f}"
                f" | {report.get('ptq/ppl_fine', float('nan')):>6.1f}"
                f" | {report.get('ptq/ppl_zeropoint4b', float('nan')):>6.1f}"
            )
        print(row)
    print(f"{'='*70}")

    # Write Table-2 replica (paper §5) as Markdown + LaTeX for the report.
    _write_table2(all_reports, args.out_dir)

    # ---------------------------------------------------------------
    # Phase 4: Log to W&B
    # ---------------------------------------------------------------
    if args.wandb_runs:
        import wandb
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        run = wandb.init(
            project=args.wandb_project,
            name="analysis-summary",
            reinit=True,
        )

        # Log summary table
        columns = [
            "Variant", "1st Token Attn %", "Kurt (1st)", "Kurt (>1)",
            "|a| (1st)", "|a| (>1)",
        ]
        has_ptq = any("ptq/ppl_fp32" in r for r in all_reports.values())
        if has_ptq:
            columns += ["PPL fp32", "PPL coarse", "PPL moderate", "PPL fine", "PPL 4b-zp"]

        table_data = []
        for variant, report in all_reports.items():
            row = [
                variant,
                report["attn/first_token_dominance_overall"] * 100,
                report["kurtosis/mean_first_token"],
                report["kurtosis/mean_rest_tokens"],
                report["activation/mean_abs_first_token"],
                report["activation/mean_abs_rest_tokens"],
            ]
            if has_ptq and "ptq/ppl_fp32" in report:
                row += [
                    report["ptq/ppl_fp32"],
                    report.get("ptq/ppl_coarse"),
                    report.get("ptq/ppl_moderate"),
                    report.get("ptq/ppl_fine"),
                    report.get("ptq/ppl_zeropoint4b"),
                ]
            elif has_ptq:
                row += [None] * 5
            table_data.append(row)

        wandb.log({"analysis/summary_table": wandb.Table(columns=columns, data=table_data)})

        # Log figures
        for fig_path in figures_dir.glob("*.png"):
            wandb.log({f"analysis/figures/{fig_path.stem}": wandb.Image(str(fig_path))})

        wandb.finish()

    print(f"\nAnalysis complete. All results in {args.out_dir}")


if __name__ == "__main__":
    main()
