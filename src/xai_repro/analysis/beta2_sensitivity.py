"""β₂ sensitivity study: kurtosis emergence under different Adam β₂ values.

The paper hypothesises that the exponentially decaying average of the
second moment in Adam is the primary contributor to outlier activations
(Section 4.1).  This script tests the hypothesis by running short
micro-training runs with β₂ ∈ {0.90, 0.95, 0.99, 0.999} using OrthoAdam,
plus a **standard AdamW baseline** (same β₂=0.999, no orthogonal rotation),
and tracks mean hidden-state kurtosis every ``eval_every`` steps.

Does NOT require convergence — the claim is about *outlier emergence*
in early training, not final performance.

Usage::

    python -m xai_repro.analysis.beta2_sensitivity \
        --config configs/gpt2_60m.yaml \
        --max_steps 2000 \
        --eval_every 200 \
        --out_dir analysis_results/beta2_sensitivity
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup

from xai_repro.analysis.interpretability import HiddenStateStats
from xai_repro.data import load_c4
from xai_repro.model import load_config
from xai_repro.optim import OrthoAdam

# ---------------------------------------------------------------------------
# Lightweight kurtosis probe (no attention hooks — just hidden states)
# ---------------------------------------------------------------------------


@torch.no_grad()
def quick_kurtosis(
    model: GPT2LMHeadModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 8,
) -> float:
    """Compute mean kurtosis across all layers with minimal overhead."""
    n_layers = model.config.n_layer
    n_embd = model.config.n_embd
    stats = HiddenStateStats(n_layers=n_layers, n_embd=n_embd)

    handles = []
    for i, block in enumerate(model.transformer.h):
        def make_hook(layer_idx: int):
            def hook(_m, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                stats.update(layer_idx, h.detach())
            return hook
        handles.append(block.register_forward_hook(make_hook(i)))

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

    kurt = stats.mean_kurtosis_per_layer("all")
    return float(kurt.mean())


# ---------------------------------------------------------------------------
# Micro-training loop
# ---------------------------------------------------------------------------


def run_micro_train(
    beta2: float,
    cfg: dict,
    data_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    max_steps: int,
    eval_every: int,
    use_orthoadam: bool = False,
) -> list[tuple[int, float]]:
    """Train a fresh GPT-2 for ``max_steps`` with the given β₂.

    Parameters
    ----------
    use_orthoadam:
        If True, use OrthoAdam (the proposed fix).  Otherwise use standard
        AdamW — this is the *control* condition that shows outliers emerge
        regardless of β₂ when the rotation is absent.

    Returns a list of (step, mean_kurtosis) measurements.
    """
    model_cfg = GPT2Config(**cfg["model"])
    model = GPT2LMHeadModel(model_cfg).to(device)
    model.train()

    tcfg = cfg["training"]
    orth_cfg = cfg.get("orthoadam", {})

    if use_orthoadam:
        # Split parameters the same way OrthoAdamTrainer does
        decay_params, no_decay_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or name.endswith(".bias"):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        groups = [
            {"params": decay_params, "weight_decay": tcfg["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = OrthoAdam(
            groups,
            lr=tcfg["learning_rate"],
            betas=(tcfg["adam_beta1"], beta2),
            eps=tcfg["adam_epsilon"],
            weight_decay=tcfg["weight_decay"],
            max_rotate_dim=orth_cfg.get("max_rotate_dim", 4096),
            seed=orth_cfg.get("seed", 0),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tcfg["learning_rate"],
            betas=(tcfg["adam_beta1"], beta2),
            eps=tcfg["adam_epsilon"],
            weight_decay=tcfg["weight_decay"],
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(tcfg["warmup_steps"], max_steps // 5),
        num_training_steps=max_steps,
    )

    tag = (f"OrthoAdam β₂={beta2}" if use_orthoadam else f"AdamW β₂={beta2}")
    results: list[tuple[int, float]] = []
    data_iter = iter(data_loader)

    # Initial kurtosis at step 0
    kurt_0 = quick_kurtosis(model, eval_loader, device, max_batches=4)
    results.append((0, kurt_0))
    print(f"  [{tag}] step=0 kurtosis={kurt_0:.2f}")

    for step in range(1, max_steps + 1):
        model.train()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % eval_every == 0 or step == max_steps:
            kurt = quick_kurtosis(model, eval_loader, device, max_batches=4)
            results.append((step, kurt))
            print(f"  [{tag}] step={step} loss={loss.item():.3f} kurtosis={kurt:.2f}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="β₂ sensitivity analysis for kurtosis emergence.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--beta2_values", type=str, default="0.90,0.95,0.99,0.999",
                    help="Comma-separated β₂ values to test with OrthoAdam.")
    p.add_argument("--no_adam_baseline", action="store_true",
                    help="Skip the standard AdamW baseline condition.")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--wandb_project", type=str, default=None,
                    help="If set, log results to this W&B project.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beta2_values = [float(x.strip()) for x in args.beta2_values.split(",")]

    print(f"Device: {device}")
    print(f"OrthoAdam β₂ values: {beta2_values}")
    print(f"Max steps: {args.max_steps}, eval every: {args.eval_every}")

    data = load_c4(seq_len=cfg["training"]["seq_len"])
    train_loader = DataLoader(
        data.train,
        batch_size=args.batch_size,
        collate_fn=data.collator,
        shuffle=True,
    )
    eval_loader = DataLoader(
        data.validation,
        batch_size=args.batch_size,
        collate_fn=data.collator,
        shuffle=False,
    )

    all_results: dict[str, list[tuple[int, float]]] = {}

    # ------------------------------------------------------------------
    # Condition 0: Standard AdamW baseline (β₂=0.999, no rotation)
    # This is the control that shows outliers are an *Adam* phenomenon.
    # ------------------------------------------------------------------
    if not args.no_adam_baseline:
        print(f"\n{'='*50}")
        print("Running standard AdamW baseline (β₂=0.999, no OrthoAdam)")
        print(f"{'='*50}")
        baseline_curve = run_micro_train(
            beta2=0.999,
            cfg=cfg,
            data_loader=train_loader,
            eval_loader=eval_loader,
            device=device,
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            use_orthoadam=False,
        )
        all_results["adam_baseline"] = baseline_curve

    # ------------------------------------------------------------------
    # Conditions 1-N: OrthoAdam with varying β₂
    # ------------------------------------------------------------------
    for beta2 in beta2_values:
        print(f"\n{'='*50}")
        print(f"Running OrthoAdam with β₂ = {beta2}")
        print(f"{'='*50}")
        curve = run_micro_train(
            beta2=beta2,
            cfg=cfg,
            data_loader=train_loader,
            eval_loader=eval_loader,
            device=device,
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            use_orthoadam=True,
        )
        all_results[str(beta2)] = curve

    # Save results
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "beta2_sensitivity.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate plot
    from xai_repro.analysis.visualize import plot_beta2_sensitivity
    plot_beta2_sensitivity(all_results, args.out_dir / "beta2_sensitivity.png")

    # Optionally log to W&B
    if args.wandb_project:
        import wandb
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        run = wandb.init(project=args.wandb_project, name="beta2-sensitivity", reinit=True)
        for beta2_str, curve in all_results.items():
            for step, kurt in curve:
                wandb.log({
                    f"beta2_sensitivity/kurtosis_b2={beta2_str}": kurt,
                    "beta2_sensitivity/step": step,
                })
        wandb.log({"beta2_sensitivity/plot": wandb.Image(str(args.out_dir / "beta2_sensitivity.png"))})
        wandb.finish()

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'β₂':>8} | {'Initial Kurt':>12} | {'Final Kurt':>12} | {'Ratio':>8}")
    print(f"{'-'*60}")
    for beta2_str, curve in sorted(all_results.items()):
        initial = curve[0][1]
        final = curve[-1][1]
        ratio = final / max(initial, 1e-6)
        print(f"{beta2_str:>8} | {initial:>12.2f} | {final:>12.2f} | {ratio:>8.1f}x")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
