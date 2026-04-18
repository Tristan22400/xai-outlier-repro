"""Hidden-state kurtosis — the primary outlier diagnostic.

Implements the paper's Eq. 2 (Kaul et al., ICLR 2025, arXiv 2410.17174):

    κ_{m,l} = E_d[(X_{m,l,d} − μ_{m,l})⁴] / E_d[(X_{m,l,d} − μ_{m,l})²]²

i.e. for each (layer m, token position l), the kurtosis is computed **across
feature channels** of that single D-dimensional residual-stream vector.  We
report raw kurtosis (Gaussian ≈ 3) so numbers are directly comparable to the
paper's Table 2.  Means are partitioned into {first-token, rest, all} tokens.

This is a thin CLI wrapper around ``HiddenStateStats`` in the
``interpretability`` module so that the two analyses stay in lockstep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel

from xai_repro.analysis.interpretability import HiddenStateStats
from xai_repro.analysis.ptq_int8 import _load_checkpoint
from xai_repro.data import load_c4
from xai_repro.model import load_config


@torch.no_grad()
def collect_hidden_kurtosis(
    model: GPT2LMHeadModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 32,
) -> HiddenStateStats:
    """Run forward hooks and return a populated ``HiddenStateStats``."""
    stats = HiddenStateStats(n_layers=model.config.n_layer, n_embd=model.config.n_embd)

    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_idx: int):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            stats.update(layer_idx, h.detach())
        return hook

    for i, block in enumerate(model.transformer.h):
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

    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--variant", type=str, default=None,
                   choices=("baseline", "softmax1", "orthoadam"),
                   help="Model variant (needed to reconstruct RMSNormSingle architecture).")
    p.add_argument("--max_batches", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--wandb_run", type=str, default=None, help="Existing run id to log to.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use _load_checkpoint instead of from_pretrained so that the RMSNormSingle
    # architecture (produced by build_model) is correctly reconstructed.
    model = _load_checkpoint(args.checkpoint, device, variant=args.variant, config_path=args.config)

    data = load_c4(seq_len=cfg["training"]["seq_len"])
    loader = DataLoader(
        data.validation,
        batch_size=args.batch_size,
        collate_fn=data.collator,
        shuffle=False,
    )

    stats = collect_hidden_kurtosis(model, loader, device, max_batches=args.max_batches)

    report = {
        "analysis/kurtosis/mean_first_token": float(stats.kurtosis_per_layer("first").mean()),
        "analysis/kurtosis/mean_rest_tokens": float(stats.kurtosis_per_layer("rest").mean()),
        "analysis/kurtosis/mean_all_layers": float(stats.kurtosis_per_layer("all").mean()),
        "analysis/kurtosis/first_per_layer": stats.kurtosis_per_layer("first").tolist(),
        "analysis/kurtosis/rest_per_layer": stats.kurtosis_per_layer("rest").tolist(),
        "analysis/kurtosis/all_per_layer": stats.kurtosis_per_layer("all").tolist(),
    }
    print(report)

    if args.wandb_run is not None:
        wandb.init(project=cfg["training"]["wandb_project"], id=args.wandb_run, resume="must")
        wandb.log(report)
        wandb.finish()


if __name__ == "__main__":
    main()
