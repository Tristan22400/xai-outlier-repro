"""Per-channel activation kurtosis — the primary outlier diagnostic.

An outlier feature is a channel of the residual stream whose
pre-quantization distribution is extremely heavy-tailed. Excess kurtosis

    E[(x - mean)^4] / var^2 - 3

is the standard summary statistic. A Gaussian has excess kurtosis 0; a
"sink" channel in a stock GPT-2 can exceed several hundred. Both
softmax-1 and OrthoAdam are claimed to cut this by 1-2 orders of
magnitude while preserving validation perplexity.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import torch
import wandb
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel

from xai_repro.data import load_wikitext103
from xai_repro.model import load_config


@torch.no_grad()
def collect_channel_stats(
    model: GPT2LMHeadModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 32,
) -> dict[str, Tensor]:
    """Collect per-channel mean / var / M4 for every block's residual output.

    Uses running Welford-style accumulators to stay O(d) in memory.
    """
    d = model.config.n_embd
    n_layers = model.config.n_layer
    count = torch.zeros(n_layers, device=device)
    mean = torch.zeros(n_layers, d, device=device)
    m2 = torch.zeros(n_layers, d, device=device)
    m4 = torch.zeros(n_layers, d, device=device)

    handles: list[torch.utils.hooks.RemovableHandle] = []

    hook_fn_t = Callable[[nn.Module, tuple[Tensor, ...], Tensor | tuple[Tensor, ...]], None]

    def make_hook(layer_idx: int) -> hook_fn_t:
        def hook(
            _m: nn.Module,
            _inp: tuple[Tensor, ...],
            out: Tensor | tuple[Tensor, ...],
        ) -> None:
            h = out[0] if isinstance(out, tuple) else out
            x = h.reshape(-1, h.shape[-1]).to(torch.float64)
            n = x.shape[0]
            count[layer_idx] += n
            delta = x - mean[layer_idx]
            mean[layer_idx] += delta.sum(dim=0) / count[layer_idx]
            delta2 = x - mean[layer_idx]
            m2[layer_idx] += (delta * delta2).sum(dim=0)
            m4[layer_idx] += (delta2**4).sum(dim=0)

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

    var = m2 / count.unsqueeze(-1).clamp(min=1.0)
    # Excess kurtosis = E[(x-mu)^4] / sigma^4 - 3. Use biased m4/n estimator.
    fourth = m4 / count.unsqueeze(-1).clamp(min=1.0)
    kurt = fourth / (var**2).clamp(min=1e-30) - 3.0
    return {"kurtosis": kurt.detach().cpu(), "var": var.detach().cpu()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--max_batches", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--wandb_run", type=str, default=None, help="Existing run id to log to.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint).to(device)

    data = load_wikitext103(seq_len=cfg["training"]["seq_len"])
    loader = DataLoader(
        data.validation,
        batch_size=args.batch_size,
        collate_fn=data.collator,
        shuffle=False,
    )

    stats = collect_channel_stats(model, loader, device, max_batches=args.max_batches)
    kurt = stats["kurtosis"]  # (n_layers, d)
    max_per_layer = kurt.max(dim=-1).values
    top10_per_layer = kurt.topk(10, dim=-1).values.mean(dim=-1)

    report = {
        "analysis/kurtosis/max_overall": float(kurt.max()),
        "analysis/kurtosis/top10_mean_overall": float(top10_per_layer.mean()),
        "analysis/kurtosis/max_per_layer": max_per_layer.tolist(),
        "analysis/kurtosis/top10_mean_per_layer": top10_per_layer.tolist(),
    }
    print(report)

    if args.wandb_run is not None:
        wandb.init(project=cfg["training"]["wandb_project"], id=args.wandb_run, resume="must")
        wandb.log(report)
        wandb.finish()


if __name__ == "__main__":
    main()
