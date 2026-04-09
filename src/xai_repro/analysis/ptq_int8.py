"""Post-training INT8 perplexity degradation.

This is the downstream signal the two interventions are supposed to
improve. We compare fp32 and dynamic-INT8 perplexity on the WikiText-103
validation split; a large ``Δppl = ppl_int8 - ppl_fp32`` means the
model is brittle to quantization (= has outlier features).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel

from xai_repro.data import load_wikitext103
from xai_repro.model import load_config


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int | None = None
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, labels=labels)
        n = labels.numel()
        total_loss += float(out.loss) * n
        total_tokens += n
    return math.exp(total_loss / max(total_tokens, 1))


def quantize_dynamic_int8(model: GPT2LMHeadModel) -> nn.Module:
    """Apply PyTorch dynamic INT8 quantization to all ``nn.Linear`` layers.

    Dynamic quantization keeps weights in INT8 and quantizes activations
    on the fly during inference — the simplest PTQ baseline and the one
    most sensitive to outlier features.
    """
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--wandb_run", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_fp32 = GPT2LMHeadModel.from_pretrained(args.checkpoint).to(device)
    data = load_wikitext103(seq_len=cfg["training"]["seq_len"])
    loader = DataLoader(
        data.validation,
        batch_size=args.batch_size,
        collate_fn=data.collator,
        shuffle=False,
    )

    ppl_fp32 = evaluate_perplexity(model_fp32, loader, device, args.max_batches)

    # Dynamic int8 inference runs on CPU in torch.ao.quantization.
    model_cpu = GPT2LMHeadModel.from_pretrained(args.checkpoint).to("cpu")
    model_int8 = quantize_dynamic_int8(model_cpu)
    cpu_loader = DataLoader(
        data.validation,
        batch_size=args.batch_size,
        collate_fn=data.collator,
        shuffle=False,
    )
    ppl_int8 = evaluate_perplexity(model_int8, cpu_loader, torch.device("cpu"), args.max_batches)

    report = {
        "analysis/ptq/ppl_fp32": ppl_fp32,
        "analysis/ptq/ppl_int8": ppl_int8,
        "analysis/ptq/delta_ppl": ppl_int8 - ppl_fp32,
        "analysis/ptq/relative_delta": (ppl_int8 - ppl_fp32) / max(ppl_fp32, 1e-9),
    }
    print(report)

    if args.wandb_run is not None:
        wandb.init(project=cfg["training"]["wandb_project"], id=args.wandb_run, resume="must")
        wandb.log(report)
        wandb.finish()


if __name__ == "__main__":
    main()
