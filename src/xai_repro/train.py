"""Training entrypoint: ``python -m xai_repro.train --variant {...}``.

Runs the 60M GPT-2 on WikiText-103 for one of the three variants and
logs everything to the ``xai-outlier-repro`` W&B project. Same
hyperparameters for all three variants — the only knob that changes is
which optimizer / attention module is plugged in.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from transformers import (
    Trainer,
    TrainingArguments,
)

from xai_repro.callbacks.mfu import MFUCallback
from xai_repro.callbacks.walltime import WallclockStopCallback
from xai_repro.data import load_wikitext103
from xai_repro.model import Variant, build_model, count_parameters, load_config
from xai_repro.optim.ortho_adam import OrthoAdam

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "gpt2_60m.yaml"


class OrthoAdamTrainer(Trainer):
    """HF Trainer subclass that uses OrthoAdam instead of AdamW."""

    def __init__(self, *args: Any, orthoadam_kwargs: dict[str, Any], **kwargs: Any) -> None:
        self._orthoadam_kwargs = orthoadam_kwargs
        super().__init__(*args, **kwargs)

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            decay_params: list[torch.nn.Parameter] = []
            no_decay_params: list[torch.nn.Parameter] = []
            assert self.model is not None
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim < 2 or name.endswith(".bias"):
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)
            groups = [
                {"params": decay_params, "weight_decay": self.args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]
            self.optimizer = OrthoAdam(
                groups,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                **self._orthoadam_kwargs,
            )
        return self.optimizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one variant of the outlier reproduction.")
    p.add_argument("--variant", choices=("baseline", "softmax1", "orthoadam"), required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--max_steps", type=int, default=None, help="Override config max_steps.")
    p.add_argument("--smoke", action="store_true", help="Short smoke run (200 steps).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["training"]

    max_steps = args.max_steps if args.max_steps is not None else tcfg["max_steps"]
    if args.smoke:
        max_steps = 200

    variant: Variant = args.variant
    model = build_model(variant, args.config)
    n_params = count_parameters(model)

    data = load_wikitext103(seq_len=tcfg["seq_len"])

    tokens_per_step = (
        tcfg["per_device_train_batch_size"] * tcfg["gradient_accumulation_steps"] * tcfg["seq_len"]
    )

    os.environ.setdefault("WANDB_PROJECT", tcfg["wandb_project"])
    run_name = f"{variant}-seed{tcfg['seed']}"

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        max_steps=max_steps,
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_steps=tcfg["warmup_steps"],
        weight_decay=tcfg["weight_decay"],
        adam_beta1=tcfg["adam_beta1"],
        adam_beta2=tcfg["adam_beta2"],
        adam_epsilon=tcfg["adam_epsilon"],
        max_grad_norm=tcfg["max_grad_norm"],
        logging_steps=tcfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=tcfg["eval_steps"],
        save_strategy="steps",
        save_steps=tcfg["save_steps"],
        save_total_limit=2,
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        fp16=tcfg["fp16"],
        bf16=tcfg["bf16"],
        seed=tcfg["seed"],
        report_to=["wandb"],
        run_name=run_name,
        dataloader_num_workers=2,
    )

    callbacks = [
        MFUCallback(n_params=n_params, tokens_per_step=tokens_per_step),
        WallclockStopCallback(max_hours=tcfg["wallclock_hours"]),
    ]

    trainer_cls: type[Trainer]
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": data.train,
        "eval_dataset": data.validation,
        "data_collator": data.collator,
        "tokenizer": data.tokenizer,
        "callbacks": callbacks,
    }
    if variant == "orthoadam":
        trainer_cls = OrthoAdamTrainer
        trainer_kwargs["orthoadam_kwargs"] = {
            "weight_decay": tcfg["weight_decay"],
            "max_rotate_dim": cfg["orthoadam"]["max_rotate_dim"],
            "seed": cfg["orthoadam"]["seed"],
        }
    else:
        trainer_cls = Trainer

    trainer = trainer_cls(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    metrics = trainer.evaluate()
    trainer.log_metrics("final_eval", metrics)
    trainer.save_metrics("final_eval", metrics)


if __name__ == "__main__":
    main()
