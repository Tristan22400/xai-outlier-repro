"""Training entrypoint: ``python -m xai_repro.train --variant {baseline,softmax1,orthoadam}``.

Runs the 60M GPT-2 on WikiText-103 for one of the three variants and
logs to the ``xai-outlier-repro`` W&B project.  The only difference
between variants is the optimizer (OrthoAdam vs AdamW) and attention
function (softmax-1 vs standard) — all hyperparameters are shared.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer, TrainingArguments

from xai_repro.callbacks import MFUCallback, WallclockStopCallback
from xai_repro.data import load_c4
from xai_repro.model import Variant, build_model, count_parameters, load_config
from xai_repro.optim import OrthoAdam

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "gpt2_60m.yaml"


class OrthoAdamTrainer(Trainer):
    """HF Trainer subclass that substitutes OrthoAdam for AdamW."""

    def __init__(self, *args: Any, orthoadam_kwargs: dict[str, Any], **kwargs: Any) -> None:
        self._oa_kwargs = orthoadam_kwargs
        super().__init__(*args, **kwargs)

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            assert self.model is not None
            decay, no_decay = [], []
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim < 2 or name.endswith(".bias"):
                    no_decay.append(p)
                else:
                    decay.append(p)
            self.optimizer = OrthoAdam(
                [
                    {"params": decay, "weight_decay": self.args.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=self.args.learning_rate,
                betas=self._oa_kwargs.get("betas", (self.args.adam_beta1, self.args.adam_beta2)),
                eps=self.args.adam_epsilon,
                **{k: v for k, v in self._oa_kwargs.items() if k != "betas"},
            )
        return self.optimizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one variant of the outlier reproduction.")
    p.add_argument("--variant", choices=("baseline", "softmax1", "orthoadam", "softmax1_ortho", "vanilla_gpt2"), required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="Short smoke-test run (200 steps).")
    p.add_argument("--beta2", type=float, default=None, help="Override Adam beta2.")
    p.add_argument("--run_name", type=str, default=None, help="Override W&B run name.")
    p.add_argument("--wallclock_hours", type=float, default=None, help="Override wallclock_hours from config.")
    p.add_argument("--max_rotate_dim", type=int, default=None, help="Override orthoadam.max_rotate_dim from config.")
    p.add_argument("--lr", type=float, default=None, help="Override learning_rate from config.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["training"]

    max_steps = args.max_steps or tcfg["max_steps"]
    effective_lr = args.lr if args.lr is not None else tcfg["learning_rate"]
    if args.smoke:
        max_steps = 200

    variant: Variant = args.variant
    model = build_model(variant, args.config)
    n_params = count_parameters(model)
    data = load_c4(seq_len=tcfg["seq_len"])

    tokens_per_step = (
        tcfg["per_device_train_batch_size"]
        * tcfg["gradient_accumulation_steps"]
        * tcfg["seq_len"]
    )

    os.environ.setdefault("WANDB_PROJECT", tcfg["wandb_project"])

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        max_steps=max_steps,
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=effective_lr,
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_steps=tcfg["warmup_steps"],
        weight_decay=tcfg["weight_decay"],
        adam_beta1=tcfg["adam_beta1"],
        adam_beta2=args.beta2 if args.beta2 is not None else tcfg["adam_beta2"],
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
        run_name=args.run_name or f"{variant}-v2-seed{tcfg['seed']}",
        dataloader_num_workers=2,
    )

    callbacks = [
        MFUCallback(n_params=n_params, tokens_per_step=tokens_per_step),
        WallclockStopCallback(max_hours=args.wallclock_hours if args.wallclock_hours is not None else tcfg["wallclock_hours"]),
    ]

    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=data.train,
        eval_dataset=data.validation,
        data_collator=data.collator,
        tokenizer=data.tokenizer,
        callbacks=callbacks,
    )

    if variant in ("orthoadam", "softmax1_ortho"):
        # Override beta2 if provided
        ortho_beta2 = args.beta2 if args.beta2 is not None else tcfg["adam_beta2"]
        
        trainer = OrthoAdamTrainer(
            **trainer_kwargs,
            orthoadam_kwargs={
                "weight_decay": tcfg["weight_decay"],
                "max_rotate_dim": args.max_rotate_dim if args.max_rotate_dim is not None else cfg["orthoadam"]["max_rotate_dim"],
                "seed": cfg["orthoadam"]["seed"],
                "betas": (tcfg["adam_beta1"], ortho_beta2),
            },
        )
    else:
        trainer = Trainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    metrics = trainer.evaluate()
    trainer.log_metrics("final_eval", metrics)
    trainer.save_metrics("final_eval", metrics)


if __name__ == "__main__":
    main()
