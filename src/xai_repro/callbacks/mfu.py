"""Model-FLOPs-Utilization logging for HF Trainer.

MFU is computed with the Chinchilla dense-transformer approximation

    flops_per_token ≈ 6 * N_non_embedding_params

and the measured tokens/sec since the last call. The denominator uses
the **P100 fp32 peak of 4.7 TFLOPs** — not the 9.3 TFLOPs fp16 peak —
because the project runs in fp32 (user-directed: fp16 has been observed
to diverge on these models).
"""

from __future__ import annotations

import time
from typing import Any

import wandb
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

# P100 (sm_60) FLOPs peaks, datasheet values.
P100_FP32_PEAK_FLOPS: float = 4.7e12
P100_FP16_PEAK_FLOPS: float = 9.3e12


def chinchilla_flops_per_token(n_params: int) -> float:
    """6 * N_params (dense forward + backward, ignoring attention O(T^2))."""
    return 6.0 * float(n_params)


class MFUCallback(TrainerCallback):
    """Logs ``throughput/mfu`` and ``throughput/tokens_per_sec`` to W&B.

    Parameters
    ----------
    n_params:
        Parameter count of the model being trained (non-embedding is
        more accurate but whole-model is close enough at this scale).
    tokens_per_step:
        Effective tokens per optimizer step = micro_batch *
        grad_accum_steps * seq_len.
    peak_flops:
        Hardware peak FLOPs to normalize against. Default is the P100
        fp32 peak; override for other hardware.
    """

    def __init__(
        self,
        n_params: int,
        tokens_per_step: int,
        peak_flops: float = P100_FP32_PEAK_FLOPS,
    ) -> None:
        self.n_params = n_params
        self.tokens_per_step = tokens_per_step
        self.peak_flops = peak_flops
        self.flops_per_token = chinchilla_flops_per_token(n_params)
        self._last_time: float | None = None
        self._last_step: int | None = None

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        now = time.perf_counter()
        step = state.global_step

        if self._last_time is None or self._last_step is None:
            self._last_time = now
            self._last_step = step
            return

        if step % args.logging_steps != 0:
            return

        dt = now - self._last_time
        dstep = step - self._last_step
        if dt <= 0 or dstep <= 0:
            return

        tokens_per_sec = (dstep * self.tokens_per_step) / dt
        achieved_flops = tokens_per_sec * self.flops_per_token
        mfu = achieved_flops / self.peak_flops

        if wandb.run is not None:
            wandb.log(
                {
                    "throughput/tokens_per_sec": tokens_per_sec,
                    "throughput/mfu": mfu,
                    "throughput/achieved_tflops": achieved_flops / 1e12,
                },
                step=step,
            )

        self._last_time = now
        self._last_step = step


__all__ = ["P100_FP32_PEAK_FLOPS", "MFUCallback", "chinchilla_flops_per_token"]
