"""Training callbacks.

MFUCallback   — logs tokens/sec and MFU to W&B every logging_steps.
WallclockStopCallback — stops training before Slurm kills the job.
"""

from __future__ import annotations

import time
from typing import Any

import wandb
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

# P100 (sm_60) FLOPs peaks, datasheet values.
P100_FP32_PEAK_FLOPS: float = 4.7e12


class MFUCallback(TrainerCallback):
    """Logs ``throughput/mfu`` and ``throughput/tokens_per_sec`` to W&B."""

    def __init__(
        self,
        n_params: int,
        tokens_per_step: int,
        peak_flops: float = P100_FP32_PEAK_FLOPS,
    ) -> None:
        self.tokens_per_step = tokens_per_step
        self.peak_flops = peak_flops
        self.flops_per_token = 6.0 * float(n_params)
        self._last_time: float | None = None
        self._last_step: int | None = None

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        now = time.perf_counter()
        step = state.global_step
        if self._last_time is None:
            self._last_time, self._last_step = now, step
            return
        if step % args.logging_steps != 0:
            return
        dt = now - self._last_time
        dstep = step - (self._last_step or 0)
        if dt <= 0 or dstep <= 0:
            return
        tps = (dstep * self.tokens_per_step) / dt
        mfu = tps * self.flops_per_token / self.peak_flops
        if wandb.run is not None:
            wandb.log({"throughput/tokens_per_sec": tps, "throughput/mfu": mfu}, step=step)
        self._last_time, self._last_step = now, step


class WallclockStopCallback(TrainerCallback):
    """Sets ``control.should_training_stop`` once ``max_hours`` have elapsed."""

    def __init__(self, max_hours: float) -> None:
        self.max_seconds = max_hours * 3600.0
        self._start: float | None = None

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        self._start = time.perf_counter()

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        if self._start and time.perf_counter() - self._start >= self.max_seconds:
            control.should_training_stop = True


__all__ = ["MFUCallback", "WallclockStopCallback", "P100_FP32_PEAK_FLOPS"]
