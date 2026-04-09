"""Wall-clock stop callback.

The gpu-telecom Slurm reservation is capped at 36h. We stop training
with a safety margin so the job has time to run final evaluation, save
the checkpoint, and sync W&B before the scheduler kills it.
"""

from __future__ import annotations

import time
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class WallclockStopCallback(TrainerCallback):
    """Sets ``control.should_training_stop`` once ``max_hours`` have elapsed."""

    def __init__(self, max_hours: float) -> None:
        if max_hours <= 0:
            raise ValueError(f"max_hours must be positive, got {max_hours}")
        self.max_seconds = max_hours * 3600.0
        self._start: float | None = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._start = time.perf_counter()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._start is None:
            return
        if time.perf_counter() - self._start >= self.max_seconds:
            control.should_training_stop = True


__all__ = ["WallclockStopCallback"]
