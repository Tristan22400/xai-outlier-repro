"""Unit tests for MFU computation."""

from __future__ import annotations

from xai_repro.callbacks import (
    P100_FP32_PEAK_FLOPS,
    chinchilla_flops_per_token,
)


def test_flops_per_token_is_six_n() -> None:
    assert chinchilla_flops_per_token(60_000_000) == 6.0 * 60_000_000


def test_mfu_formula_matches_hand_computation() -> None:
    # 60M params, 2000 tokens/sec ->
    #   achieved = 6 * 60e6 * 2000 = 7.2e11 FLOPs/s
    #   MFU vs 4.7 TFLOPs peak   = 0.1532...
    n_params = 60_000_000
    tokens_per_sec = 2000.0
    achieved = chinchilla_flops_per_token(n_params) * tokens_per_sec
    mfu = achieved / P100_FP32_PEAK_FLOPS
    assert abs(achieved - 7.2e11) < 1e6
    assert 0.15 < mfu < 0.16
