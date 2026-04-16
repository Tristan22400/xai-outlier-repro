"""Post-training quantization under the paper's exact schemes.

Paper §5.2 (Kaul et al., ICLR 2025) evaluates two quantization families:

1. **Absmax 8-bit, weights AND activations** — symmetric, scale = max|x| / (2^7 − 1).
   Three granularities:

   | Config    | Weights       | Activations   |
   |-----------|---------------|---------------|
   | coarse    | per-tensor    | per-tensor    |
   | moderate  | per-channel   | per-tensor    |
   | fine      | per-channel   | per-channel   |

2. **Zeropoint 4-bit, weights only** — asymmetric, per-channel on output features.
   Activations stay fp32.

The point of the paper: vanilla GPT-2 has huge outlier activations, so the
per-tensor absmax scale is dominated by one channel and the other 511 channels
get crushed by rounding.  softmax-1 and OrthoAdam should leave models that
tolerate coarse (per-tensor) schemes and 4-bit weights with minimal Δppl.

Implementation is *fake* quantization: we quantize then dequantize in fp32 so
we can run on GPU without INT kernels.  This matches the paper's evaluation
methodology (Dettmers et al., 2022, LLM.int8() appendix).

**NOTE:** `torch.ao.quantization.quantize_dynamic` (which this module used to
call) is NOT what the paper measures — dynamic quantization recomputes the
activation scale per batch, which is specifically designed to tolerate
outliers and therefore washes out the signal.  We use *static* calibrated
scales here.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import wandb
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel

from xai_repro.data import load_c4
from xai_repro.model import load_config

# ---------------------------------------------------------------------------
# Core quant/dequant primitives (fake quantization: returns fp32)
# ---------------------------------------------------------------------------


def absmax_quantize(x: Tensor, bits: int, axis: int | None) -> Tensor:
    """Symmetric absmax fake-quant.  Returns dequantized fp32 of same shape.

    Parameters
    ----------
    x     : input tensor.
    bits  : quantization width (e.g. 8).
    axis  : ``None`` → per-tensor; ``int`` → per-channel along that axis.
    """
    qmax = 2 ** (bits - 1) - 1  # signed symmetric range
    if axis is None:
        scale = x.abs().max().clamp(min=1e-12)
    else:
        # Collapse all axes except ``axis``
        reduce_dims = tuple(i for i in range(x.ndim) if i != axis)
        scale = x.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-12)
    scale = scale / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q * scale


def zeropoint_quantize(x: Tensor, bits: int, axis: int | None) -> Tensor:
    """Asymmetric zero-point fake-quant (min–max)."""
    qmax = 2**bits - 1
    if axis is None:
        x_min = x.min()
        x_max = x.max()
    else:
        reduce_dims = tuple(i for i in range(x.ndim) if i != axis)
        x_min = x.amin(dim=reduce_dims, keepdim=True)
        x_max = x.amax(dim=reduce_dims, keepdim=True)
    scale = (x_max - x_min).clamp(min=1e-12) / qmax
    zero_point = torch.round(-x_min / scale)
    q = torch.round(x / scale + zero_point).clamp(0, qmax)
    return (q - zero_point) * scale


# ---------------------------------------------------------------------------
# Fake-quantized Linear wrapper
# ---------------------------------------------------------------------------


@dataclass
class QuantConfig:
    """Static quantization configuration for one PTQ scheme."""

    weight_bits: int = 8
    weight_scheme: Literal["absmax", "zeropoint"] = "absmax"
    weight_axis: int | None = 0  # 0 = per-output-channel for nn.Linear.weight (out, in)
    act_bits: int | None = 8     # ``None`` disables activation quantization
    act_scheme: Literal["absmax", "zeropoint"] = "absmax"
    act_axis: int | None = None  # None → per-tensor; -1 → per-input-channel

    @classmethod
    def absmax_coarse_8bit(cls) -> "QuantConfig":
        return cls(weight_axis=None, act_axis=None)

    @classmethod
    def absmax_moderate_8bit(cls) -> "QuantConfig":
        return cls(weight_axis=0, act_axis=None)

    @classmethod
    def absmax_fine_8bit(cls) -> "QuantConfig":
        return cls(weight_axis=0, act_axis=-1)

    @classmethod
    def zeropoint_4bit_weight_only(cls) -> "QuantConfig":
        return cls(
            weight_bits=4, weight_scheme="zeropoint", weight_axis=0,
            act_bits=None,
        )


def _quant(x: Tensor, bits: int, scheme: str, axis: int | None) -> Tensor:
    if scheme == "absmax":
        return absmax_quantize(x, bits, axis)
    if scheme == "zeropoint":
        return zeropoint_quantize(x, bits, axis)
    raise ValueError(f"Unknown scheme: {scheme}")


class QuantizedLinear(nn.Module):
    """Drop-in fake-quantized replacement for ``nn.Linear``.

    Weights are statically quantized once at construction.  Activations are
    quantized on-the-fly using a *pre-computed* scale from calibration — not
    per-batch, to match paper semantics.

    Stores ``act_absmax`` (scalar or per-channel) populated during calibration.
    """

    def __init__(self, linear: nn.Linear, cfg: QuantConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Weight quantization is independent of calibration data.
        w_q = _quant(linear.weight.data, cfg.weight_bits, cfg.weight_scheme, cfg.weight_axis)
        self.weight = nn.Parameter(w_q, requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # Activation calibration tensor — filled in by calibrate() later.
        # Shape: () for per-tensor, (in_features,) for per-input-channel.
        if cfg.act_bits is not None:
            init = torch.zeros(1) if cfg.act_axis is None else torch.zeros(self.in_features)
            self.register_buffer("act_absmax", init)
            self.register_buffer("act_min", init.clone())
            self.register_buffer("act_max", init.clone())
            self._calibrated = False
        else:
            self._calibrated = True

    def _quantize_activation(self, x: Tensor) -> Tensor:
        cfg = self.cfg
        if cfg.act_bits is None or not self._calibrated:
            return x
        qmax = 2 ** (cfg.act_bits - 1) - 1
        if cfg.act_scheme == "absmax":
            if cfg.act_axis is None:
                scale = (self.act_absmax.clamp(min=1e-12) / qmax).item()
                q = torch.round(x / scale).clamp(-qmax, qmax)
                return q * scale
            # per-input-channel: last dim aligns with in_features
            scale = self.act_absmax.clamp(min=1e-12) / qmax  # (in_features,)
            q = torch.round(x / scale).clamp(-qmax, qmax)
            return q * scale
        raise ValueError("zeropoint activation quant not required by paper")

    def forward(self, x: Tensor) -> Tensor:
        x_q = self._quantize_activation(x)
        return torch.nn.functional.linear(x_q, self.weight, self.bias)


# ---------------------------------------------------------------------------
# Model-level PTQ: replace all Linear layers and calibrate
# ---------------------------------------------------------------------------


def _replace_linears(model: nn.Module, cfg: QuantConfig) -> list[QuantizedLinear]:
    """In-place swap every ``nn.Linear`` for ``QuantizedLinear``. Returns the new layers."""
    replaced: list[QuantizedLinear] = []
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and not isinstance(child, QuantizedLinear):
                q = QuantizedLinear(child, cfg).to(child.weight.device)
                setattr(parent, child_name, q)
                replaced.append(q)
    return replaced


@torch.no_grad()
def _calibrate(
    model: nn.Module,
    quant_layers: list[QuantizedLinear],
    loader: DataLoader,
    device: torch.device,
    n_batches: int,
) -> None:
    """Run ``n_batches`` through the fp32-weight-quant model with activation
    quantization disabled, recording absmax of each QuantizedLinear input."""
    # Install pre-forward hooks
    handles = []
    stats: dict[int, dict[str, Tensor]] = {}

    def make_hook(idx: int, layer: QuantizedLinear):
        def hook(_m, inputs):
            x = inputs[0].detach()
            if layer.cfg.act_axis is None:
                cur = x.abs().max()
            else:
                # per-input-channel absmax: reduce all axes except last
                reduce_dims = tuple(range(x.ndim - 1))
                cur = x.abs().amax(dim=reduce_dims)
            s = stats.setdefault(idx, {"absmax": torch.zeros_like(cur)})
            s["absmax"] = torch.maximum(s["absmax"], cur)
            return None
        return hook

    for i, layer in enumerate(quant_layers):
        handles.append(layer.register_forward_pre_hook(make_hook(i, layer)))

    model.eval()
    try:
        for step, batch in enumerate(loader):
            if step >= n_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
    finally:
        for h in handles:
            h.remove()

    for i, layer in enumerate(quant_layers):
        if i in stats:
            layer.act_absmax.copy_(stats[i]["absmax"].to(layer.act_absmax.device))
            layer._calibrated = True


def apply_ptq(
    model: GPT2LMHeadModel,
    cfg: QuantConfig,
    calib_loader: DataLoader | None,
    device: torch.device,
    n_calib_batches: int = 16,
) -> GPT2LMHeadModel:
    """Replace every ``nn.Linear`` in ``model`` with a fake-quantized layer
    and calibrate activation scales on ``calib_loader`` (if activation
    quantization is enabled)."""
    quant_layers = _replace_linears(model, cfg)
    if cfg.act_bits is not None:
        if calib_loader is None:
            raise ValueError("Activation quantization requires a calibration loader.")
        _calibrate(model, quant_layers, calib_loader, device, n_calib_batches)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
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


PTQ_SCHEMES: dict[str, QuantConfig] = {
    "coarse":      QuantConfig.absmax_coarse_8bit(),
    "moderate":    QuantConfig.absmax_moderate_8bit(),
    "fine":        QuantConfig.absmax_fine_8bit(),
    "zeropoint4b": QuantConfig.zeropoint_4bit_weight_only(),
}


def _load_checkpoint(
    checkpoint: Path,
    device: torch.device,
    variant: str | None = None,
    config_path: Path | None = None,
) -> GPT2LMHeadModel:
    """Load a checkpoint into the correct model architecture.

    Checkpoints saved by ``train.py`` use ``build_model()`` which applies
    RMSNormSingle (key ``ln_*.scale``) and bias-free nn.Linear MLP layers.
    HuggingFace's ``from_pretrained`` cannot reconstruct this architecture, so
    we build the model from scratch (via ``build_model``) and load weights
    directly from the safetensors file.

    When ``variant`` is None (legacy callers), we auto-detect architecture from
    the checkpoint's state dict: if ``ln_1.scale`` keys are present, we build
    with RMSNormSingle + no-bias MLPs; otherwise we fall back to a plain HF
    ``from_pretrained`` (standard GPT-2 layout, old checkpoints only).
    """
    from safetensors.torch import load_file as sf_load

    ckpt_sd = sf_load(str(checkpoint / "model.safetensors"), device=str(device))
    uses_rmsnorm = any("ln_1.scale" in k for k in ckpt_sd)

    if uses_rmsnorm:
        # Modern checkpoint: built with build_model() — reconstruct same architecture.
        from xai_repro.model import build_model
        if config_path is None:
            # Try to locate config relative to checkpoint root
            _roots = [checkpoint] + list(checkpoint.parents)[:4]
            for _r in _roots:
                _c = _r / "configs" / "gpt2_60m.yaml"
                if _c.exists():
                    config_path = _c
                    break
        if config_path is None or not config_path.exists():
            raise FileNotFoundError(
                "Cannot locate gpt2_60m.yaml config. "
                "Pass --config or set config_path= explicitly."
            )
        _variant = variant if variant is not None else "baseline"
        # build_model applies: RMSNormSingle, no-bias MLP, and (if softmax1*)
        # inject_softmax1. For PTQ we always build "baseline" (only arch matters
        # for weight loading; softmax1 is re-applied by run_analysis.py later).
        _build_variant = "baseline" if _variant in ("baseline", "orthoadam") else _variant
        model = build_model(_build_variant, config_path).to(device)
        missing, unexpected = model.load_state_dict(ckpt_sd, strict=False)
        if missing:
            print(f"  [_load_checkpoint] Missing keys after load: {missing[:5]}")
        if unexpected:
            print(f"  [_load_checkpoint] Unexpected keys: {unexpected[:5]}")
    else:
        # Legacy checkpoint: standard GPT-2 Conv1D layout (no RMSNorm, MLP has biases).
        model = GPT2LMHeadModel.from_pretrained(
            str(checkpoint),
            attn_implementation="eager",
            ignore_mismatched_sizes=True,
        ).to(device)
        model_sd = model.state_dict()
        patch: dict[str, torch.Tensor] = {}
        for key, ckpt_tensor in ckpt_sd.items():
            if key not in model_sd:
                continue
            model_tensor = model_sd[key]
            if ckpt_tensor.shape == model_tensor.shape:
                patch[key] = ckpt_tensor
            elif ckpt_tensor.ndim == 2 and ckpt_tensor.shape == model_tensor.shape[::-1]:
                patch[key] = ckpt_tensor.T.contiguous()
        if patch:
            model.load_state_dict({**model_sd, **patch}, strict=False)

    return model


@torch.no_grad()
def run_all_schemes(
    checkpoint: Path,
    loader: DataLoader,
    device: torch.device,
    max_eval_batches: int | None = None,
    n_calib_batches: int = 16,
) -> dict[str, float]:
    """Evaluate fp32 PPL + every PTQ scheme.  Returns a flat metric dict."""
    report: dict[str, float] = {}

    # fp32 reference
    model_fp32 = _load_checkpoint(checkpoint, device)
    ppl_fp32 = evaluate_perplexity(model_fp32, loader, device, max_eval_batches)
    report["ptq/ppl_fp32"] = ppl_fp32
    del model_fp32
    torch.cuda.empty_cache()

    for name, cfg in PTQ_SCHEMES.items():
        model_q = _load_checkpoint(checkpoint, device)
        apply_ptq(model_q, cfg, loader, device, n_calib_batches=n_calib_batches)
        ppl = evaluate_perplexity(model_q, loader, device, max_eval_batches)
        report[f"ptq/ppl_{name}"] = ppl
        report[f"ptq/delta_{name}"] = ppl - ppl_fp32
        del model_q
        torch.cuda.empty_cache()

    return report


# Backwards-compatible helper retained so callers of the old API don't break.
def quantize_dynamic_int8(model: GPT2LMHeadModel) -> GPT2LMHeadModel:
    """DEPRECATED: use ``apply_ptq(model, QuantConfig.absmax_fine_8bit(), ...)``.

    Paper §5.2 uses static absmax/zeropoint PTQ, not dynamic INT8.  This shim
    applies the ``fine`` absmax 8-bit scheme with an empty (uncalibrated)
    model — numerically meaningless but keeps callsites alive during refactor.
    """
    _replace_linears(model, QuantConfig.absmax_fine_8bit())
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--calib_batches", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--wandb_run", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_c4(seq_len=cfg["training"]["seq_len"])
    loader = DataLoader(
        data.validation,
        batch_size=args.batch_size,
        collate_fn=data.collator,
        shuffle=False,
    )

    report = run_all_schemes(
        args.checkpoint, loader, device,
        max_eval_batches=args.max_batches,
        n_calib_batches=args.calib_batches,
    )
    for k, v in report.items():
        print(f"  {k:>25s} : {v:.4f}")

    if args.wandb_run is not None:
        wandb.init(project=cfg["training"]["wandb_project"], id=args.wandb_run, resume="must")
        wandb.log(report)
        wandb.finish()


if __name__ == "__main__":
    main()
