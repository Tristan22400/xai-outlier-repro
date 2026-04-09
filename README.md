# xai-outlier-repro

Reproducing two interventions that are claimed to eliminate the
activation-outlier / privileged-basis pathology in small transformer
language models:

1. **softmax-1** (Miller, *Attention Is Off By One*) — replaces
   `softmax(x)` in attention with `exp(x_i) / (1 + Σ exp(x_j))`, letting
   heads "attend to nothing".
2. **OrthoAdam** — performs Adam's per-coordinate moment updates in a
   random orthogonal basis per parameter, breaking the coordinate-wise
   privilege that Adam otherwise injects.

Three ~60M-parameter GPT-2 models are trained under **identical data and
schedule** on WikiText-103, on a single NVIDIA P100 via Slurm, and
compared on three metrics:

| Metric | Implementation |
|---|---|
| Validation perplexity | HF Trainer eval loop |
| Per-channel activation kurtosis | `src/xai_repro/analysis/kurtosis.py` |
| INT8 post-training quantization Δppl | `src/xai_repro/analysis/ptq_int8.py` |

All runs log to the `xai-outlier-repro` W&B project; training health is
verified by querying the W&B API (`analysis/wandb_health.py`) rather
than by tailing logs.

## Layout

```
src/xai_repro/
├── attention/softmax1.py       # GPT2Attention subclass
├── optim/ortho_adam.py         # torch.optim.Optimizer subclass (Kronecker Q)
├── model.py                    # 60M GPT-2 factory
├── data.py                     # WikiText-103 pipeline
├── train.py                    # HF Trainer entrypoint
├── callbacks/{mfu,walltime}.py # MFU logging, 34h stop
└── analysis/                   # wandb_health, kurtosis, ptq_int8
configs/gpt2_60m.yaml           # single source of truth for HPs
scripts/                        # setup_cluster.sh + 3 sbatch files
tests/                          # pytest: softmax1, ortho_adam, mfu
```

## Hyperparameters (all three variants — do not tune per variant)

| | |
|---|---|
| Model | 12 layers, d_model=512, 8 heads, d_ff=2048, tied embeddings |
| Sequence length | 256 |
| Effective batch | 32 seqs × 256 tok = 8 192 tok / step |
| Precision | **fp32** (P100 has no bf16; fp16 has been observed to diverge) |
| Activation checkpointing | enabled |
| Optimizer | AdamW / OrthoAdam, β = (0.9, 0.95), wd = 0.1 |
| LR schedule | cosine, peak 1e-3, min 1e-4, warmup 2000 steps |
| Max steps | 100 000 (capped by 34h wall-clock on P100) |
| Seed | 42 |

## Reproduction

### Local (CPU — tests only)

```bash
pip install -e '.[dev]'
pytest
ruff check src tests
black --check src tests
mypy src
```

### Cluster (gpu-telecom Slurm, P100, 36h reservation)

```bash
ssh gpu-telecom
git clone <this repo>
cd xai-outlier-repro
bash scripts/setup_cluster.sh     # one-shot miniconda + env
sbatch scripts/run_baseline.sbatch
sbatch scripts/run_softmax1.sbatch
sbatch scripts/run_orthoadam.sbatch
```

After each job finishes, verify health:

```bash
python -m xai_repro.analysis.wandb_health --run <run_id>
python -m xai_repro.analysis.kurtosis  --checkpoint runs/<variant>/final --config configs/gpt2_60m.yaml
python -m xai_repro.analysis.ptq_int8  --checkpoint runs/<variant>/final --config configs/gpt2_60m.yaml
```

## Known deviations from the papers

- **WikiText-103** instead of C4 — the P100 budget cannot converge on C4
  in 36h.
- **OrthoAdam vocab axis is identity**: the full `50257² ≈ 2.5 × 10⁹`
  rotation is infeasible. `max_rotate_dim=4096` skips any axis beyond
  that, so only the 512-dim side of the embedding is rotated.
- **fp32 instead of fp16**: P100 has no bf16 and we have seen fp16
  diverge on sub-100M GPT-2 runs at LR 1e-3.

## License

MIT — see [`LICENSE`](LICENSE).
