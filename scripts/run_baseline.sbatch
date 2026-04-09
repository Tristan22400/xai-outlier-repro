#!/usr/bin/env bash
#SBATCH --job-name=xai-baseline
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
# shellcheck disable=SC1091
source "${HOME}/.bashrc"
conda activate xai-repro

mkdir -p logs runs/baseline
cd "${SLURM_SUBMIT_DIR}"

python -m xai_repro.train \
    --variant baseline \
    --config configs/gpt2_60m.yaml \
    --output_dir runs/baseline
