#!/usr/bin/env bash
# Usage: sbatch scripts/train.sbatch {baseline|softmax1|orthoadam} [beta2]
#
# Trains one variant of the GPT-2 60M outlier reproduction.
#
#SBATCH --job-name=xai-train
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate xai-repro

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"

mkdir -p logs

VARIANT="$1"
BETA2="$2"

case "$VARIANT" in
    baseline|softmax1|orthoadam|softmax1_ortho|vanilla_gpt2)
        echo "Variant: $VARIANT"
        ;;
    *)
        echo "ERROR: variant must be one of: baseline, softmax1, orthoadam, softmax1_ortho, vanilla_gpt2" >&2
        exit 1
        ;;
esac

cd "${SLURM_SUBMIT_DIR}"

CMD_ARGS=("--variant" "$VARIANT" "--config" "configs/gpt2_60m.yaml")

if [[ -n "$BETA2" ]]; then
    echo "Overriding beta2 to $BETA2"
    CMD_ARGS+=("--beta2" "$BETA2")
    # Add beta2 to run name and output dir to avoid collisions
    RUN_NAME="${VARIANT}-b2-${BETA2}"
    OUT_DIR="runs/${VARIANT}_b2_${BETA2}"
    CMD_ARGS+=("--run_name" "$RUN_NAME" "--output_dir" "$OUT_DIR")
else
    CMD_ARGS+=("--output_dir" "runs/${VARIANT}")
fi

echo "Training job=${SLURM_JOB_ID} node=$(hostname)"
echo "Command: python -m xai_repro.train ${CMD_ARGS[*]}"

python -m xai_repro.train "${CMD_ARGS[@]}"
