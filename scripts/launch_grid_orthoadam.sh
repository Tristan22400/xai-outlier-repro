#!/usr/bin/env bash
# Submit the full OrthoAdam hyperparameter grid (36 short jobs).
#
# Grid:
#   beta2           : 0.9  0.95  0.99  0.999
#   max_rotate_dim  : 128  512   4096
#   learning_rate   : 3e-4 1e-3  3e-3
#
# Each job runs for 2h30 (wallclock stop at 2h15 to guarantee checkpoint save).
set -eo pipefail

BETA2_VALUES=(0.9 0.95 0.99 0.999)
MAX_RD_VALUES=(128 512 4096)
LR_VALUES=(3e-4 1e-3 3e-3)

n=0
for beta2 in "${BETA2_VALUES[@]}"; do
    for max_rd in "${MAX_RD_VALUES[@]}"; do
        for lr in "${LR_VALUES[@]}"; do
            echo "  Submitting beta2=${beta2} max_rotate_dim=${max_rd} lr=${lr} ..."
            sbatch scripts/grid_orthoadam.sbatch "${beta2}" "${max_rd}" "${lr}"
            n=$((n + 1))
        done
    done
done

echo ""
echo "Submitted ${n} grid jobs. Monitor with: squeue -u \$USER"
echo "Results will land in: runs/grid/"
