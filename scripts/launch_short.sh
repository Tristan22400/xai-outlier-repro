#!/usr/bin/env bash
# Launch 4 main variants for a 2h30 short run.
set -eo pipefail

for variant in baseline softmax1 orthoadam softmax1_ortho; do
    echo "  Submitting $variant..."
    sbatch scripts/train_short.sbatch "$variant"
done

echo "All 4 short jobs submitted. Monitor with: squeue -u \$USER"
