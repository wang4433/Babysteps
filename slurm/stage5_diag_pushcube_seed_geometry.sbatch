#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --job-name=s5-diag-pushgeom
#SBATCH --output=slurm/logs/diag-pushgeom-%j.out
#SBATCH --error=slurm/logs/diag-pushgeom-%j.err
# Stage-5 — does changing the PushCube seed change the geometry enough to create
# a NATURAL failure for a transplanted (demo-seed) intent, and can the +x-tuned
# push controller push all four cardinal directions?
#
# Gates the natural-failure + seed-decoupled loop rework:
#   Section A wide angle spread  -> raw seed-change creates failures.
#   Section A narrow             -> vary goal direction via injection instead.
#   Section B per-direction succ -> is an injection-varied eval feasible?
#   Section C transplant         -> stale demo intent fails on a different scene.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python scripts/_diag_pushcube_seed_geometry.py \
  --seeds 0-99 --ctrl-seeds 0-9 --demo-seed 0 \
  --out reports/stage5/diag_pushcube_seed_geometry/result.json

echo "=== DONE ==="
