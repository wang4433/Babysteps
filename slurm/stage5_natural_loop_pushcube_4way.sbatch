#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --job-name=s5-natloop-4way
#SBATCH --output=slurm/logs/natloop-4way-%j.out
#SBATCH --error=slurm/logs/natloop-4way-%j.err
# Stage-5 — 4-WAY natural-failure seed-decoupled loop (orient_control fix unlocks
# +-y; job 10966523 = 4x100%). Expected: random initial ~25% (P match=1/4);
# same_intent/rule_orthogonal ~0% recovery; feedback_flip partial (only the 180deg
# subset; fails perpendicular); feedback_residual/oracle_value ~high.
set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
for MM in always random never; do
  echo "=== axes=xy mismatch=$MM ==="
  python scripts/stage5_natural_loop_eval.py \
    --task PushCube-v1 --axes xy --eval-seeds 100-149 --demo-seed-offset 500 \
    --mismatch "$MM" \
    --out-dir "reports/stage5/natural_loop/PushCube-v1/4way_mismatch_$MM"
done
echo "=== DONE ==="
