#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --job-name=s5-natloop-push
#SBATCH --output=slurm/logs/natloop-push-%j.out
#SBATCH --error=slurm/logs/natloop-push-%j.err
# Stage-5 — NATURAL-failure, seed-DECOUPLED babystep loop on real PushCube-v1.
# Demo encodes one instance; Franka executes a DIFFERENT instance (demo_seed =
# exec_seed + 500); NO artificial block. Binary +x/-x via injection (native goal
# is fixed +x; controller only pushes x — job 10966085).
#
# Expected (the headline contrast):
#   mismatch=always : initial ~0% ; same_intent ~0% ; rule_orthogonal ~0%
#                     (orthogonal->y-face, controller can't push y);
#                     feedback_flip ~100% ; oracle_value ~100%.
#   mismatch=random : initial ~50% ; recovery only for feedback/oracle.
#   mismatch=never  : initial ~100% (matched-direction control).
# => only revisers that consume EXECUTION FEEDBACK recover; the block hid this.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

for MM in always random never; do
  echo "=== mismatch=$MM ==="
  python scripts/stage5_natural_loop_eval.py \
    --task PushCube-v1 --eval-seeds 100-149 --demo-seed-offset 500 \
    --mismatch "$MM" \
    --out-dir "reports/stage5/natural_loop/PushCube-v1/mismatch_$MM"
done

echo "=== DONE ==="
