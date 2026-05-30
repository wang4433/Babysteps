#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --job-name=stage5-m3-baselines
#SBATCH --output=slurm/logs/stage5-m3-baselines-%j.out
#SBATCH --error=slurm/logs/stage5-m3-baselines-%j.err
# Stage-5 M3 — procedural baselines main table.
#
# Runs all 7 policies x 3 tasks x 50 seeds. Each episode is a short rollout
# (two attempts max); the bottleneck is ManiSkill env reset + physics step.
# Estimated: ~7 policies x 3 tasks x 50 seeds x ~2s/episode ≈ ~35 min.
# Reserving 2h for safety margin.

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python scripts/stage5_m3_baselines_eval.py \
    --tasks PushCube-v1 PickCube-v1 StackCube-v1 \
    --eval-seeds 100-149 \
    --out-dir reports/stage5/m3_baselines

echo
echo "=== JOB DONE === ($(date))"
