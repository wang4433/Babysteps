#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=115G
#SBATCH --time=02:00:00
#SBATCH --job-name=baselines
#SBATCH --output=slurm/logs/baselines-%j.out
#SBATCH --error=slurm/logs/baselines-%j.err

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python scripts/run_baselines.py \
  --tasks PushCube-v1 PickCube-v1 StackCube-v1 \
  --methods all \
  --n_episodes 24 --seed_start 0 \
  --out_dir datasets/stage0_baselines
echo "baseline sweep complete"
