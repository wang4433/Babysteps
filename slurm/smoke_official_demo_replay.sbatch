#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=115G
#SBATCH --time=00:20:00
#SBATCH --job-name=smoke-official-demo
#SBATCH --output=slurm/logs/smoke-official-demo-%j.out
#SBATCH --error=slurm/logs/smoke-official-demo-%j.err
# P0 smoke: render official ManiSkill demos via state-replay (Scope A).
# Proves the unblocked path renders third-person MP4s + a mid-solve PNG
# per task, without ever touching the recorded actions channel.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover

OUT_DIR=/home/wang4433/scratch/babysteps/renders/official_demo_smoke

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/smoke_official_demo_replay.py \
    --tasks PushCube-v1 PickCube-v1 StackCube-v1 \
    --seed 0 \
    --stride 2 \
    --fps 20 \
    --out-dir "$OUT_DIR"

echo "--- artifacts ---"
ls -lh "$OUT_DIR"
