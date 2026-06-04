#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --job-name=stage5-goalstate-clip
#SBATCH --output=slurm/logs/stage5-goalstate-clip-%j.out
#SBATCH --error=slurm/logs/stage5-goalstate-clip-%j.err
# Stage-5 — StackCube goal_state CLIP separability probe.
#
# The static-config probe (reports/stage5/goal_state_probe) hit 0.99 on the
# clean final configs. This run tests the DEPLOYED representation: full demo
# CLIPS, DINOv2 spatial_mean pooled over TIME+patches. The risk is that
# clip-pooling (early frames where cubeA is at its start, ~identical across
# classes) dilutes the goal_state signal below the 0.90 gate.
#
# stack = cubeA_on_cubeB onto cubeB (5-wp); near = cube_at_target dropped at
# cubeB.xy + 0.08 along a per-seed balanced cardinal (clean place-beside).
# Both via the real Stage-0 demo executor. PASS confirms the deployed clip
# representation is separable -> green-light the full latent goal_state build.
#
# Writes reports/stage5/goal_state_probe_clip/StackCube-v1. ~15-25 min wall.

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/wang4433/hf_cache}"

python scripts/stage5_goal_state_probe.py \
    --task StackCube-v1 \
    --mode clip \
    --seeds 0-59 \
    --offset 0.08 \
    --out-dir reports/stage5/goal_state_probe_clip \
    --save-frames

echo
echo "=== JOB DONE === ($(date))"
echo "report -> reports/stage5/goal_state_probe_clip/StackCube-v1/report.md"
cat reports/stage5/goal_state_probe_clip/StackCube-v1/report.md 2>/dev/null || true
