#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --job-name=stage5-goalstate-probe
#SBATCH --output=slurm/logs/stage5-goalstate-probe-%j.out
#SBATCH --error=slurm/logs/stage5-goalstate-probe-%j.err
# Stage-5 — StackCube goal_state pixel-separability probe.
#
# Necessary-condition ceiling for latent-grounding goal_state: can frozen
# DINOv2 spatial_mean (the deployed P1 encoder) separate the two goal CONFIGS
# — cubeA stacked ON cubeB vs cubeA placed NEAR/beside cubeB? The near-offset
# direction is balanced across the 4 cardinals per seed so absolute shift
# can't proxy the label.
#
# PASS (>=0.90): goal_state IS pixel-groundable -> green-light a faithful
# clip-based latent goal_state collection (the only surviving 2nd-latent-task
# lead; PickCube contact_region is invisible and StackCube object_motion is
# representation-blocked). FAIL: consolidate on PushCube end-to-end.
#
# Renders 2 configs x 60 seeds = 120 frames, DINOv2-encodes each, runs the
# G1-protocol probe ladder. Writes reports/stage5/goal_state_probe/StackCube-v1.
# ~10-15 min wall.

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/wang4433/hf_cache}"

python scripts/stage5_goal_state_probe.py \
    --task StackCube-v1 \
    --seeds 0-59 \
    --out-dir reports/stage5/goal_state_probe \
    --save-frames

echo
echo "=== JOB DONE === ($(date))"
echo "report -> reports/stage5/goal_state_probe/StackCube-v1/report.md"
cat reports/stage5/goal_state_probe/StackCube-v1/report.md 2>/dev/null || true
