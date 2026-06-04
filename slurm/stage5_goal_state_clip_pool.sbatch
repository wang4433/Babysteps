#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --job-name=stage5-goalstate-cpool
#SBATCH --output=slurm/logs/stage5-goalstate-cpool-%j.out
#SBATCH --error=slurm/logs/stage5-goalstate-cpool-%j.err
# Stage-5 — StackCube goal_state CLIP temporal-pooling DIAGNOSTIC.
#
# The static final config separates at 0.99 but the deployed
# spatial_mean-over-clip pooling fell to 0.65. This run re-encodes the SAME
# clips under several temporal poolings (deployed mean / last5 / first_last /
# final_frame) to decide: is the clip FAIL mean-pooling DILUTION (a
# final-state-aware pooling recovers it) or a representation block (all fail)?
#
# goal_state is a FINAL-STATE factor, so mean-over-trajectory may be the wrong
# pooling for it. Outcome:
#   * final-state pooling PASS while spatial_mean FAIL -> dilution; goal_state
#     latent-viable ONLY with a per-factor pooling change (reviewer-visible
#     inconsistency caveat).
#   * all poolings FAIL -> representation block -> consolidate on PushCube.
#
# Writes reports/stage5/goal_state_clip_pool/StackCube-v1. ~15-25 min wall.

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/wang4433/hf_cache}"

python scripts/stage5_goal_state_probe.py \
    --task StackCube-v1 \
    --mode clip-pool \
    --seeds 0-59 \
    --offset 0.08 \
    --out-dir reports/stage5/goal_state_clip_pool

echo
echo "=== JOB DONE === ($(date))"
echo "report -> reports/stage5/goal_state_clip_pool/StackCube-v1/report.md"
cat reports/stage5/goal_state_clip_pool/StackCube-v1/report.md 2>/dev/null || true
