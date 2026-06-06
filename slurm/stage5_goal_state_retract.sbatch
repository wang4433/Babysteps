#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --job-name=s5-goalstate-retract
#SBATCH --output=slurm/logs/goalstate-retract-%j.out
#SBATCH --error=slurm/logs/goalstate-retract-%j.err
# Stage-5 goal_state — RETRACT-GRIPPER demo render (the decisive lever).
#
# WHY: the high-oblique camera sweep (job 10969709) was FALSIFIED — goal_state is
# viewpoint-invariant (~0.82 best pooling at all oblique angles) because the
# gripper is co-located with the just-placed cubes at the final frame and
# occludes the stack-vs-near relation from EVERY exterior angle. The ARMLESS
# static config ceiling is 0.99, so the signal is there once the gripper is gone.
#
# FIX UNDER TEST: --retract lifts the open gripper clear of the cubes and dwells,
# so the demo's final frames show the clean placement. If final-state pooling
# (final_frame / last5_mean) now clears 0.90, StackCube goal_state becomes a
# genuine 2nd positive latent task (the strongest paper fix per the reviewer panel).
#
# Control (no retract, default camera, committed): deployed spatial_mean 0.633,
# best last5_mean 0.817 -> FAIL. Both runs here are DINOv2 / default camera /
# seeds 0-59 (n=120, majority 0.500); the only difference is --retract.
#
# DECISION:
#   retract final-state pooling >= 0.90 -> goal_state UNBLOCKED by a goal-
#       disambiguating (retract) demo render -> build the StackCube goal_state loop.
#   retract still < 0.90 -> the residual block is execution quality (imperfect
#       stack / near-coincident placements) + small cubes, not just the gripper.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

ENC=dinov2_vitb14
SEEDS=0-59
OUT=reports/stage5/goal_state_retract

echo "=== [1/2] CONTROL: clip-pool, default camera, NO retract (reproduce ~0.63/0.82) ==="
python scripts/stage5_goal_state_probe.py \
  --task StackCube-v1 --mode clip-pool --seeds "$SEEDS" \
  --encoder "$ENC" --save-frames --out-dir "$OUT"

echo "=== [2/2] TEST: clip-pool, default camera, --retract (gripper cleared) ==="
python scripts/stage5_goal_state_probe.py \
  --task StackCube-v1 --mode clip-pool --seeds "$SEEDS" \
  --encoder "$ENC" --retract --save-frames --out-dir "$OUT"

echo ""
echo "=== SUMMARY (control vs retract) ==="
echo "--- CONTROL (no retract) ---"
grep -hE "Deployed|best final-state|POOLING|PARTIAL|REPRESENTATION|^\| \`" \
  "$OUT/StackCube-v1/report.md" 2>/dev/null || echo "  (no control report)"
echo "--- RETRACT ---"
grep -hE "Deployed|best final-state|POOLING|PARTIAL|REPRESENTATION|^\| \`" \
  "$OUT/StackCube-v1/retract/report.md" 2>/dev/null || echo "  (no retract report)"
echo "=== DONE ==="
