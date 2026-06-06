#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --job-name=s5-goalstate-cam
#SBATCH --output=slurm/logs/goalstate-cam-%j.out
#SBATCH --error=slurm/logs/goalstate-cam-%j.err
# Stage-5 DUAL-CAMERA — does a HIGH-OBLIQUE demo camera unblock StackCube
# `goal_state` (stack-vs-near) on the deployed CLIP representation?
#
# WHY: the static armless config separates stack-vs-near at 0.99 FROM THE SAME
# default oblique camera, but the deployed demo CLIP caps at ~0.72 (spatial_mean)
# / ~0.82 (last5_mean) < 0.90. So the clip block is NOT the camera angle per se —
# it is the gripper/arm occluding the final placement + whole-clip mean-pool
# dilution + (some seeds) near-identical last frames. A high-oblique view that
# looks OVER the descending gripper should recover the final-frame signal the
# default low (~15deg) oblique loses to self-occlusion.
#
# DELIBERATELY NOT NADIR: pure top-down collapses the height that DEFINES
# stacking -> stack-vs-near becomes XY-footprint coincidence (a near-tautology
# with the label). The oblique_* presets keep world-z as image-up.
#
# Baseline (committed, DINOv2/DINOv3 clip ladder, default camera, seeds 0-59,
# n=120, majority 0.500): deployed spatial_mean ~0.72, best last5_mean ~0.82 -> FAIL.
#
# DECISION:
#   any oblique clip-pool best-pooling >= 0.90 -> Camera-1 (high-oblique demo
#       render) UNBLOCKS goal_state -> build the StackCube goal_state loop.
#   all obliques ~= 0.82 (default-level) -> the residual block is pooling /
#       execution-identical-frames, NOT occlusion -> high-oblique alone is not
#       enough; address pooling + render-disambiguation next.
#
# In-job control: the "default" camera run reproduces the committed ~0.72/0.82.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

ENC=dinov2_vitb14
SEEDS=0-59
OUT=reports/stage5/goal_state_camera

for CAM in default oblique_high oblique_higher oblique_corner; do
  echo "=== goal_state clip-pool ladder — camera=$CAM (enc=$ENC, seeds=$SEEDS) ==="
  python scripts/stage5_goal_state_probe.py \
    --task StackCube-v1 --mode clip-pool --seeds "$SEEDS" \
    --encoder "$ENC" --camera "$CAM" --save-frames \
    --out-dir "$OUT"
done

echo ""
echo "=== SUMMARY: deployed spatial_mean vs best final-state pooling per camera ==="
for CAM in default oblique_high oblique_higher oblique_corner; do
  if [ "$CAM" = "default" ]; then
    D="$OUT/StackCube-v1"
  else
    D="$OUT/StackCube-v1/$CAM"
  fi
  echo "--- camera=$CAM ---"
  grep -hE "Deployed|best final-state|POOLING|PARTIAL|REPRESENTATION|^\| \`" \
    "$D/report.md" 2>/dev/null || echo "  (no report at $D)"
done
echo "=== DONE ==="
