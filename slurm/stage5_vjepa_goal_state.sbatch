#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --job-name=s5-vjepa-goalstate
#SBATCH --output=slurm/logs/vjepa-goalstate-%j.out
#SBATCH --error=slurm/logs/vjepa-goalstate-%j.err
# Stage-5 — V-JEPA 2.1 on StackCube `goal_state` (the factor that ACTUALLY drives
# StackCube execution: skills/stack.py:127 dispatches on goal_state).
#
# WHY: the StackCube babystep loop is driven by goal_state, NOT object_motion
# (object_motion is decorative — 0 refs in stackcube_runner). object_motion is
# the only vision-groundable VARYING factor in the loop cut, but the controller
# ignores it; goal_state drives execution but its initial_intent is constant
# (can't ground) and on real CLIPS the Stage-0 demo deliberately HIDES the
# vertical stack motion. So whether V-JEPA can read goal_state from the CLIP is
# the question that decides if a V-JEPA-load-bearing StackCube loop is possible.
#
# Baseline (committed, same script/protocol/seeds 0-59, n=120, majority 0.500):
#   DINOv3-L@512 clip ladder: deployed spatial_mean 0.717, best (last5_mean) 0.825
#   DINOv2 config (static ceiling): 0.99
#   -> all CLIP cells FAIL the 0.90 gate ("real-clip caps ~0.82, encoder-invariant")
#
# DECISION:
#   V-JEPA clip best-pooling >= 0.90  -> goal_state IS clip-groundable by a video
#       encoder -> a V-JEPA-load-bearing StackCube loop becomes buildable.
#   V-JEPA clip ~= 0.82 (DINO-level)  -> goal_state is ENCODER-INVARIANT clip-blocked
#       (even a video encoder can't read the demo-hidden vertical motion) -> a clean
#       falsification-grade boundary point; the block is the demo design, not capacity.
#
# --standardize: fixes the un-normalized-Z IntentHead optimizer under-read for
# richer encoders (V-JEPA ~43 vs DINO ~24); see vjepa_object_motion/FINDINGS.md.
# Weights (~5 GB ViT-L) already cached in TORCH_HOME/hub/checkpoints (no download).

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

ENC=vjepa2_1_vitl16
SEEDS=0-59

echo "=== [0/3] V-JEPA download+load smoke (validates strict state_dict load) ==="
python scripts/stage5_cache_vjepa.py --check --encoder "$ENC" --device cuda

echo "=== [1/3] CLIP temporal-pooling ladder (deployed repr; vs DINOv3 0.717/0.825) ==="
python scripts/stage5_goal_state_probe.py \
  --task StackCube-v1 --mode clip-pool --seeds "$SEEDS" \
  --encoder "$ENC" --standardize --save-frames \
  --out-dir reports/stage5/goal_state_probe_vjepa21_clip

echo "=== [2/3] STATIC CONFIG ceiling (vs DINOv2 0.99 — is the block the demo or the encoder?) ==="
python scripts/stage5_goal_state_probe.py \
  --task StackCube-v1 --mode config --seeds "$SEEDS" \
  --encoder "$ENC" --standardize --save-frames \
  --out-dir reports/stage5/goal_state_probe_vjepa21_config

echo "=== goal_state V-JEPA result rows ==="
echo "--- CLIP ladder ---"
sed -n '/Temporal-pooling ladder/,/Honesty/p' reports/stage5/goal_state_probe_vjepa21_clip/StackCube-v1/report.md || true
echo "--- CONFIG ceiling ---"
grep -hE "global_dino|IntentHead-CV|Verdict|PASS|FAIL|WEAK" reports/stage5/goal_state_probe_vjepa21_config/StackCube-v1/report.md || true
echo "=== DONE ==="
