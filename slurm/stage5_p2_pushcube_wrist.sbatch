#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --job-name=stage5-p2-pushcube-wrist
#SBATCH --output=slurm/logs/stage5-p2-pushcube-wrist-%j.out
#SBATCH --error=slurm/logs/stage5-p2-pushcube-wrist-%j.err
# Stage-5 P2 — PushCube multi-image (third-person + panda_wristcam first-person)
# VLM attribution run. Three steps in one job:
#   1. render failure frames (50 held-out seeds) WITH the wrist view + full
#      per-step wrist_rgb .npz recording  -> datasets/stage5/p2_vlm_wrist/
#   2. VLM eval, multi-image (wrist + third-person)  -> reports/.../p2_vlm_wrist/
#   3. VLM eval, --no-wrist A/B on the SAME frames    -> reports/.../p2_vlm_wrist_nowrist/
#
# A NEW out-dir (p2_vlm_wrist) is used so the existing single-image PushCube
# P2 result (datasets/stage5/p2_vlm/PushCube-v1) stays intact for comparison.
# InternVL3.5-8B (BF16, ~17 GB) loads once per eval invocation (~3 min each).

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/wang4433/hf_cache}"

DATA_DIR="datasets/stage5/p2_vlm_wrist/PushCube-v1"
EPISODES="$DATA_DIR/episodes.jsonl"

echo "=== [1/3] render PushCube failure frames + wrist view (seeds 100-149) ==="
python scripts/stage5_p2_render_failure_frames.py \
    --task PushCube-v1 --seeds 100-149 --out-dir "$DATA_DIR"
# find (not ls) so a 0-count glob doesn't trip `set -e` via pipefail.
echo "wrist frames written:    $(find "$DATA_DIR/frames" -name '*_attempt_wrist.png' 2>/dev/null | wc -l)"
echo "rollout .npz recordings: $(find "$DATA_DIR/rollouts" -name '*.npz' 2>/dev/null | wc -l) (0 expected — PushCube P2 failures are planner_failed, env never steps)"

echo
echo "=== [2/3] VLM eval — MULTI-IMAGE (wrist + third-person) ==="
python scripts/stage5_p2_vlm_eval.py \
    --task PushCube-v1 --episodes "$EPISODES" \
    --out-dir reports/stage5/p2_vlm_wrist/PushCube-v1

echo
echo "=== [3/3] VLM eval — SINGLE-IMAGE A/B (--no-wrist, same frames) ==="
python scripts/stage5_p2_vlm_eval.py \
    --task PushCube-v1 --episodes "$EPISODES" \
    --out-dir reports/stage5/p2_vlm_wrist_nowrist/PushCube-v1 \
    --no-wrist

echo
echo "=== JOB DONE === ($(date))"
echo "multi-image report:  reports/stage5/p2_vlm_wrist/PushCube-v1/report.md"
echo "single-image report: reports/stage5/p2_vlm_wrist_nowrist/PushCube-v1/report.md"
