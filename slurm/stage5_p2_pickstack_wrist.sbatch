#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --job-name=stage5-p2-pickstack-wrist
#SBATCH --output=slurm/logs/stage5-p2-pickstack-wrist-%j.out
#SBATCH --error=slurm/logs/stage5-p2-pickstack-wrist-%j.err
# Stage-5 P2 — PickCube + StackCube multi-image (third-person + panda_wristcam
# first-person) VLM attribution. Unlike PushCube (symbolic planner_failed), these
# tasks STEP the env, so the wrist view sees the actual failure (grasp_slip /
# mis-place) — this is where the egocentric channel can plausibly help.
#
# Per task: (1) render 50 failure frames + wrist view + per-step wrist_rgb .npz
# (~2.6 GB total, measured by the smoke job 10937594); (2) VLM eval multi-image;
# (3) VLM eval --no-wrist A/B on the SAME frames. New out-dirs keep the existing
# single-image P2 results intact. InternVL3.5-8B loads once per eval invocation.

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/wang4433/hf_cache}"

for TASK in PickCube-v1 StackCube-v1; do
    DATA_DIR="datasets/stage5/p2_vlm_wrist/$TASK"
    EPISODES="$DATA_DIR/episodes.jsonl"

    echo "================= [$TASK 1/3] render failure frames + wrist (seeds 100-149) ================="
    python scripts/stage5_p2_render_failure_frames.py \
        --task "$TASK" --seeds 100-149 --out-dir "$DATA_DIR"
    # find (not ls) so a 0-count glob doesn't trip `set -e` via pipefail.
    echo "wrist frames: $(find "$DATA_DIR/frames" -name '*_attempt_wrist.png' 2>/dev/null | wc -l)"
    echo "rollout npz:  $(find "$DATA_DIR/rollouts" -name '*.npz' 2>/dev/null | wc -l)"
    echo "npz total:    $(du -sh "$DATA_DIR/rollouts" 2>/dev/null | cut -f1 || echo n/a)"

    echo "================= [$TASK 2/3] VLM eval — MULTI-IMAGE (wrist + third-person) ================="
    python scripts/stage5_p2_vlm_eval.py \
        --task "$TASK" --episodes "$EPISODES" \
        --out-dir "reports/stage5/p2_vlm_wrist/$TASK"

    echo "================= [$TASK 3/3] VLM eval — SINGLE-IMAGE A/B (--no-wrist) ================="
    python scripts/stage5_p2_vlm_eval.py \
        --task "$TASK" --episodes "$EPISODES" \
        --out-dir "reports/stage5/p2_vlm_wrist_nowrist/$TASK" \
        --no-wrist
done

echo
echo "=== JOB DONE === ($(date))"
for TASK in PickCube-v1 StackCube-v1; do
    echo "$TASK  multi:  reports/stage5/p2_vlm_wrist/$TASK/report.md"
    echo "$TASK  single: reports/stage5/p2_vlm_wrist_nowrist/$TASK/report.md"
done
