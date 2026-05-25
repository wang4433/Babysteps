#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --job-name=stage5-p2-vlm
#SBATCH --output=slurm/logs/stage5-p2-vlm-%j.out
#SBATCH --error=slurm/logs/stage5-p2-vlm-%j.err
# Stage-5 P2 — VLM attribution + retry, C1 vs C2, three tasks.
#
# Loads InternVL3.5-8B (BF16, ~17 GB) once per task invocation. ~150 episodes
# x 2 conditions x ~5s/call x 3 tasks ≈ ~75 min wall. Reuses cached failure
# frames from datasets/stage5/p2_vlm/<task>/.

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/wang4433/hf_cache}"

for TASK in PushCube-v1 PickCube-v1 StackCube-v1; do
    echo "=== $TASK ==="
    python scripts/stage5_p2_vlm_eval.py \
        --task "$TASK" \
        --episodes "datasets/stage5/p2_vlm/$TASK/episodes.jsonl" \
        --out-dir "reports/stage5/p2_vlm_attribution/$TASK"
done

echo
echo "=== JOB DONE === ($(date))"
