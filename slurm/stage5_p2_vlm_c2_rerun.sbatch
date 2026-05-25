#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --job-name=stage5-p2-vlm-c2
#SBATCH --output=slurm/logs/stage5-p2-vlm-c2-%j.out
#SBATCH --error=slurm/logs/stage5-p2-vlm-c2-%j.err
# Stage-5 P2 — re-run C2 only with bumped max_new_tokens.
#
# The first VLM job had max_new_tokens=64 for both prompts; that truncated
# 100% of C2 outputs (the 6-field intent JSON is ~80 tokens), making the
# C2 baseline meaningless (parse_failed=1.0 on every task).
# vlm_attribute.py now splits the limits: C1=32, C2=256. This re-run
# overwrites c2_results.json only; c1_results.json stays from job 10806596.

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
        --out-dir "reports/stage5/p2_vlm_attribution/$TASK" \
        --conditions c2
done

echo
echo "=== JOB DONE === ($(date))"
