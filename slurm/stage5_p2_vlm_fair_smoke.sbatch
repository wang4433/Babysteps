#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH --job-name=stage5-p2-vlm-fair-smoke
#SBATCH --output=slurm/logs/stage5-p2-vlm-fair-smoke-%j.out
#SBATCH --error=slurm/logs/stage5-p2-vlm-fair-smoke-%j.err
# Stage-5 P2 — FAIR baseline prompt-synthesis SMOKE on the two tension tasks.
#
# Validates the combined C2 free-form prompt (allowed-value menu + "this failed,
# must change, do not echo") resolves BOTH failure modes at once:
#   - PushCube: C2 must CHANGE approach_direction (not echo) -> succ ~0.96
#   - StackCube: C2 must parse (no out-of-vocab translate_+z) -> parse_fail ~0,
#     and still over-edit (goal_state + object_motion).
# If both hold, run the full 5-task stage5_p2_vlm_fair.sbatch for the final table.
# Writes to a SMOKE dir so the current p2_vlm_fair/ results stay intact.

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/wang4433/hf_cache}"

for TASK in PushCube-v1 StackCube-v1; do
    echo "================= $TASK ================="
    python scripts/stage5_p2_vlm_eval.py \
        --task "$TASK" \
        --episodes "datasets/stage5/p2_vlm/$TASK/episodes.jsonl" \
        --out-dir "reports/stage5/p2_vlm_fair_smoke/$TASK"
done

echo
echo "=== JOB DONE === ($(date))"
