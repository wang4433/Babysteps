#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --job-name=stage5-p2-vlm-fair
#SBATCH --output=slurm/logs/stage5-p2-vlm-fair-%j.out
#SBATCH --error=slurm/logs/stage5-p2-vlm-fair-%j.err
# Stage-5 P2 — FAIR re-run: C1 (ours) vs C2 (free-replan) with the fixed,
# non-straw-man free-form baseline + the three selectivity metrics.
#
# What changed vs job 10842893 (the committed p2_vlm_attribution table):
#   - C2 free-form prompt now enumerates allowed VALUES per factor (removes the
#     C1/C2 asymmetry that caused StackCube C2 parse_failure_rate=0.76).
#   - parse_free_form_output: neutral case/whitespace/quote normalization only
#     (NO synonym-snapping); ONE format-repair re-prompt on first parse-fail;
#     raw VLM text persisted to c2_results.json.
#   - _per_episode_c1/c2 now emit preservation / unnecessary_changes /
#     harmful_changes (harmful measured vs adapter.oracle_correct_intent).
#   C1 (constrained) is UNCHANGED — greedy decoding on the same cached frames,
#   so fair-run C1 should reproduce old C1 byte-for-byte (consistency check).
#
# Writes to reports/stage5/p2_vlm_fair/<task> — the committed
# reports/stage5/p2_vlm_attribution/<task> (old straw-man C2) is left intact
# for the before/after comparison. Reuses cached failure frames from
# datasets/stage5/p2_vlm/<task>/. ~50 min wall (model reload ~3 min x 5).

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/wang4433/hf_cache}"

for TASK in PushCube-v1 PickCube-v1 StackCube-v1 TurnFaucet-v1 CrossViewPush-v1; do
    echo "================= $TASK ================="
    python scripts/stage5_p2_vlm_eval.py \
        --task "$TASK" \
        --episodes "datasets/stage5/p2_vlm/$TASK/episodes.jsonl" \
        --out-dir "reports/stage5/p2_vlm_fair/$TASK"
done

echo
echo "=== JOB DONE === ($(date))"
for TASK in PushCube-v1 PickCube-v1 StackCube-v1 TurnFaucet-v1 CrossViewPush-v1; do
    echo "$TASK  -> reports/stage5/p2_vlm_fair/$TASK/report.md"
done
