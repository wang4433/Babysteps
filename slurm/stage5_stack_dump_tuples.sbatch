#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --job-name=s5-stack-tuples
#SBATCH --output=slurm/logs/stack-tuples-%j.out
#SBATCH --error=slurm/logs/stack-tuples-%j.err
# Dump StackCube goal_state tuples (predicate->goal) for the pooled shared-scorer
# training set (LOTO primary checkpoint = Push contact_region + Stack goal_state,
# g_i masked). PushCube tuples already exist (datasets/.../four_way/.../tuples_train.jsonl).
set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

python scripts/stage5_goalstate_loop_eval.py \
  --pack-dir models/stage5/goalstate/StackCube-v1 \
  --demo-features-dir datasets/stage5/goalstate/StackCube-v1/eval_feats \
  --feature-suffix dinov2_fl --demo-class stack --eval-seeds 200-249 \
  --dump-tuples datasets/stage5/goalstate/StackCube-v1/tuples_train.jsonl \
  --out-dir reports/stage5/goalstate_loop/StackCube-v1
