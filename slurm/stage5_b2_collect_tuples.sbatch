#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --job-name=s5-b2-tuples
#SBATCH --output=slurm/logs/b2-tuples-%j.out
#SBATCH --error=slurm/logs/b2-tuples-%j.err
# Stage-5 B.2 — collect REAL seed-mismatch residual-head training tuples on a
# disjoint TRAINING seed range (200-299, vs eval 100-149). Each failed mismatched
# episode emits {demo_face, correct_face, residual_xy (non-privileged observed
# feedback), failure_predicate}. The residual head is then trained on the login
# node (stage5_train_residual_revise_head.py). orient_control 4-way (--axes xy).
set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python scripts/stage5_natural_loop_eval.py \
  --task PushCube-v1 --axes xy --eval-seeds 200-299 --demo-seed-offset 500 \
  --mismatch random \
  --revisers feedback_residual,oracle_value \
  --dump-tuples datasets/stage5/four_way/PushCube-v1/tuples_train.jsonl \
  --out-dir reports/stage5/natural_loop/PushCube-v1/4way_train_collect
echo "=== DONE ==="
