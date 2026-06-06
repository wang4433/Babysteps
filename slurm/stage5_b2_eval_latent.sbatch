#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --job-name=s5-b2-eval
#SBATCH --output=slurm/logs/b2-eval-%j.out
#SBATCH --error=slurm/logs/b2-eval-%j.err
# Stage-5 B.2 — eval the LEARNED latent_learned reviser on held-out seeds
# (100-149, disjoint from the 200-299 training tuples). The residual ReviseHead
# edits the contact_region slot in the 4-way vision-grounded latent space and
# nearest-centroid-decodes the corrected face — no hand rule. Target: latent_learned
# ~= feedback_residual (0.94) ~= oracle (0.96), >> same_intent / feedback_flip.
set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

PACK=models/stage5/p1_vision_4way/PushCube-v1
RHEAD=$PACK/revise_head_residual.pt

# Pipeline-ordering guard: the residual head is trained on the LOGIN node
# (stage5_train_residual_revise_head.py) AFTER the collect-tuples job and BEFORE
# this eval. Fail fast with a clear message rather than a confusing torch.load
# FileNotFoundError mid-construction.
test -f "$RHEAD" || { echo "ERROR: missing $RHEAD — run on the login node first:"; \
  echo "  python scripts/stage5_train_residual_revise_head.py --pack-dir $PACK \\"; \
  echo "    --tuples datasets/stage5/four_way/PushCube-v1/tuples_train.jsonl"; exit 1; }
test -d "$PACK" || { echo "ERROR: missing pack $PACK — run stage5_b2_render_4way_pack.sbatch first"; exit 1; }

for MM in random always never; do
  echo "=== axes=xy mismatch=$MM (latent_learned vs hand rule vs oracle) ==="
  python scripts/stage5_natural_loop_eval.py \
    --task PushCube-v1 --axes xy --eval-seeds 100-149 --demo-seed-offset 500 \
    --mismatch "$MM" \
    --revisers same_intent,feedback_flip,feedback_residual,latent_learned,oracle_value \
    --pack-dir "$PACK" --residual-head "$RHEAD" \
    --out-dir "reports/stage5/natural_loop/PushCube-v1/4way_latent_mismatch_$MM"
done
echo "=== DONE ==="
