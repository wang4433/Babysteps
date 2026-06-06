#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --job-name=s5-fv-eval
#SBATCH --output=slurm/logs/fv-eval-%j.out
#SBATCH --error=slurm/logs/fv-eval-%j.err
# Stage-5 FULL-VISION closed-loop eval on held-out seeds (exec 100-149, demo
# 1000-1049 via offset 900 — disjoint from pack-train 600-679). The initial
# intent is decoded from the cached demo CLIP feature (no JSON/scripted intent);
# natural failure -> learned latent_learned reviser. Two encoder tracks compared:
# DINOv2 (default, no scaler) vs V-JEPA 2.1 (standardized). Target: latent_learned
# on-fail recovery >=0.92, matching the scripted-intent baseline (0.925).
set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

FEATS=datasets/stage5/four_way/PushCube-v1/eval_demo_features
DINO_PACK=models/stage5/p1_vision_4way/PushCube-v1
VJEPA_PACK=models/stage5/p1_vision_4way_vjepa/PushCube-v1
REV=same_intent,feedback_residual,latent_learned,oracle_value

test -d "$FEATS" || { echo "ERROR: missing $FEATS — run fv_render_eval_demos first"; exit 1; }
test -f "$VJEPA_PACK/revise_head_residual.pt" || { echo "ERROR: missing V-JEPA pack/head — run fv_vjepa_pack first"; exit 1; }

for ENC in dinov2 vjepa21; do
  if [ "$ENC" = "dinov2" ]; then PACK=$DINO_PACK; else PACK=$VJEPA_PACK; fi
  for MM in random always never; do
    echo "=== FULL-VISION enc=$ENC mismatch=$MM ==="
    python scripts/stage5_natural_loop_eval.py \
      --task PushCube-v1 --axes xy --eval-seeds 100-149 --demo-seed-offset 900 \
      --mismatch "$MM" \
      --vision-intent --vision-encoder "$ENC" --vision-features-dir "$FEATS" \
      --pack-dir "$PACK" --residual-head "$PACK/revise_head_residual.pt" \
      --revisers "$REV" \
      --out-dir "reports/stage5/natural_loop/PushCube-v1/fullvision_${ENC}_mismatch_${MM}"
  done
done
echo "=== DONE ==="
