#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --job-name=s5-fv-vjepa-pack
#SBATCH --output=slurm/logs/fv-vjepa-pack-%j.out
#SBATCH --error=slurm/logs/fv-vjepa-pack-%j.err
# Stage-5 full-vision — V-JEPA 2.1 track pack. Cache V-JEPA features on the SAME
# pack-train demo frames (600-679) the DINOv2 pack used, train a STANDARDIZED
# 4-way V-JEPA pack (the feature-norm fix: V-JEPA ~43 vs DINOv2 ~24 -> must
# standardize or the IntentHead underfits), then train the V-JEPA residual head
# on the SAME encoder-agnostic 76 tuples. Apples-to-apples with DINOv2.
set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

FRAMES=datasets/stage5/four_way/PushCube-v1/frames
VFEATS=datasets/stage5/four_way/PushCube-v1/features_vjepa
PACK=models/stage5/p1_vision_4way_vjepa/PushCube-v1
TUPLES=datasets/stage5/four_way/PushCube-v1/tuples_train.jsonl

echo "=== [1/3] cache V-JEPA 2.1 features on pack-train frames (600-679) ==="
python scripts/stage5_cache_vjepa.py --frames-dir "$FRAMES" --out-dir "$VFEATS" \
  --encoder vjepa2_1_vitl16

echo "=== [2/3] train STANDARDIZED 4-way V-JEPA pack (+ held-out CV) ==="
python scripts/stage5_train_4way_pack.py \
  --features-dir "$VFEATS" --feature-suffix vjepa21 \
  --labels "$FRAMES/labels.json" --out-dir "$PACK" --standardize --cv

echo "=== [3/3] train V-JEPA residual ReviseHead (same 76 tuples) ==="
python scripts/stage5_train_residual_revise_head.py \
  --pack-dir "$PACK" --tuples "$TUPLES"
echo "=== DONE ==="
