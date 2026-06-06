#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --job-name=s5-b2-render-pack
#SBATCH --output=slurm/logs/b2-render-pack-%j.out
#SBATCH --error=slurm/logs/b2-render-pack-%j.err
# Stage-5 B.2 — build the 4-way vision-grounded LatentPack for PushCube.
#   1. render 80 demo clips, 4 directions x 20 (orient_control 4-way push)
#   2. cache DINOv2 ViT-B/14 features (768-d, cls_mean) per clip
#   3. train the 4-way IntentHead + 4 contact_region / object_motion centroids
# Run AFTER the smoke job confirms clean 4-way pushes. The pack feeds the learned
# latent_learned reviser (its residual head is trained separately on real tuples).
set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

FRAMES=datasets/stage5/four_way/PushCube-v1/frames
FEATS=datasets/stage5/four_way/PushCube-v1/features
PACK=models/stage5/p1_vision_4way/PushCube-v1

echo "=== [1/3] render 4-way demo frames (600-679) ==="
python scripts/stage5_render_demo_frames.py \
  --four-way-range 600-679 --out-dir "$FRAMES" --skip-existing

echo "=== [2/3] cache DINOv2 features ==="
python scripts/stage5_cache_dinov2.py \
  --frames-dir "$FRAMES" --out-dir "$FEATS"

echo "=== [3/3] train 4-way LatentPack (with held-out CV separability check) ==="
python scripts/stage5_train_4way_pack.py \
  --features-dir "$FEATS" --labels "$FRAMES/labels.json" \
  --out-dir "$PACK" --cv
echo "=== DONE ==="
