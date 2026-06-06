#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --job-name=s5-fv-evaldemos
#SBATCH --output=slurm/logs/fv-evaldemos-%j.out
#SBATCH --error=slurm/logs/fv-evaldemos-%j.err
# Stage-5 full-vision — render HELD-OUT eval demo clips + cache BOTH encoders.
# Eval demo seeds 1000-1049 (= exec 100-149 + offset 900) are DISJOINT from the
# pack-train demo seeds 600-679, so the demo-decode is genuinely held-out. All 4
# cardinal directions per seed (--each-seed-all-dirs) so the extractor can look
# up (demo_seed, demo_motion). DINOv2 (768-d) + V-JEPA 2.1 (1024-d) features land
# in the same dir with different suffixes.
set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

FRAMES=datasets/stage5/four_way/PushCube-v1/eval_demo_frames
FEATS=datasets/stage5/four_way/PushCube-v1/eval_demo_features

echo "=== [1/3] render held-out eval demo clips (1000-1049, all 4 dirs) ==="
python scripts/stage5_render_demo_frames.py \
  --four-way-range 1000-1049 --each-seed-all-dirs \
  --out-dir "$FRAMES" --skip-existing

echo "=== [2/3] cache DINOv2 features ==="
python scripts/stage5_cache_dinov2.py --frames-dir "$FRAMES" --out-dir "$FEATS"

echo "=== [3/3] cache V-JEPA 2.1 features ==="
python scripts/stage5_cache_vjepa.py --frames-dir "$FRAMES" --out-dir "$FEATS" \
  --encoder vjepa2_1_vitl16
echo "=== DONE ==="
