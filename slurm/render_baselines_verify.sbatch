#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=115G
#SBATCH --time=00:40:00
#SBATCH --job-name=render-baselines-verify
#SBATCH --output=slurm/logs/render-baselines-verify-%j.out
#SBATCH --error=slurm/logs/render-baselines-verify-%j.err
# Visual verification before the M3 baseline sweep:
#   - canonical three-phase render for the 3 main-table tasks (2 seeds each)
#   - PushCube baseline contrast: selective vs full_replan_analogue (2 seeds)
# All MP4s land in $OUT_DIR/videos_maniskill.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

OUT_DIR=/scratch/gilbreth/wang4433/render_baseline_verify
mkdir -p "$OUT_DIR"

for TASK in PushCube-v1 PickCube-v1 StackCube-v1; do
  echo "=== canonical render: $TASK ==="
  python scripts/render_stage0_maniskill.py \
    --task "$TASK" --out_dir "$OUT_DIR" --n_episodes 2 --seed_start 0
done

echo "=== PushCube baseline contrast (selective vs full_replan) ==="
python scripts/render_baseline_contrast.py \
  --out_dir "$OUT_DIR" --n_episodes 2 --seed_start 0

echo "--- all MP4s ---"
ls -lh "$OUT_DIR/videos_maniskill"
echo "render verification complete"
