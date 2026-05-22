#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=115G
#SBATCH --time=00:20:00
#SBATCH --job-name=stage0-render-stackcube
#SBATCH --output=slurm/logs/render-stackcube-%j.out
#SBATCH --error=slurm/logs/render-stackcube-%j.err
# Stage-0 Sub-project C — StackCube goal under-specification.
# Closes C's acceptance gate item 4 (GPU MP4 spot-check).

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover

OUT_DIR=renders/stackcube
mkdir -p "$OUT_DIR"

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/render_stage0_maniskill.py \
    --task StackCube-v1 \
    --out_dir "$OUT_DIR" \
    --n_episodes 2 \
    --seed_start 0

echo "--- output ---"
ls -lh "$OUT_DIR/videos_maniskill"
