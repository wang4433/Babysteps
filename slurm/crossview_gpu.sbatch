#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=115G
#SBATCH --time=00:40:00
#SBATCH --job-name=crossview
#SBATCH --output=slurm/logs/crossview-%j.out
#SBATCH --error=slurm/logs/crossview-%j.err

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "=== [1/2] CrossViewPush render (3 episodes) ==="
RENDER_DIR=/scratch/gilbreth/wang4433/render_crossview
python scripts/render_stage0_maniskill.py \
  --task CrossViewPush-v1 \
  --out_dir "$RENDER_DIR" \
  --n_episodes 3 \
  --seed_start 0
ls -lh "$RENDER_DIR/videos_maniskill"

echo "=== [2/2] CrossViewPush data cut (24 seeds) + summarize ==="
DATA_DIR=/scratch/gilbreth/wang4433/data_crossview
python scripts/stage0_collect.py --task CrossViewPush-v1 \
  --n_episodes 24 --seed_start 0 --out_dir "$DATA_DIR"
python scripts/stage0_summarize.py \
  --samples "$DATA_DIR/samples.jsonl" \
  --out_dir "$DATA_DIR"

echo "=== report.json ==="
cat "$DATA_DIR/report.json"
echo "=== DONE ==="
