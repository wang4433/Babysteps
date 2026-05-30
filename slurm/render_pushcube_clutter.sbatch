#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a30
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=115G
#SBATCH --time=00:20:00
#SBATCH --job-name=pushcube-clutter-render
#SBATCH --output=slurm/logs/pushcube-clutter-%j.out
#SBATCH --error=slurm/logs/pushcube-clutter-%j.err
# Phase 1 verification render — PushCube with the new grey-brown clutter
# object in place of the red wall (see redesign_failure_paradigm.md §"Phase 1"
# and babysteps/render/pushcube.py _get_or_build_obstacle / _OBSTACLE_HALF_*).
# Output goes to renders/pushcube_clutter/ so the wall (renders/pushcube/) and
# clutter renders can be compared side-by-side without overwriting.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover

OUT_DIR=renders/pushcube_clutter
mkdir -p "$OUT_DIR"

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/render_stage0_maniskill.py \
    --task PushCube-v1 \
    --out_dir "$OUT_DIR" \
    --n_episodes 3 \
    --seed_start 0

echo "--- output ---"
ls -lh "$OUT_DIR/videos_maniskill"
