#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=115G
#SBATCH --time=01:00:00
#SBATCH --job-name=stage5-p1-iconic-pushcube
#SBATCH --output=slurm/logs/stage5-p1-iconic-pushcube-%j.out
#SBATCH --error=slurm/logs/stage5-p1-iconic-pushcube-%j.err
# Stage-5 P1 — iconic per-policy PushCube renders.
# 5 held-out seeds × 4 policies = 20 self-contained "full episode" MP4s.
# Wall time: 60 min. Sibling `render_pushcube.sbatch` does 6 clips in 20 min
# (~17s/phase, ~50s/episode after env load). 20 episodes × ~50s + ~3-5 min
# env load ≈ 22 min expected; 60 min is ~2.7× safety margin.
# Spec:  docs/superpowers/specs/2026-05-24-stage5-p1-iconic-renders-design.md
# Plan:  docs/superpowers/plans/2026-05-24-stage5-p1-iconic-renders.md

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover

OUT_DIR=renders/stage5_p1_iconic/pushcube
mkdir -p "$OUT_DIR"

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/render_stage5_p1_iconic.py \
    --seeds 100,110,120,130,143 \
    --policies latent,oracle_factor_revision,babysteps_selective,same_intent_retry \
    --pack-dir models/stage5/p1_vision/PushCube-v1 \
    --features-dir datasets/stage5/varied_intent/PushCube-v1/features \
    --out-dir "$OUT_DIR" \
    --fps 20

echo "--- output ---"
ls -lh "$OUT_DIR" | head -25
