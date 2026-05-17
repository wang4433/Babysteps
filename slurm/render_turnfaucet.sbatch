#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=115G
#SBATCH --time=00:30:00
#SBATCH --job-name=stage0-render-turnfaucet
#SBATCH --output=slurm/logs/render-turnfaucet-%j.out
#SBATCH --error=slurm/logs/render-turnfaucet-%j.err
# Stage-0 Sub-project D — TurnFaucet constraint_violation.
# Closes D's acceptance gate item 4 (GPU MP4 spot-check).
#
# PREREQUISITE: partnet_mobility_faucet asset must be downloaded once
# (login node is fine, no GPU needed):
#   python -m mani_skill.utils.download_asset partnet_mobility_faucet
# This sbatch script invokes that command as a no-op if the asset is
# already present, otherwise downloads it (adds ~few minutes to first
# run; the GPU stays idle while download finishes).

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover

echo "--- ensure partnet_mobility_faucet asset is present ---"
python -m mani_skill.utils.download_asset partnet_mobility_faucet -y || true

OUT_DIR=/scratch/gilbreth/wang4433/render_turnfaucet
mkdir -p "$OUT_DIR"

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/render_stage0_maniskill.py \
    --task TurnFaucet-v1 \
    --out_dir "$OUT_DIR" \
    --n_episodes 2 \
    --seed_start 0

echo "--- output ---"
ls -lh "$OUT_DIR/videos_maniskill"
