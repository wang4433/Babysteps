#!/bin/bash
# Submit all 4 Stage-0 render jobs to slurm.
# Each job requests 1 A100 GPU, ~20-30 min.
# Output MP4s land in /scratch/gilbreth/wang4433/render_<task>/videos_maniskill/.
# Logs land in slurm/logs/.
#
# Usage:  bash slurm/submit_all.sh
# Or submit individually:  sbatch slurm/render_<task>.sbatch

set -euo pipefail

cd "$(dirname "$0")/.."

# Pre-download TurnFaucet's asset on the login node so the GPU job doesn't
# block on it. Skips silently if already present. Remove this line if
# you've already downloaded it.
echo "--- pre-download TurnFaucet asset (idempotent) ---"
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
python -m mani_skill.utils.download_asset partnet_mobility_faucet -y || true

echo "--- submitting 4 render jobs ---"
for task in pushcube pickcube stackcube turnfaucet; do
  sbatch_file="slurm/render_${task}.sbatch"
  echo ">> sbatch $sbatch_file"
  sbatch "$sbatch_file"
done

echo "--- job queue ---"
squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.6D %.20R"
