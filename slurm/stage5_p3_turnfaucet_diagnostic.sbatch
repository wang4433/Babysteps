#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=115G
#SBATCH --time=01:00:00
#SBATCH --job-name=p3-tf-diag
#SBATCH --output=slurm/logs/p3-tf-diag-%j.out
#SBATCH --error=slurm/logs/p3-tf-diag-%j.err
# Stage-5 Phase-3 diagnostic — TurnFaucet poke_turn failure-mode breakdown
# on seeds 100-149. See redesign_failure_paradigm.md §"Phase 3" decision gate.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/stage5_p3_turnfaucet_diagnostic.py \
    --seed_start 100 \
    --seed_end   149 \
    --out_dir    reports/stage5/turnfaucet_diagnostic

echo "--- summary ---"
cat reports/stage5/turnfaucet_diagnostic/summary.md
