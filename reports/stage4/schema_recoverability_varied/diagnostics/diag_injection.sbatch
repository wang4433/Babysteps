#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --job-name=diag-injection
#SBATCH --output=reports/stage4/schema_recoverability_varied/diagnostics/diag_injection.out
#SBATCH --error=reports/stage4/schema_recoverability_varied/diagnostics/diag_injection.err
set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python reports/stage4/schema_recoverability_varied/diagnostics/diag_injection.py
