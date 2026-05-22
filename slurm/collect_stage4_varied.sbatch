#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=115G
#SBATCH --time=01:00:00
#SBATCH --job-name=stage4-collect-varied
#SBATCH --output=slurm/logs/collect-stage4-varied-%j.out
#SBATCH --error=slurm/logs/collect-stage4-varied-%j.err
# Stage-4 varied-intent collection: PushCube (injection) + StackCube (rejection).

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/stage4_collect_varied.py --task PushCube-v1

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/stage4_collect_varied.py --task StackCube-v1
