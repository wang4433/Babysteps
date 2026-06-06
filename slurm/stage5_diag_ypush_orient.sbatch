#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --requeue
#SBATCH --job-name=s5-ypush-orient
#SBATCH --output=slurm/logs/ypush-orient-%j.out
#SBATCH --error=slurm/logs/ypush-orient-%j.err
set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
echo "=== ORIENT CONTROL ON (the 4-way fix) ==="
python scripts/_diag_pushcube_ypush.py --orient --seeds 0-9 \
  --out reports/stage5/diag_pushcube_ypush/result_orient.json
echo "=== DONE ==="
