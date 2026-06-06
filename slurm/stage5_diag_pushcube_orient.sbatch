#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --requeue
#SBATCH --job-name=s5-diag-orient
#SBATCH --output=slurm/logs/diag-orient-%j.out
#SBATCH --error=slurm/logs/diag-orient-%j.err
set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python scripts/_diag_pushcube_orient.py --seed 0 --mag 0.5 --steps 15
echo "=== DONE ==="
