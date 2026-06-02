#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH --job-name=stage5-vlm-motion-read
#SBATCH --output=slurm/logs/stage5-vlm-motion-read-%j.out
#SBATCH --error=slurm/logs/stage5-vlm-motion-read-%j.err
# Stage-5 option-3 probe v2: VLM reads object_motion via BEFORE/AFTER frames.
# PushCube scored (clean image->world map); StackCube cross-tab. ~120 VLM
# calls + one model load. Reuses P1 demo cache + Stage-4 labels.
set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/wang4433/hf_cache}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python scripts/stage5_vlm_motion_read.py \
    --task PushCube-v1 --task StackCube-v1 \
    --out-dir reports/stage5/p3demo_vlm_motion_read
echo "=== JOB DONE === ($(date))"
