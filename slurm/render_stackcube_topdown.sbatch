#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --job-name=render-stack-topdown
#SBATCH --output=slurm/logs/render-stack-topdown-%j.out
#SBATCH --error=slurm/logs/render-stack-topdown-%j.err
# Stage-5 viewpoint test: re-render StackCube varied-intent demos TOP-DOWN.
# Same physics/labels as datasets/stage4/varied_intent/StackCube-v1 (40 seeds),
# only the render camera changes -> frames where world ±x/±y de-entangle.
set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python scripts/stage5_render_demo_frames.py \
    --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl \
    --out-dir datasets/stage5/varied_intent_topdown/StackCube-v1/frames \
    --topdown
echo "=== RENDER DONE === ($(date))"
