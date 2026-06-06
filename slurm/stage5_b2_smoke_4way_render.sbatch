#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --requeue
#SBATCH --job-name=s5-b2-smoke-render
#SBATCH --output=slurm/logs/b2-smoke-render-%j.out
#SBATCH --error=slurm/logs/b2-smoke-render-%j.err
# Stage-5 B.2 SMOKE: render 8 PushCube 4-way demo clips (2 per direction) and
# self-check the realized push direction. The render-path gripper-yaw port
# (orient_control) must push +-y cleanly (ang_err < 30deg = OK) or the 4-way
# centroids would be corrupted. ABORT the full render if many lines say BAD.
set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python scripts/stage5_render_demo_frames.py \
  --four-way-range 600-607 \
  --out-dir datasets/stage5/four_way/PushCube-v1/frames_smoke
echo "=== DONE (inspect ang_err / OK|BAD per line above) ==="
