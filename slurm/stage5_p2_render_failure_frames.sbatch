#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --job-name=stage5-p2-failure-frames
#SBATCH --output=slurm/logs/stage5-p2-failure-frames-%j.out
#SBATCH --error=slurm/logs/stage5-p2-failure-frames-%j.err
# Stage-5 P2 — render one first-person attempt frame per held-out seed,
# for each of the three eval tasks. Seeds 100-149 (50 per task → 150 total).

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

for TASK in PushCube-v1 PickCube-v1 StackCube-v1; do
    echo "=== $TASK ==="
    python scripts/stage5_p2_render_failure_frames.py \
        --task "$TASK" --seeds 100-149 \
        --out-dir "datasets/stage5/p2_vlm/$TASK"
done

echo
echo "=== JOB DONE === ($(date))"
