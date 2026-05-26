#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --job-name=stage5-p2-frames-5task
#SBATCH --output=slurm/logs/stage5-p2-frames-5task-%j.out
#SBATCH --error=slurm/logs/stage5-p2-frames-5task-%j.err
# Stage-5 P2 — render first-person attempt frames for the two new tasks
# being added to the main table (TurnFaucet + CrossViewPush). Seeds 100-149
# (50 per task → 100 frames total). The three existing tasks already have
# frames cached at datasets/stage5/p2_vlm/{PushCube,PickCube,StackCube}-v1/.

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

for TASK in TurnFaucet-v1 CrossViewPush-v1; do
    echo "=== $TASK ==="
    python scripts/stage5_p2_render_failure_frames.py \
        --task "$TASK" --seeds 100-149 \
        --out-dir "datasets/stage5/p2_vlm/$TASK"
done

echo
echo "=== JOB DONE === ($(date))"
