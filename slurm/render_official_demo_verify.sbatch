#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=115G
#SBATCH --time=00:25:00
#SBATCH --job-name=render-official-verify
#SBATCH --output=slurm/logs/render-official-verify-%j.out
#SBATCH --error=slurm/logs/render-official-verify-%j.err
# P2-verify: run the real driver end-to-end with --demo-source official for
# all three cube tasks. The 1_demo clip becomes the official ManiSkill oracle
# (state-replay, third-person); attempt/retry stay on the babysteps env.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover

RENDERS=/home/wang4433/scratch/babysteps/renders

for TASK in PushCube-v1 PickCube-v1 StackCube-v1; do
    echo "=================== $TASK ==================="
    LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
    python scripts/render_stage0_maniskill.py \
        --task "$TASK" \
        --demo-source official \
        --official-source state_replay \
        --n_episodes 1 \
        --seed_start 0 \
        --fps 20 \
        --out_dir "$RENDERS/official_demo_verify/${TASK}"
done

echo "--- artifacts ---"
find "$RENDERS/official_demo_verify" -name '*.mp4' -printf '%s  %p\n' 2>/dev/null | sort
