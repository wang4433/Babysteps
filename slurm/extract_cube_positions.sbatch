#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --job-name=s5-cube-positions
#SBATCH --output=slurm/logs/extract-cube-positions-%j.out
#SBATCH --error=slurm/logs/extract-cube-positions-%j.err
# Stage-5 — extract cubeA-start + cubeB positions for the EXACT StackCube
# varied_intent cut (seeds 0-39), so the relation oracle ceiling can be
# reported on the same seeds that produced the DINOv2 0.42 (not the cross-cut
# p2_vlm_wrist seeds 100-149). Tiny job: 40 native resets, no rollout.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover

JSONL=datasets/stage4/varied_intent/StackCube-v1/samples.jsonl
POSDIR=datasets/stage5/relation_oracle/StackCube-v1

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/stage5_extract_cube_positions.py \
    --jsonl "$JSONL" \
    --out-dir "$POSDIR"

# Exact-cut oracle ceiling on the same seeds as DINOv2 0.42 (CPU; runs here too).
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/stage5_relation_oracle_probe.py \
    --source dir --positions-dir "$POSDIR" --jsonl "$JSONL" \
    --out-dir reports/stage5/relation_oracle_exactcut/

echo "DONE: exact-cut relation oracle ceiling → reports/stage5/relation_oracle_exactcut/"
