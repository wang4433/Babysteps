#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=stage4-m2a-a3-stackcube
#SBATCH --output=slurm/logs/stage4-m2a-a3-stackcube-%j.out
#SBATCH --error=slurm/logs/stage4-m2a-a3-stackcube-%j.err
# Stage-4 M2a A3 — StackCube G4/G5 eval on real ManiSkill env.
#
# Prerequisite: StackCube pack must be trained with counterfactual
# synthesis enabled (default in scripts/stage4_m2a_train_pack.py since
# the counterfactual feature was added 2026-05-23). Without it the
# StackCube pack has 0 certable revisions and the ReviseHead is at
# random init (see reports/stage4/m2a_a2/notes.md §"Data limitation").
#
# Eval seeds: 200-249 (50 held-out from the StackCube training cut,
# which uses seeds in range 0-84).

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover

if [ ! -f models/stage4/m2a/StackCube-v1/intent_head.pt ]; then
    echo "FATAL: models/stage4/m2a/StackCube-v1/intent_head.pt missing." >&2
    echo "       Run: python scripts/stage4_m2a_train_pack.py \\" >&2
    echo "                --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl \\" >&2
    echo "                --out-dir models/stage4/m2a/StackCube-v1/" >&2
    exit 1
fi

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/stage4_m2a_run_eval.py \
    --task StackCube-v1 \
    --pack-dir models/stage4/m2a/StackCube-v1 \
    --out-dir reports/stage4/m2a_a3_stackcube \
    --eval-seeds 200-249 \
    --policies latent,babysteps_selective,same_intent_retry,oracle_factor_revision

echo
echo "=== JOB DONE === ($(date))"
echo "results: reports/stage4/m2a_a3_stackcube/a3_results.json"
