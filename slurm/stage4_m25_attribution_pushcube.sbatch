#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=stage4-m25-attribution-pushcube
#SBATCH --output=slurm/logs/stage4-m25-attribution-pushcube-%j.out
#SBATCH --error=slurm/logs/stage4-m25-attribution-pushcube-%j.err
# Stage-4 M2.5 — PushCube G4/G5 no-regression check with learned attribution.
#
# Verifies that swapping the rule-based attribute_failure for the joint
# AttributionHead does not regress PushCube performance. The rule
# already gets PushCube right (approach_blocked → approach_direction
# 44/44), and the joint head matches it 44/44 on the training data
# (sim-free). Real ManiSkill eval is the end-to-end no-regression
# sanity check. Seeds 100-149 (50 held-out).

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover

if [ ! -f models/stage4/m25/packs/PushCube-v1/attribution_head.pt ]; then
    echo "FATAL: M2.5 PushCube pack is missing attribution_head.pt." >&2
    exit 1
fi

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/stage4_m2a_run_eval.py \
    --task PushCube-v1 \
    --pack-dir models/stage4/m25/packs/PushCube-v1 \
    --out-dir reports/stage4/m25_attribution_pushcube \
    --eval-seeds 100-149 \
    --policies latent,babysteps_selective,same_intent_retry,oracle_factor_revision

echo
echo "=== JOB DONE === ($(date))"
echo "results: reports/stage4/m25_attribution_pushcube/a3_results.json"
