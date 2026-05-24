#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=stage4-m25-attribution-stackcube
#SBATCH --output=slurm/logs/stage4-m25-attribution-stackcube-%j.out
#SBATCH --error=slurm/logs/stage4-m25-attribution-stackcube-%j.err
# Stage-4 M2.5 — StackCube G5-strict eval with the learned attribution head.
#
# Same eval shape as M2a A3 (seeds 200-249, 4 policies). The only
# difference is the LatentPack: models/stage4/m25/packs/StackCube-v1/
# bundles the joint AttributionHead trained on PushCube + StackCube
# Stage-0 baseline failures (208 + 188 = 396 supervised pairs, 2
# unique label classes, held-out kfold acc = 1.000, shuffled-label
# control = 0.530).
#
# Expected outcome (per design spec §8):
#   - latent matches or beats babysteps_selective on the 5 ambiguous
#     seeds {201, 209, 220, 225, 233} that previously misattributed
#     direction_error/no_motion → approach_direction.
#   - G5 strict gap improves from −10 pp to within ±5 pp vs oracle.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover

if [ ! -f models/stage4/m25/packs/StackCube-v1/attribution_head.pt ]; then
    echo "FATAL: M2.5 StackCube pack is missing attribution_head.pt." >&2
    echo "       Run:" >&2
    echo "         python scripts/stage4_m25_train_attribution.py \\" >&2
    echo "             --task StackCube-v1 --task PushCube-v1 \\" >&2
    echo "             --out-dir models/stage4/m25/joint/ \\" >&2
    echo "             --report-dir reports/stage4/m25_attribution/joint/" >&2
    echo "         python scripts/stage4_m2a_train_pack.py \\" >&2
    echo "             --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl \\" >&2
    echo "             --out-dir models/stage4/m25/packs/StackCube-v1/ \\" >&2
    echo "             --attribution-head-pt models/stage4/m25/joint/attribution_head.pt" >&2
    exit 1
fi

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/stage4_m2a_run_eval.py \
    --task StackCube-v1 \
    --pack-dir models/stage4/m25/packs/StackCube-v1 \
    --out-dir reports/stage4/m25_attribution_stackcube \
    --eval-seeds 200-249 \
    --policies latent,babysteps_selective,same_intent_retry,oracle_factor_revision

echo
echo "=== JOB DONE === ($(date))"
echo "results: reports/stage4/m25_attribution_stackcube/a3_results.json"
