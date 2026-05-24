#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=stage4-m2a-a3-pushcube
#SBATCH --output=slurm/logs/stage4-m2a-a3-pushcube-%j.out
#SBATCH --error=slurm/logs/stage4-m2a-a3-pushcube-%j.err
# Stage-4 M2a A3 — PushCube G4/G5 eval on real ManiSkill env.
#
# What this does:
#   - (Pre-flight) sanity-load the trained LatentPack at
#     models/stage4/m2a/PushCube-v1; abort if missing.
#   - Run scripts/stage4_m2a_run_eval.py with 4 policies over seeds
#     100-149 (50 held-out seeds — disjoint from the training cut's
#     seeds 0-19).
#   - Output: reports/stage4/m2a_a3_pushcube/a3_results.json with
#     per-policy success rates + Δpp summary.
#
# What unblocks next:
#   - delta_pp_vs_same_intent_retry ≥ 10  → G4 PASS on real env.
#   - delta_pp_vs_oracle ≥ -5             → G5 PASS on real env.
#   Both passing means M2a closes end-to-end on PushCube with a
#   credible non-fake baseline contrast; the report becomes the M2a
#   headline alongside m2a_a2/a2_results and m2a_intent_head_g1/.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover

# Pre-flight: pack must exist.
if [ ! -f models/stage4/m2a/PushCube-v1/intent_head.pt ]; then
    echo "FATAL: models/stage4/m2a/PushCube-v1/intent_head.pt missing." >&2
    echo "       Run: python scripts/stage4_m2a_train_pack.py \\" >&2
    echo "                --jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl \\" >&2
    echo "                --out-dir models/stage4/m2a/PushCube-v1/" >&2
    exit 1
fi

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/stage4_m2a_run_eval.py \
    --task PushCube-v1 \
    --pack-dir models/stage4/m2a/PushCube-v1 \
    --out-dir reports/stage4/m2a_a3_pushcube \
    --eval-seeds 100-149 \
    --policies latent,babysteps_selective,same_intent_retry,oracle_factor_revision

echo
echo "=== JOB DONE === ($(date))"
echo "results: reports/stage4/m2a_a3_pushcube/a3_results.json"
