#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=115G
#SBATCH --time=01:00:00
#SBATCH --job-name=p3-tf-diag-v1geom
#SBATCH --output=slurm/logs/p3-tf-diag-v1geom-%j.out
#SBATCH --error=slurm/logs/p3-tf-diag-v1geom-%j.err
# Stage-5 Phase-3 diagnostic RE-RUN after the v1 poke-geometry port
# (OBB handle centre + true circular tangent threaded through scene.extra).
# Same seeds 100-149 as the 4% baseline in
# reports/stage5/turnfaucet_diagnostic/{summary.md,per_seed.jsonl}; writes the
# new results to the v1geom/ subdir so the baseline stays pristine for the
# lift comparison. Decision gate (redesign_failure_paradigm.md §"Phase 3"):
# success >= 30% -> proceed; < 30% after 3 days -> TurnFaucet to appendix.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/stage5_p3_turnfaucet_diagnostic.py \
    --seed_start 100 \
    --seed_end   149 \
    --out_dir    reports/stage5/turnfaucet_diagnostic/v1geom

echo "--- v1geom summary ---"
cat reports/stage5/turnfaucet_diagnostic/v1geom/summary.md
echo
echo "--- baseline (perp-axis) for comparison ---"
cat reports/stage5/turnfaucet_diagnostic/summary.md
