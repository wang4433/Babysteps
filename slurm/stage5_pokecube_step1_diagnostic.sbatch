#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --job-name=s5-poke-step1
#SBATCH --output=slurm/logs/poke-step1-%j.out
#SBATCH --error=slurm/logs/poke-step1-%j.err
# Stage-5 step 1 — VLM wrong-factor HISTOGRAM on held-out PokeCube.
# Recovers the DISTRIBUTION behind the main table's attribution_accuracy~0.017:
# which factor does the 8B VLM pick from a single third-person failure frame?
# Re-sources each (seed,direction,wrong_face) episode (same _source_episode the
# main table used) and feeds the real frame to the SAME VLMAttributor. Cheap:
# one initial attempt + one VLM constrained call per episode (no retries).
#
# A tiny real-VLM smoke (--target-n 2) runs FIRST (set -e stops on failure).
set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/wang4433/hf_cache}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

echo "=== SMOKE (real VLM, target-n 2) $(date) ==="
python scripts/stage5_pokecube_step1_diagnostic.py \
  --seeds 0-299 --directions +x,+y,-y --max-approach-dist 0.785 --target-n 2 \
  --out reports/stage5/pokecube_step1_diagnostic/smoke_results.json

echo "=== FULL RUN (real VLM, target-n 20) $(date) ==="
python scripts/stage5_pokecube_step1_diagnostic.py \
  --seeds 0-299 --directions +x,+y,-y --max-approach-dist 0.785 --target-n 20 \
  --out reports/stage5/pokecube_step1_diagnostic/results.json

echo "=== JOB DONE === ($(date))"
echo "primary -> reports/stage5/pokecube_step1_diagnostic/results.json"
