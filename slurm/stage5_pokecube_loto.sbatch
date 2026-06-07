#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --requeue
#SBATCH --job-name=s5-poke-loto
#SBATCH --output=slurm/logs/poke-loto-%j.out
#SBATCH --error=slurm/logs/poke-loto-%j.err
# Stage-5 LOTO eval: FROZEN shared scorer transfer to held-out PokeCube.
# Primary = pooled Push+Stack g_i-masked; ablation = Push-only g_i-masked.
# (centroid-g_i is NOT evaluable on a held-out task — no PokeCube centroids — a
# finding that itself supports g_i=None for LOTO.)
set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

python scripts/stage5_pokecube_loto_eval.py \
  --scorer models/stage5/shared_policy/pooled_gi_none.pt \
  --seeds 0-299 --directions +x,+y,-y --max-approach-dist 0.785 --target-n 20 \
  --out reports/stage5/pokecube_loto/results_pooled_gi_none.json

python scripts/stage5_pokecube_loto_eval.py \
  --scorer models/stage5/shared_policy/pushonly_gi_none.pt \
  --seeds 0-299 --directions +x,+y,-y --max-approach-dist 0.785 --target-n 20 \
  --out reports/stage5/pokecube_loto/results_pushonly_gi_none.json
