#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --job-name=s5-poke-recgate
#SBATCH --output=slurm/logs/poke-recgate-%j.out
#SBATCH --error=slurm/logs/poke-recgate-%j.err
# Stage-5 RECOVERY GATE — does swapping the 8B VLM for the distilled head restore
# recovery on held-out PokeCube? Runs the 8-condition fair-recovery main table
# PLUS the two @distilled_attr rows (--distilled-head): the SAME paired
# conditions with the VLM diagnosis replaced by the VLM-free DistilledAttributor
# and the value policy held fixed, so the row isolates ATTRIBUTION quality.
# Expected: shared_revision_policy@distilled_attr ~= @oracle_attr (0.967) >>
# @vlm (0.008), and distilled decision latency << 3.05 s VLM full replan. The
# deployed PokeCube failures are clean contact_region mis-groundings, so the
# distilled head attributes contact_region (verified sim-free 6/6).
#
# A tiny real-VLM smoke (--target-n 2) runs FIRST (set -e stops on failure).
set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/wang4433/hf_cache}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

SCORER=models/stage5/shared_policy/pooled_gi_none.pt
PACK=models/stage5/p1_vision_4way/PushCube-v1
RHEAD=$PACK/revise_head_residual.pt
DHEAD=models/stage5/attribution_head/head.pt

echo "=== SMOKE (real VLM + distilled, target-n 2) $(date) ==="
python scripts/stage5_pokecube_maintable_eval.py \
  --scorer "$SCORER" --pack-dir "$PACK" --residual-head "$RHEAD" \
  --distilled-head "$DHEAD" \
  --seeds 0-299 --directions +x,+y,-y --max-approach-dist 0.785 --target-n 2 \
  --n-boot 2000 \
  --out reports/stage5/pokecube_recovery_gate/smoke_results.json

echo "=== FULL RUN (real VLM + distilled, target-n 20) $(date) ==="
python scripts/stage5_pokecube_maintable_eval.py \
  --scorer "$SCORER" --pack-dir "$PACK" --residual-head "$RHEAD" \
  --distilled-head "$DHEAD" \
  --seeds 0-299 --directions +x,+y,-y --max-approach-dist 0.785 --target-n 20 \
  --n-boot 10000 \
  --out reports/stage5/pokecube_recovery_gate/results.json

echo "=== JOB DONE === ($(date))"
echo "primary -> reports/stage5/pokecube_recovery_gate/results.json"
