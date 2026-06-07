#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --job-name=s5-poke-maintable
#SBATCH --output=slurm/logs/poke-maintable-%j.out
#SBATCH --error=slurm/logs/poke-maintable-%j.err
# Stage-5 — PokeCube 6-condition FAIR-RECOVERY main table (held-out family).
# User's locked step 1: all six unified conditions on held-out PokeCube under a
# fair protocol (same first failure / controller / seed / retry budget), with the
# FROZEN shared scorer (pooled Push+Stack, g_i masked) as the proposed method and
# the FAIR (enumerated-values, non-straw-man) free-replan VLM baseline.
#
# Conditions: same_intent_retry, random_factor_local_edit, vlm_free_replan,
# vlm_diagnosis_local_edit (VLM factor + PushCube-family residual editor, reused
# = a transfer baseline), shared_revision_policy (VLM factor + ONE shared
# scorer), oracle_single_slot. Emits Table 2 (fair recovery + clustered CI),
# Table 4 (latency split / VLM tokens+calls / edit cardinality / preservation /
# harmful). Reach filter = the validated 3-way kill-gate (--max-approach-dist
# 0.785); 3 reachable faces; ~20 seeds x 3 dir x 2 wrong = ~120 episodes.
#
# A tiny real-VLM smoke (--target-n 2) runs FIRST to validate the GPU
# runner+frames+InternVL pipeline before the full run (set -e stops on failure).
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

echo "=== SMOKE (real VLM, target-n 2) $(date) ==="
python scripts/stage5_pokecube_maintable_eval.py \
  --scorer "$SCORER" --pack-dir "$PACK" --residual-head "$RHEAD" \
  --seeds 0-299 --directions +x,+y,-y --max-approach-dist 0.785 --target-n 2 \
  --n-boot 2000 \
  --out reports/stage5/pokecube_maintable/smoke_results.json

echo "=== FULL RUN (real VLM, target-n 20) $(date) ==="
python scripts/stage5_pokecube_maintable_eval.py \
  --scorer "$SCORER" --pack-dir "$PACK" --residual-head "$RHEAD" \
  --seeds 0-299 --directions +x,+y,-y --max-approach-dist 0.785 --target-n 20 \
  --n-boot 10000 \
  --out reports/stage5/pokecube_maintable/results.json

echo "=== JOB DONE === ($(date))"
echo "primary -> reports/stage5/pokecube_maintable/results.json"
