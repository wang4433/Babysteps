#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --requeue
#SBATCH --job-name=s5-poke-killgate
#SBATCH --output=slurm/logs/poke-killgate-%j.out
#SBATCH --error=slurm/logs/poke-killgate-%j.err
# Stage-5 step-3 GPU KILL-GATE for the PokeCube contact_region family.
#
# Confirms the grasp+poke PokeCubeEnvRunner can SUCCEED with the ORACLE contact
# face — the make-or-break feasibility check before PokeCube can be the held-out
# leave-one-task-family-out cell. Reports per-direction success + grasp/contact
# rates + cube displacement so the geometry constants can be calibrated.
#
#   +x (n=24)  = the gate floor (zero-yaw native poke). If this < 0.8 the family
#                is dead; tune pokecube_runner.py geometry + re-run.
#   -x,+y,-y   = the 4-face enrichment (peg yaw reorientation); maps which extra
#                candidates are executable.
set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

# +x at full n (the gate floor), then the REACHABLE enrichment faces.
# -x is dropped on purpose: it is geometrically reach-dead for EVERY seed (to
# poke -x the gripper must sit on the cube's far +x side, 0.89-1.09 m from the
# Panda base at x=-0.615, beyond ~0.85 m reach) — confirmed analytically in the
# pre-GPU review, so running it would only burn 350 steps x n on a guaranteed
# fail and misreport multi_candidate_confirmed. +y/-y are reach-PARTIAL (peg
# extends sideways, no x-reach help) so they map which y-seeds are executable.
python scripts/stage5_pokecube_killgate.py \
  --seeds 0-23 --directions +x --gate 0.8 \
  --out reports/stage5/pokecube_killgate/findings_plusx.json

python scripts/stage5_pokecube_killgate.py \
  --seeds 0-11 --directions +x,+y,-y --gate 0.8 \
  --out reports/stage5/pokecube_killgate/findings_yface.json
