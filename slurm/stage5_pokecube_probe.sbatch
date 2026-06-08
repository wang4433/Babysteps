#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --job-name=s5-pokecube-probe
#SBATCH --output=slurm/logs/pokecube-probe-%j.out
#SBATCH --error=slurm/logs/pokecube-probe-%j.err
# Stage-5 step-3 GPU PROBE (feasibility-design gate, before writing the runner).
#
# PART 1 — Is PokeCube-v1 a real ManiSkill 3 env? Mechanics: peg start state
# (grasped vs must-grasp), goal radius, goal FIXED across seeds (=> goal-move
# injection needed, like PushCube), obs keys, action space. If the id is wrong,
# the probe lists the registered Poke/Push/Peg envs so we use the real name.
#
# PART 2 — PushCube 4-face feasibility under orient_control (the user chose
# "enrich to 4 faces"): per-face oracle-push success for +x/-x/+y/-y. If
# min-face success < 0.8, the shared contact_region rule the LOTO tests is only
# x-axis (near-trivial) and the controller must be fixed before the GPU LOTO.
#
# This decides: (a) the real PokeCube env id + whether injection is needed for
# the grasp+poke runner, (b) the poke-feasibility kill-gate design, (c) whether
# 4-face enrichment is already feasible. NO runner is written until this passes.
set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

python scripts/stage5_pokecube_probe.py \
  --seeds 0-9 \
  --out reports/stage5/pokecube_probe/findings.json
