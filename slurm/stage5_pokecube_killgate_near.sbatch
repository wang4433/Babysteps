#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --job-name=s5-poke-near
#SBATCH --output=slurm/logs/poke-near-%j.out
#SBATCH --error=slurm/logs/poke-near-%j.err
# Stage-5 step-3 NEAR-SEED kill-gate: confirm PokeCube is a NON-DEGENERATE
# multi-candidate contact_region family.
#
# The full-seed kill-gate (job 10979211) PASSED +x at 0.833 but showed ±y are
# reach-dead for far cubes (gripper can't reach the lateral poke pose past
# base-dist ~0.77). A +x-only family is DEGENERATE for the contact_region choice
# (correct face is always minus_x_face). On NEAR cubes (cube_x <= 0.16, base-dist
# <= ~0.775) +x/+y/-y are ALL reachable, so the correct face varies over
# {minus_x_face, minus_y_face, plus_y_face} -> a genuine 3-way LOTO cell.
#
# This scans seeds 0-149, keeps the first ~15 reachable (cube_x<=0.16) per
# direction, and reports per-direction oracle-face success. PASS (all 3 >= 0.8)
# => PokeCube is a real multi-candidate family; proceed to the frozen LOTO eval.
set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

python scripts/stage5_pokecube_killgate.py \
  --seeds 0-149 --directions +x,+y,-y --gate 0.8 \
  --max-cube-x 0.16 --target-n 15 \
  --out reports/stage5/pokecube_killgate/findings_near.json
