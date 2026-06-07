#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --job-name=s5-poke-3way
#SBATCH --output=slurm/logs/poke-3way-%j.out
#SBATCH --error=slurm/logs/poke-3way-%j.err
# Stage-5 step-3 3-WAY kill-gate: confirm PokeCube as a 3-candidate
# contact_region family {minus_x_face(+x), plus_y_face(-y), minus_y_face(+y)}.
#
# The near-seed run (cube_x<=0.16) gave +x=1.00, -y=0.87 but +y=0.73 — +y fails
# on cubes far in -y because the +y lateral approach point (cube_y-0.155) leaves
# reach. Fix = principled ALL-DIRECTIONS reachability gate: keep a seed only if
# its worst prepoke/poke waypoint distance from the Panda base (over +x,+y,-y) is
# <= 0.785 (sim-free calibration cleanly separates the observed pass population
# <=0.785 from the fail population >=0.794). This selects the SAME reachable seed
# set across directions, so the 3 faces are compared on identical seeds.
# PASS (all 3 >= 0.8) => 3-candidate cell; proceed to the frozen LOTO eval.
set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

python scripts/stage5_pokecube_killgate.py \
  --seeds 0-299 --directions +x,+y,-y --gate 0.8 \
  --max-approach-dist 0.785 --target-n 15 \
  --out reports/stage5/pokecube_killgate/findings_3way.json
