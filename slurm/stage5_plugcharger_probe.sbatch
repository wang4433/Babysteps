#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --job-name=stage5-plugcharger-probe
#SBATCH --output=slurm/logs/stage5-plugcharger-probe-%j.out
#SBATCH --error=slurm/logs/stage5-plugcharger-probe-%j.err
# Stage-5 — PlugCharger-v1 latent-groundability SCOUT (no BabySteps loop).
#
# Is there a SECOND clean latent end-to-end task beyond PushCube? Probe whether
# the deployed frozen encoder (DINOv2 ViT-B/14, spatial_mean) can separate
# PlugCharger's randomized factors from a third-person reset frame:
#   charger_yaw    (-> object_motion)     PRIMARY CANDIDATE  (asymmetric base
#                  visibly rotates +-60 deg; initial-state, so no clip discount)
#   charger_xy     (-> approach_direction) SECONDARY  (optically large but
#                  predicted to FAIL under spatial_mean — pooling dilution)
#   receptacle_yaw (-> constraint_region) NEGATIVE CONTROL  (camera mounted on
#                  the receptacle -> yaw cancelled in-frame -> MUST read ~chance;
#                  a PASS would mean the probe cheats on the fixed background)
#
# All three cells share the SAME rendered frames (they differ only in label).
# Renders 120 reset frames once, encodes at 224 (deployed) and 518 (scale
# ceiling), runs the G1-protocol probe ladder per cell. ~15-20 min wall.
# Writes reports/stage5/plugcharger_probe/PlugCharger-v1/<factor>_res<res>/.

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/wang4433/hf_cache}"

python scripts/stage5_plugcharger_probe.py \
    --task PlugCharger-v1 \
    --seeds 0-119 \
    --factors charger_yaw,charger_xy,receptacle_yaw \
    --resolutions 224,518 \
    --out-dir reports/stage5/plugcharger_probe \
    --save-frames

echo
echo "=== JOB DONE === ($(date))"
for d in reports/stage5/plugcharger_probe/PlugCharger-v1/*/; do
    echo "----- ${d}report.md -----"
    cat "${d}report.md" 2>/dev/null || true
done
