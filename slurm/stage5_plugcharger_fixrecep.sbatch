#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --job-name=stage5-plugcharger-fixrecep
#SBATCH --output=slurm/logs/stage5-plugcharger-fixrecep-%j.out
#SBATCH --error=slurm/logs/stage5-plugcharger-fixrecep-%j.err
# Stage-5 — PlugCharger CONTROL: receptacle held at a canonical pose.
#
# The deployed-mode scout found the negative control (receptacle_yaw) PASSING at
# 0.958 — the receptacle-mounted camera makes the BACKGROUND (table+robot)
# counter-rotate with receptacle yaw, a strong global cue, and the relative-yaw
# label correlates with it. This run PINS the receptacle to a canonical pose so
# the camera (and background) is identical every frame; only the charger varies.
# Any residual charger_yaw / charger_xy separability MUST then come from the
# CHARGER, not the background. Decisive disambiguation of the deployed PASS.
#
# Writes reports/stage5/plugcharger_probe_fixrecep/PlugCharger-v1/<factor>_res<res>/.

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/wang4433/hf_cache}"

python scripts/stage5_plugcharger_probe.py \
    --task PlugCharger-v1 \
    --seeds 0-119 \
    --factors charger_yaw,charger_xy \
    --resolutions 224,518 \
    --fix-receptacle \
    --out-dir reports/stage5/plugcharger_probe_fixrecep \
    --save-frames

echo
echo "=== JOB DONE === ($(date))"
for d in reports/stage5/plugcharger_probe_fixrecep/PlugCharger-v1/*/; do
    echo "----- ${d}report.md -----"
    cat "${d}report.md" 2>/dev/null || true
done
