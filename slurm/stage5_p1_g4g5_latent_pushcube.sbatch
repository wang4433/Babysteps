#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --job-name=stage5-p1-g4g5-latent
#SBATCH --output=slurm/logs/stage5-p1-g4g5-latent-%j.out
#SBATCH --error=slurm/logs/stage5-p1-g4g5-latent-%j.err
# Stage-5 — FULLY-LATENT-INPUT recovery gates (G4/G5) on PushCube.
#
# The existing P1 G4/G5 (reports/stage5/p1_vision_g4_g5) used the latent
# REVISION but a scripted/demo-derived attempt-1 intent. This run adds
# --latent-initial (Sever A): the attempt-1 intent is decoded from vision
# (DINOv2 -> IntentHead -> nearest-centroid) for ALL policies, so the whole
# loop is latent-input, not just the revision. No VLM — procedural policies +
# the learned latent_revision_factory on real PushCube sim, seeds 100-149.
#
# Gates: G4 = Δpp(latent vs same_intent_retry) >= 10 ; G5 = Δpp(latent vs
# oracle) >= -5. Expectation: ~identical to the scripted-attempt-1 P1 run,
# since the vision-decoded initial reproduces the scripted/JSON intent ~49/50
# and attempt-1 fails by construction either way (the recovery is what the
# gates measure). Confirms the latent input costs nothing on recovery.
#
# Writes reports/stage5/p1_vision_g4_g5_latent/PushCube-v1/. ~10 min wall.

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/wang4433/hf_cache}"

python scripts/stage5_p1_run_eval.py \
    --task PushCube-v1 \
    --pack-dir models/stage5/p1_vision/PushCube-v1 \
    --features-dir datasets/stage5/varied_intent/PushCube-v1/features \
    --out-dir reports/stage5/p1_vision_g4_g5_latent/PushCube-v1 \
    --eval-seeds 100-149 \
    --latent-initial

echo
echo "=== JOB DONE === ($(date))"
echo "report -> reports/stage5/p1_vision_g4_g5_latent/PushCube-v1/report.md"
