#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --job-name=stage5-p2-latent-push
#SBATCH --output=slurm/logs/stage5-p2-latent-push-%j.out
#SBATCH --error=slurm/logs/stage5-p2-latent-push-%j.err
# Stage-5 — LATENT-INPUT P2 eval on PushCube (the latent-intent pivot proof).
#
# Same C1-vs-C2 selectivity harness as stage5_p2_vlm_fair.sbatch, but with
# --latent: the initial intent is decoded from vision (DINOv2 -> IntentHead
# -> nearest-centroid) and C1's repair is the learned slot-local ReviseHead,
# instead of the hand-authored JSON intent + discrete operator. The JSON
# factors are then used ONLY for supervision (centroid codebook) + eval
# (oracle reference). This is the reviewer-defense: "JSON factors are used
# for evaluation, not as privileged method input."
#
# Pre-flight (GPU-free, already run): the vision-decoded initial intent
# reproduces the JSON intent 49/50 on these exact seeds
# (reports/stage5/latent_decode_check/PushCube-v1), so the cached failure
# frames + oracle labels stay valid; n_latent_mismatch is logged per run.
#
# Expectation: latent C1 ~= discrete C1 (attr 1.0, pres 1.0, succ ~0.98).
# PushCube preservation is at ceiling for BOTH C1 and C2 — this run proves
# the latent pipeline recovers success end-to-end; the C1>>C2 selectivity
# GAP is a PickCube/StackCube story (separate cut).
#
# Writes reports/stage5/p2_vlm_latent/PushCube-v1/. Reuses cached failure
# frames from datasets/stage5/p2_vlm/PushCube-v1/. ~15 min wall.

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/wang4433/hf_cache}"

python scripts/stage5_p2_vlm_eval.py \
    --task PushCube-v1 \
    --episodes datasets/stage5/p2_vlm/PushCube-v1/episodes.jsonl \
    --out-dir reports/stage5/p2_vlm_latent/PushCube-v1 \
    --latent \
    --pack-dir models/stage5/p1_vision/PushCube-v1 \
    --features-dir datasets/stage5/varied_intent/PushCube-v1/features \
    --train-jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl

echo
echo "=== JOB DONE === ($(date))"
echo "report -> reports/stage5/p2_vlm_latent/PushCube-v1/report.md"
echo "latent audit: grep n_latent_mismatch reports/stage5/p2_vlm_latent/PushCube-v1/c1_results.json"
