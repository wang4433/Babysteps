#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=s5-obj-tokens
#SBATCH --output=slurm/logs/extract-object-tokens-%j.out
#SBATCH --error=slurm/logs/extract-object-tokens-%j.err
# Stage-5 Step-2 — extract object-LOCAL DINOv2 patch tokens for the StackCube
# varied_intent cut (the exact 40 seeds that produced the 0.42 baseline), then
# run the object-centric relation probe. GPU is used only for the DINOv2
# forward pass over cached frames — NO ManiSkill/Vulkan.
# QoS standby: lower priority but typically starts sooner on idle A100 nodes;
# this job is tiny (~5-10 min) so preemption risk is low.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover

FRAMES=datasets/stage5/varied_intent/StackCube-v1/frames
JSONL=datasets/stage4/varied_intent/StackCube-v1/samples.jsonl
TOKENS=datasets/stage5/object_relation/StackCube-v1/object_tokens.npz

# 1) GPU: cached frames -> object-local DINOv2 patch tokens + blob centroids.
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/stage5_extract_object_patch_tokens.py \
    --frames-dir "$FRAMES" \
    --jsonl "$JSONL" \
    --out "$TOKENS" \
    --device cuda

# 2) CPU: object-centric visual relation probe (runs on the GPU node too).
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/stage5_object_relation_probe.py \
    --tokens "$TOKENS" \
    --out-dir reports/stage5/object_relation_probe/ \
    --radius 1

echo "DONE: object-relation probe -> reports/stage5/object_relation_probe/"
