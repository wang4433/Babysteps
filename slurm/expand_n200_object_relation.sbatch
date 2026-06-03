#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --job-name=s5-objrel-n200
#SBATCH --output=slurm/logs/objrel-n200-%j.out
#SBATCH --error=slurm/logs/objrel-n200-%j.err
# Stage-5 Step-2 — n≈200 expansion of the StackCube object-centric relation
# probe. Tightens the noisy n=40 lift (0.625±0.18) and tests whether a genuine
# cubeA<->cubeB relation emerges beyond the cubeB-single-object signal.
#
# Writes to a SEPARATE namespace (datasets/stage5/object_relation_n200/) so the
# committed 40-seed cut + its 0.42 baseline are untouched.
#
# Standby + --requeue + idempotent steps => preemption-safe: collect re-runs
# only if the jsonl is incomplete; render --skip-existing resumes per-seed;
# extract/probe are fast and just re-run.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

OUTROOT=datasets/stage5/object_relation_n200
ROOT=$OUTROOT/StackCube-v1
JSONL=$ROOT/samples.jsonl
FRAMES=$ROOT/frames
TOKENS=$ROOT/object_tokens.npz
PERCLASS=50
MAXSCAN=2000
NEED=$((PERCLASS * 4))   # 4 cardinal classes

# 1) Collect a balanced 50-per-class StackCube varied cut (GPU sim).
#    Idempotent + partial-write-safe: re-collect unless the jsonl already has
#    all NEED records.
if [ ! -f "$JSONL" ] || [ "$(wc -l < "$JSONL")" -lt "$NEED" ]; then
    echo "[collect] building $NEED-record cut (per_class=$PERCLASS, max_scan=$MAXSCAN)"
    python scripts/stage4_collect_varied.py \
        --task StackCube-v1 --out_dir "$OUTROOT" \
        --per_class "$PERCLASS" --max_scan "$MAXSCAN"
else
    echo "[collect] skip: $JSONL has $(wc -l < "$JSONL") records (>= $NEED)"
fi

# 2) Render demo frames (GPU sim). --skip-existing resumes after preemption.
echo "[render] frames -> $FRAMES (skip-existing)"
python scripts/stage5_render_demo_frames.py \
    --jsonl "$JSONL" --out-dir "$FRAMES" --skip-existing

# 3) Extract object-local DINOv2 patch tokens (GPU DINO; NO Vulkan). Fast.
echo "[extract] object tokens -> $TOKENS"
python scripts/stage5_extract_object_patch_tokens.py \
    --frames-dir "$FRAMES" --jsonl "$JSONL" --out "$TOKENS" --device cuda

# 4) Object-centric relation probe (CPU). radius=1 (best of the n=40 sweep).
echo "[probe] -> reports/stage5/object_relation_probe_n200/"
python scripts/stage5_object_relation_probe.py \
    --tokens "$TOKENS" --out-dir reports/stage5/object_relation_probe_n200/ --radius 1

echo "DONE: n200 object-relation probe -> reports/stage5/object_relation_probe_n200/"
