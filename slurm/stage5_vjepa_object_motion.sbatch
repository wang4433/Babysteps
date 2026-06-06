#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --job-name=s5-vjepa-objmotion
#SBATCH --output=slurm/logs/vjepa-objmotion-%j.out
#SBATCH --error=slurm/logs/vjepa-objmotion-%j.err
# Stage-5 — V-JEPA 2.1 video encoder on StackCube object_motion.
#
# The temporal lever: frame-mean-pooled DINOv2 gets 0.685 @ n=200 on the
# two-object relational motion direction (reports/stage5/p1_vision_g1_n200).
# V-JEPA 2.1 is a CLIP encoder built for temporally-consistent dense features —
# the one untried lever that targets that bottleneck.
#
# Decisive control: --shuffle-frame-order permutes each clip's frames in time.
#   ordered ≫ shuffled-time and ≫ 0.685  -> genuine TEMPORAL lift (2nd hard task)
#   ordered ≈ shuffled-time ≈ 0.685       -> no temporal signal (boundary closed)
#
# Standby + --requeue + idempotent per-seed skip => preemption-safe.
# Frames already cached at n=200 (no re-render). Weights (~5 GB ViT-L) download
# once into TORCH_HOME on scratch; the repo's hub URL is a localhost stub so
# vision_features._load_vjepa pulls the real checkpoint from fbaipublicfiles.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

ROOT=datasets/stage5/object_relation_n200/StackCube-v1
JSONL=$ROOT/samples.jsonl
FRAMES=$ROOT/frames
FEATURES=$ROOT/features
ENC=vjepa2_1_vitl16
OUT=reports/stage5/vjepa_object_motion

echo "=== [0/4] download + load smoke (validates strict state_dict load) ==="
python scripts/stage5_cache_vjepa.py --check --encoder "$ENC" --device cuda

echo "=== [1/4] cache ORDERED V-JEPA features (idempotent) ==="
python scripts/stage5_cache_vjepa.py \
  --frames-dir "$FRAMES" --out-dir "$FEATURES" \
  --encoder "$ENC" --feature-suffix vjepa21 --device cuda

echo "=== [2/4] cache SHUFFLED-TIME control features (idempotent) ==="
python scripts/stage5_cache_vjepa.py \
  --frames-dir "$FRAMES" --out-dir "$FEATURES" \
  --encoder "$ENC" --feature-suffix vjepa21 --shuffle-frame-order --device cuda

echo "=== [3/4] probe ORDERED (apples-to-apples vs DINOv2 0.685) ==="
python scripts/stage5_p1_g1_cert.py \
  --jsonl "$JSONL" --features-dir "$FEATURES" \
  --feature-suffix vjepa21 --pool token_mean \
  --encoder-label "V-JEPA 2.1 ViT-L/16 (clip token-mean)" --no-narrative \
  --out-dir "$OUT/StackCube-v1" --seed 0 || true

echo "=== [4/4] probe SHUFFLED-TIME control ==="
python scripts/stage5_p1_g1_cert.py \
  --jsonl "$JSONL" --features-dir "$FEATURES" \
  --feature-suffix vjepa21_shuf --pool token_mean \
  --encoder-label "V-JEPA 2.1 ViT-L/16 (SHUFFLED-TIME control)" --no-narrative \
  --out-dir "$OUT/StackCube-v1_shuftime" --seed 0 || true

echo "=== object_motion rows ==="
grep -h object_motion "$OUT"/StackCube-v1*/report.md || true
echo "=== DONE ==="
