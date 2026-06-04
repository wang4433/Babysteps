#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --job-name=stage5-dinov3-swap
#SBATCH --output=slurm/logs/stage5-dinov3-swap-%j.out
#SBATCH --error=slurm/logs/stage5-dinov3-swap-%j.err
# Stage-5 — ENCODER-SWAP ablation: DINOv2 -> DINOv3 on the cells DINOv2 FAILED.
#
# The Stage-5 limitation is not BabySteps; it is the frozen visual encoder's
# ability to recover observation-grounded latent factors. We test whether a
# stronger image encoder (DINOv3, Meta's dense-feature successor) crosses the
# SAME gate (confound-controlled IntentHead-CV >= 0.90) on cells where frozen
# DINOv2 ViT-B/14 fell short. Fairness: identical labels + identical probe
# protocol; identical ImageNet normalization and full-frame square FOV; each
# encoder uses its native resize kernel (DINOv2 bilinear, DINOv3 bicubic+
# antialias) so neither is handicapped. ViT-L/16 ("slightly bigger") is the
# headline; pass vitb16 too (matched size) to separate "v3 training" from
# "more params".
#
# Weights come from timm's UNGATED re-host (vit_*_patch16_dinov3.lvd1689m) —
# no HF token, no license acceptance needed. They are pre-cached in HF_HOME.
#
# Usage:  sbatch slurm/stage5_dinov3_encoder_swap.sbatch dinov3_vitl16
#         sbatch slurm/stage5_dinov3_encoder_swap.sbatch dinov3_vitb16
#
# Cells (self-contained probes that already pipe --encoder -> extract_vision_features):
#   1. StackCube goal_state, --mode config   (POSITIVE CONTROL: DINOv2=0.992;
#      a DINOv3 result far from ~0.99 would mean a broken integration, not a finding)
#   2. StackCube goal_state, --mode clip-pool (THE TEST: DINOv2 capped 0.82 on real clips)
#   3. PlugCharger fix-receptacle             (THE TEST: DINOv2 charger_yaw 0.83-0.86 FAIL,
#      background frozen so any separability is the charger itself)
#   4. PlugCharger deployed                   (keeps the receptacle_yaw negative control
#      comparable across encoders)
# object_motion (temporal) is deferred to the V-JEPA round; PickCube is invisible (skip).

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/wang4433/hf_cache}"

ENC="${1:-dinov3_vitl16}"
SHORT="${ENC#dinov3_}"   # vitl16 / vitb16
echo "=== encoder-swap: $ENC (short=$SHORT) === ($(date))"

# Fail fast (before any rendering) if the timm weights can't be loaded.
python - "$ENC" <<'PY'
import sys
from babysteps.stage4.vision_features import _DINOV3_ALIASES, _load_dinov3
enc = sys.argv[1]
name = _DINOV3_ALIASES.get(enc, enc)
print(f"loading timm {name} ...", flush=True)
m = _load_dinov3(enc, "cuda")
print(f"OK: embed_dim={m.embed_dim} num_prefix_tokens={m.num_prefix_tokens}", flush=True)
PY

echo "----- [1/5] goal_state config @224 (positive control) -----"
python scripts/stage5_goal_state_probe.py \
    --task StackCube-v1 --seeds 0-59 --encoder "$ENC" --mode config --resolution 224 \
    --out-dir "reports/stage5/goal_state_probe_${SHORT}_config"

echo "----- [2/5] goal_state clip-pool @224 (deployed-res test vs DINOv2 0.82) -----"
python scripts/stage5_goal_state_probe.py \
    --task StackCube-v1 --seeds 0-59 --encoder "$ENC" --mode clip-pool --resolution 224 \
    --out-dir "reports/stage5/goal_state_probe_${SHORT}_clip224" --save-frames

echo "----- [3/5] goal_state clip-pool @512 (DINOv3 hi-res lever) -----"
python scripts/stage5_goal_state_probe.py \
    --task StackCube-v1 --seeds 0-59 --encoder "$ENC" --mode clip-pool --resolution 512 \
    --out-dir "reports/stage5/goal_state_probe_${SHORT}_clip512"

echo "----- [4/5] plugcharger fix-receptacle @{224,512} (the test) -----"
python scripts/stage5_plugcharger_probe.py \
    --task PlugCharger-v1 --seeds 0-119 --encoder "$ENC" \
    --factors charger_yaw,charger_xy --resolutions 224,512 --fix-receptacle \
    --out-dir "reports/stage5/plugcharger_probe_${SHORT}_fixrecep"

echo "----- [5/5] plugcharger deployed @{224,512} (neg-control comparison) -----"
python scripts/stage5_plugcharger_probe.py \
    --task PlugCharger-v1 --seeds 0-119 --encoder "$ENC" \
    --factors charger_yaw,charger_xy,receptacle_yaw --resolutions 224,512 \
    --out-dir "reports/stage5/plugcharger_probe_${SHORT}_deployed"

echo
echo "=== JOB DONE === ($(date)) encoder=$ENC"
echo "### goal_state config @224 (control):"
cat "reports/stage5/goal_state_probe_${SHORT}_config/StackCube-v1/report.md" 2>/dev/null | sed -n '1,40p' || true
echo "### goal_state clip-pool @224 (test vs DINOv2 0.82):"
cat "reports/stage5/goal_state_probe_${SHORT}_clip224/StackCube-v1/report.md" 2>/dev/null | sed -n '1,60p' || true
echo "### goal_state clip-pool @512 (hi-res lever):"
cat "reports/stage5/goal_state_probe_${SHORT}_clip512/StackCube-v1/report.md" 2>/dev/null | sed -n '1,60p' || true
echo "### plugcharger fix-receptacle (test):"
for d in reports/stage5/plugcharger_probe_${SHORT}_fixrecep/PlugCharger-v1/*/; do
    echo "--- ${d}report.md ---"; cat "${d}report.md" 2>/dev/null | sed -n '1,20p' || true
done
echo "### plugcharger deployed (neg-control):"
for d in reports/stage5/plugcharger_probe_${SHORT}_deployed/PlugCharger-v1/*/; do
    echo "--- ${d}report.md ---"; cat "${d}report.md" 2>/dev/null | sed -n '1,20p' || true
done
