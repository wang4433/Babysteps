#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --job-name=s5-goalstate-retract-n300
#SBATCH --output=slurm/logs/goalstate-retract-n300-%j.out
#SBATCH --error=slurm/logs/goalstate-retract-n300-%j.err
# Stage-5 goal_state — RETRACT confirmation at larger n.
#
# n=120 (job 10969792) cleared the gate borderline: retract first_last
# IntentHead-CV 0.908 PASS, but direct-LR 0.90-0.93 >> CV 0.83-0.91 with high
# fold std (+/-0.07-0.18) = underpowered IntentHead. This re-runs the retract
# clip-pool at seeds 0-149 (n=300) to firm the CV toward the direct-LR (~0.93)
# and tighten the variance. Same encoder/camera/protocol; only n changes.
#
# PASS-solid if first_last / final-state CV >= 0.90 with a tight std at n=300.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

ENC=dinov2_vitb14
SEEDS=0-149
OUT=reports/stage5/goal_state_retract_n300

echo "=== retract clip-pool, default camera, n=300 (seeds $SEEDS) ==="
python scripts/stage5_goal_state_probe.py \
  --task StackCube-v1 --mode clip-pool --seeds "$SEEDS" \
  --encoder "$ENC" --retract --save-frames --out-dir "$OUT"

echo ""
echo "=== RETRACT n=300 result ==="
grep -hE "Deployed|best final-state|POOLING|PARTIAL|REPRESENTATION|^\| \`" \
  "$OUT/StackCube-v1/retract/report.md" 2>/dev/null || echo "(no report)"
echo "=== DONE ==="
