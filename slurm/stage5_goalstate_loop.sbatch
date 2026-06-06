#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --job-name=s5-goalstate-loop
#SBATCH --output=slurm/logs/goalstate-loop-%j.out
#SBATCH --error=slurm/logs/goalstate-loop-%j.err
# Stage-5 — StackCube goal_state FULL-VISION LOOP (the 2nd positive latent task).
#
# Grounding confirmed (job 10969827): retract render + first_last pooling ->
# goal_state IntentHead-CV 0.920 +/-0.019 (n=300). This builds the end-to-end loop:
#   1. dump TRAIN demo features (both classes, retract, first_last)  [seeds 0-149]
#   2. train the goal_state LatentPack                               [2 classes]
#   3. dump held-out EVAL demo features (disjoint)                   [seeds 200-249]
#   4. run the loop: vision-decode goal_state -> exec -> fail -> revise -> retry
#
# Expected: vision-decode acc ~0.92 on held-out (== the CV); initial_success ~=
# decode_acc x stack-exec-ceiling (~0.82, the runner's intrinsic pick-and-place
# reliability, NOT 1.0); operator(goal_refinement)/oracle recover the mis-grounded
# failures -> final ~= the exec ceiling; same_intent stays at initial. The metric
# is recovery vs no-recovery. (Whole-clip grounding is ~0.77 -> the demo render
# trades grounding vs revision-reliance; this run is the grounding-heavy end.)
#
# SEED-DISJOINT: pack-train 0-149, eval 200-249 -> honest held-out generalization.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

ENC=dinov2_vitb14
SUF=dinov2_fl
TRAIN_SEEDS=0-149
EVAL_SEEDS=200-249
TRAIN_FEATS=datasets/stage5/goalstate/StackCube-v1/train_feats
EVAL_FEATS=datasets/stage5/goalstate/StackCube-v1/eval_feats
PACK=models/stage5/goalstate/StackCube-v1
LOOP_OUT=reports/stage5/goalstate_loop/StackCube-v1

echo "=== [1/4] dump TRAIN demo features (both-class retract first_last, seeds $TRAIN_SEEDS) ==="
python scripts/stage5_goal_state_probe.py --task StackCube-v1 --mode clip-pool \
  --seeds "$TRAIN_SEEDS" --encoder "$ENC" --retract \
  --dump-features "$TRAIN_FEATS" --dump-pool first_last --feature-suffix "$SUF" \
  --out-dir reports/stage5/goalstate_pack_train

echo "=== [2/4] train goal_state LatentPack ==="
python scripts/stage5_train_goalstate_pack.py --features-dir "$TRAIN_FEATS" \
  --feature-suffix "$SUF" --out-dir "$PACK" --cv

echo "=== [3/4] dump HELD-OUT EVAL demo features (seeds $EVAL_SEEDS, disjoint) ==="
python scripts/stage5_goal_state_probe.py --task StackCube-v1 --mode clip-pool \
  --seeds "$EVAL_SEEDS" --encoder "$ENC" --retract \
  --dump-features "$EVAL_FEATS" --dump-pool first_last --feature-suffix "$SUF" \
  --out-dir reports/stage5/goalstate_pack_eval

echo "=== [4/4] run the StackCube goal_state loop (held-out) ==="
python scripts/stage5_goalstate_loop_eval.py --pack-dir "$PACK" \
  --demo-features-dir "$EVAL_FEATS" --feature-suffix "$SUF" \
  --eval-seeds "$EVAL_SEEDS" --out-dir "$LOOP_OUT"

echo ""
echo "=== LOOP RESULT ==="
python - <<'PY'
import json
d = json.load(open("reports/stage5/goalstate_loop/StackCube-v1/goalstate_loop_results.json"))
s = d["summary"]
print(f"n={s['n']} n_initial_fail={s['n_initial_fail']}")
print(f"vision-decode acc = {s['vision_decode_acc']:.3f}")
print(f"initial success   = {s['initial_success_rate']:.3f}")
for k,v in s['final_success_rate'].items():
    print(f"final[{k:14s}] all={v:.3f} on_fail={s['final_success_rate_on_initial_fail'][k]:.3f}")
PY
echo "=== DONE ==="
