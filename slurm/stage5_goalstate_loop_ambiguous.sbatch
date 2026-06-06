#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --job-name=s5-goalstate-loop-amb
#SBATCH --output=slurm/logs/goalstate-loop-amb-%j.out
#SBATCH --error=slurm/logs/goalstate-loop-amb-%j.err
# Stage-5 — StackCube goal_state loop, AMBIGUOUS (revision-heavy) end of the
# grounding<->revision spectrum.
#
# The retract loop (job 10970071) is the grounding-heavy end: retract+first_last
# grounds goal_state 0.98 -> vision ~= oracle, revision barely fires (1/5 fails
# were misgroundings). To DEMONSTRATE the failure-as-evidence revision, use the
# STANDARD ambiguous demo: NO retract + whole-clip spatial_mean = the deployed
# representation that grounds goal_state only ~0.63. Then ~37% misground ->
# execute the under-specified cube_at_target -> collide/scatter (goal_not_satisfied)
# -> goal_refinement lifts to cubeA_on_cubeB -> retry recovers.
#
# Expected: vision-decode ~0.63; initial ~0.58 (0.63 x stack-exec-ceiling);
# operator/oracle recover the misgroundings -> big on_fail lift; same_intent ~0.
# Contrast with the retract loop = the spectrum (demo render trades grounding vs
# revision-reliance; the loop closes either way).
#
# Same scripts as the retract pipeline, different flags: no --retract, whole-clip
# pooling (--dump-pool "spatial_mean (all)"), suffix dinov2_wc. Seed-disjoint:
# pack-train 0-149, eval 200-249.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/scratch/gilbreth/wang4433/torch_hub_cache

ENC=dinov2_vitb14
SUF=dinov2_wc
POOL="spatial_mean (all)"
TRAIN_SEEDS=0-149
EVAL_SEEDS=200-249
TRAIN_FEATS=datasets/stage5/goalstate/StackCube-v1/train_feats_wc
EVAL_FEATS=datasets/stage5/goalstate/StackCube-v1/eval_feats_wc
PACK=models/stage5/goalstate_ambiguous/StackCube-v1
LOOP_OUT=reports/stage5/goalstate_loop_ambiguous/StackCube-v1

echo "=== [1/4] dump TRAIN demo features (both-class, NO retract, whole-clip) ==="
python scripts/stage5_goal_state_probe.py --task StackCube-v1 --mode clip-pool \
  --seeds "$TRAIN_SEEDS" --encoder "$ENC" \
  --dump-features "$TRAIN_FEATS" --dump-pool "$POOL" --feature-suffix "$SUF" \
  --out-dir reports/stage5/goalstate_pack_train_wc

echo "=== [2/4] train AMBIGUOUS goal_state LatentPack ==="
python scripts/stage5_train_goalstate_pack.py --features-dir "$TRAIN_FEATS" \
  --feature-suffix "$SUF" --out-dir "$PACK" --cv

echo "=== [3/4] dump HELD-OUT EVAL demo features (NO retract, whole-clip) ==="
python scripts/stage5_goal_state_probe.py --task StackCube-v1 --mode clip-pool \
  --seeds "$EVAL_SEEDS" --encoder "$ENC" \
  --dump-features "$EVAL_FEATS" --dump-pool "$POOL" --feature-suffix "$SUF" \
  --out-dir reports/stage5/goalstate_pack_eval_wc

echo "=== [4/4] run the AMBIGUOUS StackCube goal_state loop (held-out) ==="
python scripts/stage5_goalstate_loop_eval.py --pack-dir "$PACK" \
  --demo-features-dir "$EVAL_FEATS" --feature-suffix "$SUF" \
  --eval-seeds "$EVAL_SEEDS" --out-dir "$LOOP_OUT"

echo ""
echo "=== AMBIGUOUS LOOP RESULT ==="
python - <<'PY'
import json
d = json.load(open("reports/stage5/goalstate_loop_ambiguous/StackCube-v1/goalstate_loop_results.json"))
s = d["summary"]
print(f"n={s['n']} n_initial_fail={s['n_initial_fail']}")
print(f"vision-decode acc = {s['vision_decode_acc']:.3f}")
print(f"initial success   = {s['initial_success_rate']:.3f}")
for k,v in s['final_success_rate'].items():
    print(f"final[{k:14s}] all={v:.3f} on_fail={s['final_success_rate_on_initial_fail'][k]:.3f}")
PY
echo "=== DONE ==="
