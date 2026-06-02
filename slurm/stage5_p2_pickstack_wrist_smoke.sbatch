#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=p2-pickstack-wrist-smoke
#SBATCH --output=slurm/logs/p2-pickstack-wrist-smoke-%j.out
#SBATCH --error=slurm/logs/p2-pickstack-wrist-smoke-%j.err
# Smoke: confirm PickCube/StackCube env construction with panda_wristcam,
# wrist-frame capture, and per-step wrist_rgb .npz size (Pick/Stack step the
# env, unlike PushCube). 3 seeds each, render-only, into /tmp. No VLM.

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

for TASK in PickCube-v1 StackCube-v1; do
    OUT="/tmp/p2_wrist_smoke/$TASK"
    echo "================= $TASK smoke (seeds 100-102) ================="
    python scripts/stage5_p2_render_failure_frames.py \
        --task "$TASK" --seeds 100-102 --out-dir "$OUT"
    echo "wrist PNGs:  $(find "$OUT/frames" -name '*_attempt_wrist.png' 2>/dev/null | wc -l)"
    echo "rollout npz: $(find "$OUT/rollouts" -name '*.npz' 2>/dev/null | wc -l)"
    echo "--- per-episode wrist_rgb shape + .npz size ---"
    python - "$OUT" <<'PY'
import sys, json, glob, os
import numpy as np
out = sys.argv[1]
for line in open(os.path.join(out, "episodes.jsonl")):
    r = json.loads(line)
    print(f"seed {r['seed']}: predicate={r['failure_predicate']!r} "
          f"initial_success={r['initial_success']} oracle={r['oracle_wrong_factor']!r} "
          f"wrist_frame_path={'set' if r.get('wrist_frame_path') else 'NONE'}")
for npz in sorted(glob.glob(os.path.join(out, "rollouts", "*.npz"))):
    d = np.load(npz)
    shp = d["wrist_rgb"].shape if "wrist_rgb" in d else "(no wrist_rgb)"
    mb = os.path.getsize(npz) / 1e6
    print(f"  {os.path.basename(npz)}: wrist_rgb={shp}  size={mb:.1f} MB")
PY
done
echo "================= SMOKE DONE ================="
