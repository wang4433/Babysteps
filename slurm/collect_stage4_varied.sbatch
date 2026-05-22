#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=115G
#SBATCH --time=01:00:00
#SBATCH --job-name=stage4-collect-varied
#SBATCH --output=slurm/logs/collect-stage4-varied-%j.out
#SBATCH --error=slurm/logs/collect-stage4-varied-%j.err
# Stage-4 varied-intent collection: PushCube (injection) + StackCube (rejection).

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover

# Fail-fast injection spike (Task-5 de-risk): abort the job here if the cube-pose
# injection or PushCube success check is broken, rather than writing 40 bad
# episodes. set -e propagates the assert failure.
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python - <<'PY'
from babysteps.envs.pushcube_runner import PushCubeEnvRunner
from babysteps.envs.pushcube_adapter import PushCubeAdapter
r, a = PushCubeEnvRunner(), PushCubeAdapter()
ok = True
for m in ("translate_+x", "translate_-x", "translate_+y", "translate_-y"):
    r.set_injection(m)
    s = r.reset(7)
    i = a.oracle_correct_intent(s)
    res = r.run(i, s)
    print(f"spike {m}: derived={i.object_motion} success={res.success}", flush=True)
    ok = ok and (i.object_motion == m) and bool(res.success)
r.close()
assert ok, "INJECTION SPIKE FAILED (see above) — fix env.unwrapped.obj handle / use goal_region fallback before collecting"
print("injection spike PASSED — proceeding to collection", flush=True)
PY

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/stage4_collect_varied.py --task PushCube-v1

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/stage4_collect_varied.py --task StackCube-v1
