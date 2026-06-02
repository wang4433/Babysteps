#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH --job-name=stage5-vlm-intent-read
#SBATCH --output=slurm/logs/stage5-vlm-intent-read-%j.out
#SBATCH --error=slurm/logs/stage5-vlm-intent-read-%j.err
# Stage-5 option-3 de-risking probe: can InternVL3.5-8B READ object_motion
# off a third-person DEMO strip? Measures VLM-vs-oracle agreement on the
# varied-intent cut (PushCube 20 seeds, StackCube 40 seeds, balanced 4-way).
# No training. Reuses the P1 demo-frame cache + Stage-4 labels + the P2
# InternVL3.5 loader. ~60 VLM calls + one model load ≈ ~10 min.
#
# Pre-registered kill criterion: demo-frame agreement < 0.90 ⇒ below the G1
# gate by construction ⇒ relational-gap-reduction ablation, not a P1 pass.

set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/scratch/gilbreth/wang4433/hf_cache}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/stage5_vlm_intent_read.py \
    --task PushCube-v1 --task StackCube-v1 \
    --out-dir reports/stage5/p3demo_vlm_intent_read

echo
echo "=== JOB DONE === ($(date))"
