#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --job-name=stage5-topdown-probe
#SBATCH --output=slurm/logs/stage5-topdown-probe-%j.out
#SBATCH --error=slurm/logs/stage5-topdown-probe-%j.err
# Stage-5 viewpoint test: DINOv2 G1 object_motion probe, oblique vs top-down.
set -euo pipefail
cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "### caching DINOv2 (spatial_mean) on TOP-DOWN frames ###"
python scripts/stage5_cache_dinov2.py \
    --frames-dir datasets/stage5/varied_intent_topdown/StackCube-v1/frames \
    --out-dir   datasets/stage5/varied_intent_topdown/StackCube-v1/features \
    --pool spatial_mean

echo "### G1 probe — OBLIQUE baseline (existing spatial_mean features) ###"
python scripts/stage5_p1_g1_cert.py \
    --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl \
    --features-dir datasets/stage5/varied_intent/StackCube-v1/features \
    --out-dir reports/stage5/p3demo_topdown/oblique

echo "### G1 probe — TOP-DOWN ###"
python scripts/stage5_p1_g1_cert.py \
    --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl \
    --features-dir datasets/stage5/varied_intent_topdown/StackCube-v1/features \
    --out-dir reports/stage5/p3demo_topdown/topdown

echo "### object_motion rows ###"
grep -i "object_motion" reports/stage5/p3demo_topdown/oblique/report.md  | head -3
echo "---"
grep -i "object_motion" reports/stage5/p3demo_topdown/topdown/report.md | head -3
echo "=== JOB DONE === ($(date))"
