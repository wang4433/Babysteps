#!/bin/bash
#SBATCH --account=rpaleja
#SBATCH --partition=a30
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=115G
#SBATCH --time=00:20:00
#SBATCH --job-name=pushcube-natural-render
#SBATCH --output=slurm/logs/pushcube-natural-%j.out
#SBATCH --error=slurm/logs/pushcube-natural-%j.err
# Natural-failure paper-figure render for PushCube — no obstacle, phase 2
# executes a single-factor misgrounded contact_region (opposite face) so the
# cube physically goes the wrong way. See redesign_failure_paradigm.md
# §"Five Tasks" and babysteps/render/pushcube.py render_natural_failure_episode.
# Output lands in renders/pushcube/videos_paper_figure/<timestamp>/.

set -euo pipefail

cd /scratch/gilbreth/wang4433/babysteps
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
python scripts/render_stage0_maniskill.py \
    --task PushCube-v1 \
    --mode natural \
    --n_episodes 5 \
    --seed_start 0

echo "--- most recent natural-mode run ---"
ls -lh "$(ls -dt renders/pushcube/videos_paper_figure/*/ | head -1)"
