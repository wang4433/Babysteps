# RUNBOOK

Copy-paste operational commands for BABYSTEPS Stage-0. For *why* (boundary,
schema, claims) read `goal.md` and `CLAUDE.md`. For the SLURM-side details and
recorded gate results, see `slurm/CLAUDE.md`.

Environment (Gilbreth):

```bash
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate handover          # pre-existing env with ManiSkill 3
cd /scratch/gilbreth/wang4433/babysteps
```

## 1. Sim-free unit tests (login node, no GPU)

```bash
python -m pytest tests/ -q       # 302 tests, ~1.3s
```

## 2. Data collection

Fake env (no Vulkan/GPU — for the loop logic, CI-style):

```bash
python scripts/stage0_collect.py \
  --out_dir datasets/stage0_pushcube_blocked \
  --n_episodes 5 --seed_start 0 --fake-env
```

Real ManiSkill collection (needs a GPU+Vulkan compute node). Example
CrossViewPush data cut (~24 seeds) then summarize:

```bash
python scripts/stage0_collect.py --task CrossViewPush-v1 \
  --n_episodes 24 --seed_start 0 --out_dir /scratch/gilbreth/wang4433/data_crossview

python scripts/stage0_summarize.py \
  --samples /scratch/gilbreth/wang4433/data_crossview/samples.jsonl \
  --out_dir /scratch/gilbreth/wang4433/data_crossview
```

`--task` accepts `{PushCube-v1, PickCube-v1, StackCube-v1, TurnFaucet-v1,
CrossViewPush-v1}`. `stage0_summarize.py` derives the task from the input JSONL
(no flag). Gate (`report.json`): `delta_pp>=10`, `passed_acceptance=true`,
`frozen_factor_preservation_rate=1.0`, `unnecessary_factor_change_rate=0.0`.

## 3. Three-phase MP4 render (GPU)

All tasks use the same script with a different `--task`. Output defaults to
`renders/<task>/videos_maniskill/` under the repo (every project render lives
under `renders/` — pass `--out_dir` only to override). Each task produces
`n_episodes × 3` MP4s named
`<task_prefix>_seed_NNNN__{1_demo,2_attempt_blocked,3_retry}.mp4`.

```bash
srun --account=rpaleja --partition=a100-40gb --gres=gpu:1 --mem=115G --time=00:20:00 bash -lc '
  cd /scratch/gilbreth/wang4433/babysteps &&
  source /apps/external/conda/2025.09/etc/profile.d/conda.sh &&
  conda activate handover &&
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
  python scripts/render_stage0_maniskill.py \
    --task <TASK>-v1 \
    --n_episodes 2 \
    --seed_start 0 &&
  ls -lh renders/<task>/videos_maniskill
'
```

Per-task substitutions (`<TASK>` / `<task>` dir / `--n_episodes`):

| Task | renders/ dir | episodes | prefix |
| --- | --- | --- | --- |
| PushCube-v1 | `renders/pushcube` | 2 | `pushcube_blocked_approach` |
| PickCube-v1 | `renders/pickcube` | 2 | `pickcube_grasp_slip` |
| StackCube-v1 | `renders/stackcube` | 2 | `stackcube_underspec_goal` |
| TurnFaucet-v1 | `renders/turnfaucet` | 2 | `turnfaucet_*` |
| CrossViewPush-v1 | `renders/crossview` | 3 | `crossview_grounding` |

Equivalent batch jobs live in `slurm/*.sbatch` (and `slurm/submit_all.sh`).

## Task-specific notes

- **TurnFaucet** requires the asset once:
  `python -m mani_skill.utils.download_asset partnet_mobility_faucet`.
  The renderer auto-selects `sim_backend="gpu"` (CPU-sim IK drifts the robot).
  Phase 1 demo uses a privileged qpos teleport; Phase 2 grasp attempt fails
  physically (handle exceeds gripper opening); Phase 3 retries with a
  closed-gripper lateral poke-turn with auto-sign detection (see
  `scripts/_diag_tf_poke5.py`). Gate: ≥1/5 retry MP4s reach `info["success"]`.
- **CrossViewPush** runs on the real `PushCube-v1` gym env via
  `adapter.gym_env_id`. Observer yaw (90/180/270) lives in the grounding math;
  all phases render from the world camera. Phase 2 executes a real failing
  (wrong-way) push. Gate: ≥2 seeds reach `info["success"]` on retry and
  `2_attempt_blocked` visibly pushes the wrong way.

## Run the procedural baseline sweep (M3)

Sim-free smoke (login node):

```bash
python scripts/run_baselines.py --tasks PushCube-v1 --methods all \
  --n_episodes 2 --seed_start 0 --out_dir /tmp/baselines --fake-env
```

Full GPU sweep (7 methods × 3 tasks × 24 seeds):

```bash
sbatch slurm/run_baselines.sbatch   # writes datasets/stage0_baselines/comparison_table.md
```
