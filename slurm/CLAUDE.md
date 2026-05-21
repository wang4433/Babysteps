# slurm/ — batch jobs, logs, and the GPU run record

Gilbreth SLURM glue for the GPU work (render + real data collection), plus the
recorded acceptance-gate results. The login node has no Vulkan, so all real-sim
work runs here. For the full copy-paste command catalog see `RUNBOOK.md`.

## Files

| File | Job |
| --- | --- |
| `render_pushcube.sbatch` | Batch render of PushCube three-phase MP4s. |
| `render_pickcube.sbatch` | Batch render of PickCube MP4s. |
| `render_stackcube.sbatch` | Batch render of StackCube MP4s. |
| `render_turnfaucet.sbatch` | Batch render of TurnFaucet MP4s (needs faucet asset). |
| `crossview_gpu.sbatch` | CrossViewPush GPU gate: real PushCube physics data cut + summarize. |
| `submit_all.sh` | Submit the render batch jobs together. |
| `logs/` | `*.out` / `*.err` per job (named `<task>-<jobid>`). |

## Canonical interactive run (single GPU)

```bash
srun --account=rpaleja --partition=a100-40gb --gres=gpu:1 --mem=115G --time=00:20:00 bash -lc '
  cd /scratch/gilbreth/wang4433/babysteps &&
  source /apps/external/conda/2025.09/etc/profile.d/conda.sh &&
  conda activate handover &&
  OUT_DIR=/scratch/gilbreth/wang4433/render_<task> &&
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
  python scripts/render_stage0_maniskill.py \
    --task <TASK>-v1 --out_dir "$OUT_DIR" --n_episodes 2 --seed_start 0 &&
  ls -lh "$OUT_DIR/videos_maniskill"
'
```

Per-task substitutions and the data-collection / summarize commands are in
`RUNBOOK.md`. Key environment requirement: prepend `$CONDA_PREFIX/lib` to
`LD_LIBRARY_PATH` so the NVIDIA Vulkan ICD resolves.

## Recorded gate results

### CrossViewPush-v1 — Sub-project E (job 10737370, 2026-05-20)

Real PushCube-v1 physics, 24 seeds. **`passed_acceptance=true`.**

| Metric | Value |
| --- | --- |
| `delta_pp` | 95.8 (threshold ≥10) |
| initial / retry success | 0/24 → 23/24 |
| `frozen_factor_preservation_rate` | 1.0 |
| `unnecessary_factor_change_rate` | 0.0 |
| attribution accuracy | 23/24 |

The one miss is honest real-sim noise: seed 0019's push missed contact →
`predicate=contact_failure` (not a grounding error) → correctly delegated to
`contact_region`, **not** `direction_grounding`. All 23 grounding-style
failures were attributed and fixed. Render: 9 MP4s; `2_attempt_blocked` clips
are larger (the wrong-way push is visible).

## Rules

- Job logs are artifacts — fine to prune old ones; the gate *numbers* live in
  this file and in each sub-project's spec, not only in the logs.
- A new GPU gate result gets a row/section here when it lands.
