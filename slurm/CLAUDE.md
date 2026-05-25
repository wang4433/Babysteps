# slurm/ — batch jobs, logs, and the GPU run record

For eazy jobs use rpaleja A30 node, with heavy jobs, use A100-40gb.

Gilbreth SLURM glue for the GPU work (render + real data collection), plus the
recorded acceptance-gate results. The login node has no Vulkan, so all real-sim
work runs here. For the full copy-paste command catalog see `RUNBOOK.md`.

## Files


| File                       | Job                                                                 |
| -------------------------- | ------------------------------------------------------------------- |
| `render_pushcube.sbatch`   | Batch render of PushCube three-phase MP4s.                          |
| `render_pickcube.sbatch`   | Batch render of PickCube MP4s.                                      |
| `render_stackcube.sbatch`  | Batch render of StackCube MP4s.                                     |
| `render_turnfaucet.sbatch` | Batch render of TurnFaucet MP4s (needs faucet asset).               |
| `crossview_gpu.sbatch`     | CrossViewPush GPU gate: real PushCube physics data cut + summarize. |
| `submit_all.sh`            | Submit the render batch jobs together.                              |
| `logs/`                    | `*.out` / `*.err` per job (named `<task>-<jobid>`).                 |


## Canonical interactive run (single GPU)

```bash
srun --account=rpaleja --partition=a100-40gb --gres=gpu:1 --mem=115G --time=00:20:00 bash -lc '
  cd /scratch/gilbreth/wang4433/babysteps &&
  source /apps/external/conda/2025.09/etc/profile.d/conda.sh &&
  conda activate handover &&
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
  python scripts/render_stage0_maniskill.py \
    --task <TASK>-v1 --n_episodes 2 --seed_start 0 &&
  ls -lh renders/<task>/videos_maniskill   # default out_dir is renders/<task>
'
```

Per-task substitutions and the data-collection / summarize commands are in
`RUNBOOK.md`. Key environment requirement: prepend `$CONDA_PREFIX/lib` to
`LD_LIBRARY_PATH` so the NVIDIA Vulkan ICD resolves.

## Recorded gate results

### CrossViewPush-v1 — Sub-project E (job 10737370, 2026-05-20)

Real PushCube-v1 physics, 24 seeds. `**passed_acceptance=true`.**


| Metric                            | Value                |
| --------------------------------- | -------------------- |
| `delta_pp`                        | 95.8 (threshold ≥10) |
| initial / retry success           | 0/24 → 23/24         |
| `frozen_factor_preservation_rate` | 1.0                  |
| `unnecessary_factor_change_rate`  | 0.0                  |
| attribution accuracy              | 23/24                |


The one miss is honest real-sim noise: seed 0019's push missed contact →
`predicate=contact_failure` (not a grounding error) → correctly delegated to
`contact_region`, **not** `direction_grounding`. All 23 grounding-style
failures were attributed and fixed. Render: 9 MP4s; `2_attempt_blocked` clips
are larger (the wrong-way push is visible).

### Stage-5 P2 — VLM attribution (jobs 10806525 + 10806596 + 10806755, 2026-05-25)

InternVL3.5-8B (BF16) on A100-40GB. 50 held-out failure episodes per task
(seeds 100-149), three tasks. C1 = VLM-constrained-diagnosis (one factor
name) + slot-local discrete revision. C2 = VLM-free-form replan (verbatim
6-field JSON). Frames captured with the new `render_mode="rgb_array"` flag
on the data-collection runners.

| task           | C1 attr | rule-table | C1 pres | C2 pres | Δpres pp | C1 succ | C2 succ | Δsucc pp |
| -------------- | ------- | ---------- | ------- | ------- | -------- | ------- | ------- | -------- |
| PushCube-v1    | 1.000   | 1.000      | 1.000   | 1.000   | +0.0     | 0.980   | 0.980   | +0.0     |
| PickCube-v1    | 0.960   | 1.000      | 1.000   | 1.000   | +0.0     | 0.900   | 0.000   | +90.0    |
| StackCube-v1   | 0.000   | 0.860      | 1.000   | 0.380   | +62.0    | 0.000   | 0.000   | +0.0     |

Gates (per task: `attr ≥ rule-table · pres ≥ C2 · succ within 5pp of C2`):
- **PushCube-v1**: PASS · PASS · PASS
- **PickCube-v1**: FAIL · PASS · PASS
- **StackCube-v1**: FAIL · PASS · PASS

Headline: **C1 ≥ C2 on success everywhere and on preservation everywhere**;
strictly better on PickCube success (+90pp — C2 returns the unchanged intent
verbatim 50/50) and StackCube preservation (+62pp — C2 makes unnecessary
edits to 62% of the non-implicated factors).

VLM attribution accuracy under-runs the rule table on PickCube (2/50 wrong;
both episodes the VLM picked `embodiment_mapping` instead of `contact_region`
on a grasp_slip — plausible from the frame) and catastrophically on
StackCube (0/50; the VLM picks `object_motion` 43× when the oracle is
`goal_state`, mistaking "cube didn't end at goal" for a motion error). The
goal_state failure pattern is the genuine open problem.

C2 max_new_tokens had to be split from C1 (256 vs 32): the first run used
shared 64 and truncated every C2 JSON output → 100% parse_failed; the
re-run job 10806755 with the split limit fixed it.

Re-running just C2 (after a single-condition rerun): use
`scripts/stage5_p2_regenerate_reports.py` afterward to rebuild the merged
per-task `report.md` from the two on-disk JSONs.

## Rules

- Job logs are artifacts — fine to prune old ones; the gate *numbers* live in
this file and in each sub-project's spec, not only in the logs.
- A new GPU gate result gets a row/section here when it lands.

