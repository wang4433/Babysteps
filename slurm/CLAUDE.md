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

### Stage-5 P2 — VLM attribution v2, task-aware prompts (job 10827248, 2026-05-26)

InternVL3.5-8B (BF16) on A100-40GB. Same held-out cut as the first P2 run
(seeds 100-149, 50 episodes per task). Only change: per-task context now
injected into both C1 and C2 prompts — task name, one-line success
description, and the valid `goal_state` tokens for that task (no explicit
"pick goal_state" hint; the VLM still has to spot the symbolic mismatch).
Commit `4ddca7f`. Wall time: 10:41.

| task           | C1 attr | rule-table | C1 pres | C2 pres | Δpres pp | C1 succ | C2 succ | Δsucc pp |
| -------------- | ------- | ---------- | ------- | ------- | -------- | ------- | ------- | -------- |
| PushCube-v1    | 1.000   | 1.000      | 1.000   | 1.000   | +0.0     | 0.980   | 0.960   | +2.0     |
| PickCube-v1    | 1.000   | 1.000      | 1.000   | 1.000   | +0.0     | 0.920   | 0.000   | +92.0    |
| StackCube-v1   | 0.860   | 0.860      | 1.000   | 0.500   | +50.0    | 0.700   | 0.220   | +48.0    |

Gates (per task: `attr ≥ rule-table · pres ≥ C2 · succ within 5pp of C2`):
- **PushCube-v1**: PASS · PASS · PASS
- **PickCube-v1**: PASS · PASS · PASS
- **StackCube-v1**: PASS · PASS · PASS — **all three gates now pass on all three tasks.**

Headline: StackCube C1 attribution jumped from **0/50 → 43/50** (= rule-table
exactly); StackCube C1 success jumped from **0.000 → 0.700** (= the
`babysteps_selective` M3 baseline). PickCube nudged up to 1.000 / 0.920
(was 0.960 / 0.900). PushCube unchanged at ceiling on attribution; succ
0.98 → 0.96 (within run-to-run noise, +2pp vs C2). The 7 StackCube residual
misses all still land on `object_motion` — the genuinely visually-ambiguous
cases where the cube ended up in *some* incorrect xy and motion looks wrong
in the frame.

C2 also gets the task context for fairness. On PushCube/PickCube it's
unchanged; on StackCube C2 success climbed from 0.000 → 0.220 (the model
now knows `cubeA_on_cubeB` exists), but parse failures jumped to 76%
because the slightly longer prompt nudges JSON output to drift off the
schema. C1 wins on both the strict and parse-conditional preservation
definitions.

### Stage-5 P2 — VLM attribution 5-task expansion (job 10842893, 2026-05-26)

InternVL3.5-8B (BF16) on A100-40GB. Adds **TurnFaucet-v1** and
**CrossViewPush-v1** to the P2 main table (now 5 tasks, matching the
locked paper claim in `docs/milestone1_locked_claim.md` §6). Same
held-out cut (seeds 100-149, 50 episodes/task). Commit `50bc778`.
Wall time: 36:44.

CrossViewPush required extending the C1 factor menu to 7 (adds
`direction_grounding`). TurnFaucet uses the standard 6-factor menu.
TASK_PROMPT_INFO now carries a per-task `factor_menu` so the two
schema-sizes coexist. Failure frames rendered by job 10837407 (16 min).

| task             | C1 attr | rule-table | C1 pres | C2 pres | Δpres pp | C1 succ | C2 succ | Δsucc pp |
| ---------------- | ------- | ---------- | ------- | ------- | -------- | ------- | ------- | -------- |
| PushCube-v1      | 1.000   | 1.000      | 1.000   | 1.000   | +0.0     | 0.980   | 0.960   | +2.0     |
| PickCube-v1      | 1.000   | 1.000      | 1.000   | 1.000   | +0.0     | 0.920   | 0.000   | +92.0    |
| StackCube-v1     | 0.860   | 0.860      | 1.000   | 0.500   | +50.0    | 0.700   | 0.220   | +48.0    |
| TurnFaucet-v1    | 1.000   | 0.500      | 1.000   | 1.000   | +0.0     | 0.040   | 0.020   | +2.0     |
| CrossViewPush-v1 | 1.000   | 0.000      | 1.000   | 1.000   | +0.0     | 1.000   | 1.000   | +0.0     |

Gates: **all 5 tasks pass all 3 gates** (attr ≥ rule, pres ≥ C2, succ
within 5pp of C2).

Headlines for the two new tasks:

- **TurnFaucet attribution: VLM 1.000 vs rule 0.500.** The held-out cut
  has three failure predicates (`grasp_infeasible` 25, `contact_failure`
  20, `goal_not_satisfied` 5). Rule-table maps only `grasp_infeasible` to
  `embodiment_mapping`; the other two predicates get routed to
  `contact_region` / `goal_state` even though the oracle is always
  `embodiment_mapping` (the gripper can't enclose the handle, period).
  VLM correctly picks `embodiment_mapping` 50/50.
  **TurnFaucet succ low** (C1 2/50, C2 1/50): the poke_turn skill
  primitive is unreliable in real ManiSkill. This is an execution
  problem, not attribution — both conditions succeed at the *revision*
  (frozen_factor_preserved=1.000, factors_changed=embodiment_mapping
  for every episode). Future work to harden the poke_turn skill is
  needed; the P2 attribution claim still holds.
- **CrossViewPush at ceiling.** C1 attr 1.000, succ 1.000. Rule-table
  cannot attribute these failures (it doesn't know about
  `direction_grounding`; only `CrossViewPushAdapter.attribute_failure`
  does), so VLM beats rule by +100pp on attribution. The 7-factor menu
  with the per-task `factor_menu` (TASK_PROMPT_INFO) is doing the work.
  The grounding_substitution revision (actor_frame → observer_frame)
  recovers every episode.

This completes the 5-task main table.

### Stage-5 P2 — PushCube wrist multi-image A/B (job 10937552, 2026-06-01)

InternVL3.5-8B (BF16) on A100-40GB. PushCube only, 50 held-out failures
(seeds 100-149). New capability: `PushCubeEnvRunner(capture_wrist_rgb=True)`
mounts the `panda_wristcam` `hand_camera` (512×512) as an execution
observation; `stage5_p2_render_failure_frames.py` saves a terminal wrist PNG
(`wrist_frame_path` in `episodes.jsonl`) and `stage5_p2_vlm_eval.py` feeds the
VLM **third-person + wrist** via `_chat_multi`. `--no-wrist` re-runs the same
frames single-image for the A/B. Out-dirs: `datasets/stage5/p2_vlm_wrist/`,
`reports/stage5/p2_vlm_wrist{,_nowrist}/`. Both runs pass all gates.

| condition       | C1 attr | C1 pres | C1 succ | C2 pres | C2 fixOracle | C2 succ |
| --------------- | ------- | ------- | ------- | ------- | ------------ | ------- |
| multi (wrist+TP)| 1.000   | 1.000   | 0.980   | 1.000   | 1.000        | 0.980   |
| single (--no-wrist)| 1.000 | 1.000   | 0.980   | 1.000   | 0.920        | 0.880   |

Headline: **C1 (ours) is byte-for-byte identical across both runs** (per-seed
diff = 0 seeds). The wrist channel carries no extra signal for PushCube C1 —
expected, since the failure (`approach_blocked`) is *symbolic* (obstacle never
spawned; all 50 episodes are planner_failed, so the wrist frame is the
post-reset scene and 0 rollout `.npz` are written). The only movement is the
C2 free-form baseline: the wrist view *rescued* it on 4-5 seeds
(102/112/113/129/143) where single-image C2 emitted an unchanged intent →
fixed-oracle 0.92→1.00, succ 0.88→0.98. So multi-image **narrows** the C1−C2
success gap (+10.0pp single → +0.0pp multi); for PushCube the wrist view is
neutral-to-counterproductive for the paper framing. The interesting test
(physically-visible failures) is PickCube/StackCube — not wired for the wrist
view yet.

Caveat: single-image C2 here (0.88) ≠ the historical 0.96 (job 10842893)
because this run's third-person frames are rendered with the `panda_wristcam`
arm (cosmetically different); greedy decoding is deterministic but
input-sensitive, so a few C2 free-form outputs shifted. C1 is robust (1.000 /
0.98 regardless). The change is opt-in (`capture_wrist_rgb` default False) so
the data-collection path and the other four tasks are unaffected.

### Stage-5 P2 — PickCube + StackCube wrist multi-image A/B (job 10937613, 2026-06-01)

Extends the wrist multi-image probe to the two cube tasks that actually STEP
the env (smoke job 10937594 confirmed `panda_wristcam` constructs + grasps for
Pick/Stack; ~2.6 GB Pick + ~0.8 GB Stack per-step `wrist_rgb` .npz). Same
held-out cut (seeds 100-149, 50/task). C1 = ours; both conditions run with and
without the wrist frame (`--no-wrist`) on identical frames. All gates pass in
every cell.

| task      | condition  | C1 attr | C1 pres | C1 succ | C2 succ | C2 parse_fail |
| --------- | ---------- | ------- | ------- | ------- | ------- | ------------- |
| PickCube  | multi      | 1.000   | 1.000   | 0.920   | 0.000   | 0.000         |
| PickCube  | --no-wrist | 1.000   | 1.000   | 0.920   | 0.000   | 0.000         |
| StackCube | multi      | **0.880** | 1.000 | 0.720   | 0.340   | 0.620         |
| StackCube | --no-wrist | 0.860   | 1.000   | 0.700   | 0.220   | 0.760         |

Headline: **StackCube is the first task where the egocentric view moves *ours*
(C1) in the right direction.** Exactly one seed flips — seed 133: with the
wrist frame C1 picks `goal_state` (= oracle, correct); without it C1 picks
`object_motion` (wrong, the canonical StackCube failure mode). That nudges C1
attribution 0.860 → 0.880, so multi-image C1 now **strictly beats** the
rule-table (0.86) on StackCube, whereas no-wrist only ties it. C1 success
0.70 → 0.72. The wrist view also helps the C2 baseline (parse_fail 0.76→0.62,
succ 0.22→0.34), but C2 is the baseline and stays far below C1.

**PickCube: wrist inert** — C1 is byte-for-byte identical (0 seeds differ),
attr 1.000 / succ 0.920 either way (C1 already at ceiling; grasp_slip is
diagnosed from the third-person frame + predicate alone). C2 succ 0.000 both.

Consistency check: the `--no-wrist` StackCube run reproduces the historical
single-image numbers (job 10842893) *exactly* (attr 0.86 / succ 0.70 / C2 succ
0.22 / C2 pres 0.50) even though frames are now rendered with the
panda_wristcam arm — so the StackCube third-person view is robust to the
cosmetic robot change (unlike PushCube, where C2 shifted 0.96→0.88). PickCube
also matches history (1.000 / 0.920 / C2 0.000).

Net across all three cube tasks: the wrist channel is inert where C1 is at
ceiling (PushCube, PickCube) and gives a small real attribution gain where the
failure is relationally visible (StackCube, +1 episode). Out-dirs:
`{datasets,reports}/stage5/p2_vlm_wrist{,_nowrist}/{PickCube,StackCube}-v1`.

### Stage-5 — LATENT-INPUT P2 on PushCube (job 10951957, 2026-06-03)

The latent-intent pivot: the method INPUT is decoded from vision (frozen
DINOv2 → trained IntentHead → per-factor nearest-centroid codebook) instead
of the hand-authored JSON intent, AND C1's repair is the learned slot-local
ReviseHead (not the discrete operator). JSON factors are used only for
supervision (centroid training) + oracle eval. `stage5_p2_vlm_eval.py --latent`
on PushCube, real InternVL3.5-8B (BF16) A100-40gb, same 50 held-out seeds
(100-149). Wall 3:54.

Pre-flight (GPU-free, `reports/stage5/latent_decode_check/PushCube-v1`):
vision-decoded initial intent reproduces the JSON intent **49/50 (0.98)** per
factor (object_motion/contact_region/approach_direction) — so the cached
failure frames + oracle labels stay valid; `n_latent_mismatch=1`.

| condition | attr | frozen-pres | unnec | harmful(mean) | success |
| --------- | ---- | ----------- | ----- | ------------- | ------- |
| **C1 latent (ours)** | 1.000 | **1.000** | 0.000 | 0.000 | **0.960** |
| C2 free-replan | — | 0.700 | 0.300 | 0.047 | 0.680 |

Gates: attr ≥ rule (1.000=1.000) PASS · pres ≥ C2 (+30pp) PASS · succ within
5pp (+28pp) PASS · selectivity-pres (+6pp) PASS · harmful ≤ C2 (−4.7pp) PASS.

Headline: **statistically identical to the discrete fair run** (C1 1.000/0.98
vs C2 0.72/0.70) — replacing the JSON method input with vision-decoded latent
intent AND the discrete operator with the learned slot-local ReviseHead costs
nothing. Answers the "JSON intent is privileged input" reviewer attack on
PushCube end-to-end. PushCube is the consolidated latent task; PickCube is
representation-blocked (contact_region has zero pixel signature) and StackCube
is descoped (goal_state constant in its cut). Out:
`reports/stage5/p2_vlm_latent/PushCube-v1/`.

**Fully-latent G4/G5 (job 10954435, 2026-06-03).** `stage5_p1_run_eval.py
--latent-initial`: attempt-1 intent decoded from vision for ALL policies
(Sever A) + latent revision, real PushCube sim, seeds 100-149, no VLM. Wall
0:45. latent final **0.960** = oracle 0.960 = babysteps_selective 0.960;
same_intent_retry 0.000. **G4 = +96.0pp PASS, G5 = +0.0pp PASS** (latent =
oracle exactly). Identical to the scripted-attempt-1 P1 run (G4 +96, G5 -2) ->
the latent input costs nothing on recovery either. The whole PushCube loop is
now latent-input end-to-end (input + revision). Out:
`reports/stage5/p1_vision_g4_g5_latent/PushCube-v1/`.

### Stage-5 M3 — Procedural baselines main table (job 10826466, 2026-05-26)

A100-40GB, 50 held-out seeds (100-149), three tasks, all seven procedural
retry policies from `babysteps.policies.POLICIES`.

| policy                   | PushCube | PickCube | StackCube |
| ------------------------ | -------- | -------- | --------- |
| `one_shot`               | 0.000    | 0.000    | 0.000     |
| `same_intent_retry`      | 0.000    | 0.000    | 0.000     |
| `random_factor_revision`  | 0.540    | 0.920    | 0.420     |
| **`babysteps_selective`** | **0.980** | **0.920** | **0.700** |
| `text_feedback_replan`    | 0.000    | 0.920    | 0.700     |
| `full_replan_analogue`    | 0.000    | 0.920    | 0.820     |
| `oracle_factor_revision`  | 0.980    | 0.920    | 0.820     |

Headline: **PushCube is the clearest differentiation case.** `babysteps_selective`
(98%) matches oracle and crushes all baselines. `text_feedback_replan` and
`full_replan_analogue` get 0% because perturbing already-correct factors
(especially `approach_direction`) re-breaks the scene. `random_factor_revision`
reaches 54% by chance (sometimes picks the right factor).

PickCube has only one editable factor (`contact_region`), so every method that
does any revision at all ties at 92%. This task is a ceiling sanity check, not
a differentiator.

StackCube: `babysteps_selective` (70%) vs `oracle` (82%) — gap of 12pp from
imperfect rule attribution. `full_replan_analogue` also reaches 82% because
the only goal_state alternative is the correct one; perturbing "all other
factors" is harmless when there's nothing else to break.

## Rules

- Job logs are artifacts — fine to prune old ones; the gate *numbers* live in
this file and in each sub-project's spec, not only in the logs.
- A new GPU gate result gets a row/section here when it lands.

