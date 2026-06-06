# CODE_MAP

One-screen map of the repository. Each code directory also has its own
`CLAUDE.md` with finer detail. Read `goal.md` and root `CLAUDE.md` first.

## Top-level layout

```text
babysteps/        the Python package — all importable, tested logic
  envs/           sim adapters + runners + scene geometry + task dispatch
  render/         per-task three-phase MP4 render flows
  skills/         intent → executable skill compilers (push/pick/stack/turn)
  stage4/         sim-free Stage-4 M1 schema-recoverability probe (analysis)
scripts/          CLI entry points (collect / summarize / render) + diag scratch
tests/            sim-free pytest suite (631 tests) + JSON snapshots
docs/             design specs, TDD plans, locked claim, archived handover
slurm/            sbatch scripts + logs + the canonical GPU run commands
datasets/         collected Stage-0 episode data (JSONL + videos)
renders/          committed example MP4s, one folder per task
reports/          generated analysis reports (e.g. stage4 schema recoverability)
goal.md           Stage-0 boundary + data contract (AUTHORITY)
CLAUDE.md         high-level project instructions
RUNBOOK.md        copy-paste operational commands
README.md         quickstart
technical_def.md  related-work positioning + factorized-intent math
milestones.md     ICLR milestone roadmap (M1–M7)
goal.md / update.md  Stage-0 goal / cross-view pivot rationale
```

Paper-facing comparison policy:
`docs/related_work_and_baselines.md` distinguishes direct failure-recovery
baselines from adjacent action-policy work and records why Diffusion Policy /
ACT are not in the main table.

## `babysteps/` — the package

Pure, sim-free modules orchestrating the loop:

| Module | Job |
| --- | --- |
| `schemas.py` | data contracts: `Intent`, failure packet, revision, episode |
| `demo.py` | scripted demo-evidence utilities (proxy demo → factors) |
| `episode.py` | the Stage-0 loop: demo → intent → execute → fail → revise → retry |
| `failure.py` | structured failure detection + factor attribution |
| `revision.py` | factor-local intent revision (single-factor invariant) |
| `eval.py` | dataset-level metrics + Markdown/JSON report writer |
| `viz.py` | sim-free 2D top-down rendering for episodes |

### `babysteps/envs/` — simulation boundary

- `task_adapter.py` — the `TaskAdapter` interface (boundary for episode/CLI).
- `task_registry.py` — `--task` dispatch table.
- `scene.py` — pure, sim-agnostic scene geometry helpers.
- `<task>_adapter.py` — per-task adapters (pushcube, pickcube, stackcube,
  turnfaucet, crossview). Pure logic: intent build, attribution, frozen factors.
- `<task>_runner.py` — real ManiSkill env wrappers (GPU). `crossview_runner`
  is a thin wrapper over `pushcube_runner` (runs on the real PushCube gym env).

### `babysteps/render/` — MP4 flows

`common.py` + one module per task. Each `render_episode` produces the three
Stage-0 phases: `1_demo`, `2_attempt_blocked`, `3_retry`.

- `camera_presets.py` — shared high-oblique external-camera presets +
  `look_at_pose_list` / `oblique_camera_configs` / `camera_elevation_deg`.
  Sim-free (lazy `mani_skill` import). Backs the Stage-5 **dual-stream camera**
  setup: two external demo views (a global view for final-state/relational
  factors, a closer/oblique view for contact factors), routed per factor;
  the wrist camera is execution-only and never feeds the demo→intent path.
  Presets keep world-z as image-up (deliberately NOT nadir — a top-down view
  collapses stack height and would make `goal_state` a tautology).
- `stackcube.py` carries an optional post-place **retract+dwell** in
  `_execute_stack(..., retract=True)` (lift the open gripper up-and-back so the
  final frames show the placement unoccluded). Default off = byte-identical
  render. This — not a camera move — is what grounds StackCube `goal_state`
  (the high-oblique camera was falsified for that factor; see
  `reports/stage5/goal_state_camera/FINDINGS.md`).

### `babysteps/skills/` — skill compilers

`push.py`, `pick.py`, `stack.py`, `turn.py`. Each turns an `Intent` into an
executable skill (waypoints / motion-plan parameters). `turn.py` dispatches the
embodiment_substitution (poke-turn) path for Sub-project D.

### `babysteps/stage4/` — learned-latent track

Stage 4 (complete) + Stage 5 (active, ICLR target). Stage-4 modules:
`features.py` (firewall-strict 19-dim demo-evidence features), `probe.py`
(linear probe + chance/shuffled baselines), `report.py` (three-way
`cell_class` — trivial / label-identity / geometric — + margin gate),
`collection_plan.py` (stratified + rejection-quota planners for the varied
cut), `intent_head.py` (IntentHead MLP), `revise_head.py` (slot-local
ReviseHead), `attribution_head.py` (M2.5 learned attribution),
`latent_policy.py` (LatentPack + RetryPolicy wrapper).

Stage 5 adds:
- `vision_features.py` (frozen DINOv2/DINOv3/V-JEPA-2.1 feature extraction
  from demo RGB frames — P1). Reads only DemoEvidence-shaped fields, never
  `execution.initial_intent` (the label).
- `vision_intent.py` — decodes the *initial* slot intent from a demo clip:
  frozen encoder → `StandardScaler` → `IntentHead` → nearest-centroid.
  `VisionIntentExtractor` (single view) and `DualViewIntentExtractor` (the
  dual-stream reader — a routing table maps each factor to the external view
  that sees it, so per-factor observability stays legible; concat was rejected
  because it entangles which view grounds which factor). This is what removes
  the JSON intent from the method input.

See `goal.md` §"Stage 5" and
`docs/superpowers/specs/2026-05-24-stage5-vision-encoder-swap-design.md`.

## `scripts/`

- `stage0_collect.py` — collect episodes (real or `--fake-env`) → `samples.jsonl`.
- `stage0_summarize.py` — `samples.jsonl` → `report.{json,md}` (derives task).
- `stage4_probe_schema_recoverability.py` — Stage-4: probe schema
  recoverability from demo evidence → `reports/stage4/` (json + md).
- `stage4_collect_varied.py` — Stage-4 varied-intent collection driver (GPU).
  PushCube goal-move injection (binary ±x) + StackCube rejection sampling
  (4 directions). Mirror sbatch: `slurm/collect_stage4_varied.sbatch`.
- `render_stage0_maniskill.py` — render three-phase MP4s per task (GPU).
- `render_stage0_topdown.py` — sim-free 2D top-down render.
- `smoke_pushcube.py` — PushCube loadability check.

Stage-5 CLIs (GPU for the encoder forward / sim render; sim-free for
train/eval on cached `.npy`):

- `stage5_render_demo_frames.py` — render per-seed demo clips to `.npz`
  (`--camera` renders each external view to its own out-dir for the
  dual-stream cut).
- `stage5_cache_dinov2.py` / `stage5_cache_vjepa.py` — cache frozen features
  per seed; `--frame-select {all,final,first_last,last5}` picks the pooling
  frames (final-state pooling for `goal_state`).
- `stage5_p1_g1_cert.py`, `stage5_p1_train_pack.py`, `stage5_p1_run_eval.py`
  — P1 grounding cert + LatentPack train + held-out eval.
- `stage5_goal_state_probe.py` — StackCube `goal_state` probe; `--camera`
  (high-oblique presets), `--retract`, `--dump-features` (writes both-class
  retract features + `labels.json` as a pack train set).
- `stage5_train_4way_pack.py` (`--factors` subset → per-view packs),
  `stage5_train_goalstate_pack.py` (grounds the 2-class `goal_state` factor).
- `stage5_natural_loop_eval.py` — the PushCube **seed-decoupled** natural-failure
  loop (demo seed ≠ execution seed → natural geometry mismatch, NO injected
  block); vision-decode → execute → displacement-vector feedback → revise.
- `stage5_goalstate_loop_eval.py` — the StackCube `goal_state` loop driver
  (vision-decode → `goal_not_satisfied` → `goal_refinement` operator → retry).
- `stage5_p2_vlm_eval.py` + `stage5_p2_*` — P2 VLM-attribution eval/reports.
- `_diag_*.py` — **scratch** diagnostics (mostly TurnFaucet poke-turn探索).
  Not part of the tested codebase; `_diag_tf_poke5.py` is the empirical
  reference cited by the TurnFaucet runner.

## `tests/`

631 sim-free unit tests. Per-module (`test_schemas`, `test_failure`,
`test_revision`, `test_episode`, `test_eval`), per-task adapter/skill, CLI
(`test_stage0_collect_cli`), render modules, the Stage-4 probe tests
(`test_stage4_smoke/features/probe/report`), plus a single-factor-revision
invariant test and JSON `snapshots/`. Runs on the login node, no GPU.

## `docs/`

- `superpowers/specs/` — design spec per sub-project (dated).
- `superpowers/plans/` — TDD implementation plan per sub-project (dated).
- `milestone1_locked_claim.md` — the locked paper claim.
- `cold_start_handover_archive.md` — historical long-form handover (drifted schema).

## `slurm/`, `datasets/`, `renders/`

Operational + artifact directories. `slurm/CLAUDE.md` holds the canonical GPU
run commands and recorded gate results. See `RUNBOOK.md` for the full command
catalog.
