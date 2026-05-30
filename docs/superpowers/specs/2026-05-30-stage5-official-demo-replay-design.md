# Official ManiSkill demo replay — design + TDD

**Status:** Scope A DONE (verified on GPU, job 10908722); Scope B gated.
**Sub-projects:** A=PushCube-v1, B=PickCube-v1, C=StackCube-v1.
**Authority:** `goal.md` Data Pipeline 1 + working invariants 3–4. This spec does
not override them; it operationalizes the *demonstrator* side within them.

## Why

Today the `1_demo` phase is driven by the babysteps **scripted** waypoint +
proportional-servo rollout (`babysteps/render/{pushcube,pickcube,stackcube}.py`,
the `_execute_*` demo seam). The demo clip is therefore a fabricated
illustration. ManiSkill ships an **official motion-planning oracle** solve per
cube task; rendering *that* third-person gives the paper a stronger demonstrator
provenance ("demos are ManiSkill's official oracle solves, rendered
third-person") at near-zero risk — without changing the intent path.

This is sanctioned, not a new mechanism: `goal.md` Data Pipeline 1 allows the
demonstrator to be "an oracle planner … always the Franka arm itself," and
`envs/CLAUDE.md` invariant 2 lets the *demonstrator* use privileged state while
the *intent extractor* may not. The ManiSkill MP solver is an oracle Panda
planner. Embodiment verified Panda for all three tasks.

## The invariant line (non-negotiable)

ManiSkill official demos are **robot trajectories** (`.h5` with an `actions`
dataset + `env_states`), not videos — privileged action data. The path
**must never open the `actions` dataset** and must never let actions / qpos /
qvel reach `babysteps/stage4/vision_features.py`.

Two firewall-clean ways to source the official demo (both yield frames only):

1. **Run-live (preferred provenance).** Invoke the official solver
   (`solvePushCube`/`solvePickCube`/`solveStackCube`) live and film the
   third-person camera. Reads **no recorded trace at all** — no `.h5` is opened,
   so the firewall question doesn't even arise. This is literally "film the
   sanctioned oracle demonstrator." Needs a working `mplib` planner (currently
   blocked — see below) + a GPU/Vulkan node.
2. **State-replay (unblocked fallback).** Read recorded `env_states` from the
   downloaded `trajectory.h5` and `env.set_state_dict(state)` per frame; never
   `env.step(recorded_action)`, never open the `actions` dataset.
   `set_state_dict` alone fully determines the rendered frame (verified:
   `mani_skill/envs/sapien_env.py`, `scene.py::set_sim_state`). The stock
   `--use-env-states` path still calls `env.step(action)`; we deliberately do
   **not** use it. Needs no `mplib` — works today.

Acceptance question, unchanged: *can `extract_vision_features` see one number
from the recorded action channel?* Must be **no**. (Run-live never opens it;
state-replay reads only `env_states`.)

**Blocker for run-live (discovered during P1):** `mplib` 0.1.1 is installed in
the `handover` env but **fails to import** — NumPy 2.x vs `toppra` (NumPy-1.x
ABI) mismatch (`numpy 2.2.6`; `toppra._CythonUtils` built against 1.x). Fixing
it means `numpy<2`, which risks the NumPy-2-built `mani_skill`/`sapien`. Until
resolved, **state-replay is the executable path**; the helper supports both.

**Control-mode constraint:** the official cube solvers require
`control_mode="pd_joint_pos"` (StackCube asserts it) — incompatible with the
render driver's `pd_ee_delta_pose`. So the official-demo path builds its **own**
env (`OFFICIAL_CONTROL_MODE = "pd_joint_pos"`), independent of the driver.

## Scope decision

| Scope | Changes | Status |
| --- | --- | --- |
| **A — paper-figure demos** | `1_demo` MP4 clips (`scripts/render_stage0_maniskill.py`). Fed to no encoder; no re-validation. | **approved — build now** |
| **B — encoder data re-base** | Stage-5 P1 frame cache (`seed_NNNN.npz`) the DINOv2 encoder consumes. Requires re-deriving labels, regenerating `samples.jsonl`, adding PickCube to the frame producer, re-caching features, **and re-running P1 validation (invalidates the locked PushCube 48/50 number)**. | **gated — decide after seeing A** |

## Environment facts (verified)

- ManiSkill **3.0.0b22** in conda env **`handover`**
  (`/home/wang4433/.conda/envs/handover/bin/python`). Not a `babysteps*` env.
- `~/.maniskill/demos` is empty — must `download_demo` first.
- Rendering needs **GPU/Vulkan**; fails on the login node. Run replay+render on
  an a100/a40 (`--qos=standby` for short jobs). `download_demo` is CPU/network.
- Demos land at `~/.maniskill/demos/<ENV_ID>/motionplanning/trajectory.{h5,json}`.
- `render_camera` (the `rgb_array` render camera) is third-person, fixed
  world-frame, for all three tasks. Must build env with `render_mode="rgb_array"`.

## Design — Scope A

### Shared helper: `babysteps/render/official_demo.py` — **DONE (P1)**

Built. **sim / planner / h5py imports lazy inside function bodies** (login-node
importability; verified no leak at import). Public surface:

```python
PRIVILEGED_H5_KEYS = ("actions", "rewards", "success", "fail",
                      "terminated", "truncated")   # never bracket-indexed
SAFE_STATE_KEY = "env_states"                       # the only .h5 group read
OFFICIAL_CONTROL_MODE = "pd_joint_pos"              # solvers require this

def resolve_official_traj(env_id, demos_root=None) -> (h5_path, json_path)

def run_official_solver_frames(env_id, seed, *, shader="default",
        sim_backend="cpu", capture=render_frame) -> (frames, success)
    # RUN-LIVE: gym.make(control_mode=pd_joint_pos, render_mode=rgb_array),
    # wrap env.step to grab render_frame per step, call solvePushCube/.../
    # solveStackCube. Opens no .h5. success read via res[-1].get("success").

def replay_official_state_frames(env_id, seed=None, *, episode_index=0,
        traj_paths=None, stride=1, sim_backend="physx_cpu") -> (frames, meta)
    # STATE-REPLAY: load json env_kwargs; env.reset(seed=episode_seed);
    # states = dict_to_list_of_dicts(h5[tid][SAFE_STATE_KEY]);  # only group read
    # for st in states[::stride]: env.set_state_dict(st); render_frame.
    # NEVER indexes h5[...]["actions"].

def official_demo_frames(env_id, seed, *, source="solver", **kw) -> frames
    # dispatch: source="solver" (run-live) | "state_replay" (fallback)
```

Reuses `common._to_uint8_frame` / `render_frame` (third-person `env.render()`).
Run-live builds its own `pd_joint_pos` env; state-replay reuses the **recorded**
`control_mode` from `json['env_info']['env_kwargs']` (gym.make needs a valid
controller even though it never steps). Both pass `render_mode="rgb_array"`.

ManiSkill specifics (verified): `env_states` is a nested dict keyed by
actor/articulation name, length T+1; `dict_to_list_of_dicts` slices it per step.
Always `reset(seed=episode_seed)` first so non-state scene config matches before
the teleport pins poses. Downloaded demos are pre-curated successes; the
solver succeeds per-seed only probabilistically (iterate seeds, skip failures —
the official `--only-count-success` pattern).

### CLI wiring: `scripts/render_stage0_maniskill.py`

Add `--demo-source {scripted,official}` (default `scripted` — preserves current
behavior). When `official`, the `1_demo` phase calls `replay_official_demo_frames`
instead of the per-task `_execute_*` demo seam.

Per-task demo seams (confirmed) — add a `_replay_*_demo` mirroring `_execute_*`'s
`reset → append(render_frame) → return {final_obj_xy, success}` contract so
**everything downstream (DemoEvidence, attribution, attempt/retry) is untouched**:

- PushCube: `pushcube.py::_pushcube_setup` (`build_push_waypoints`→`_execute_push`)
- PickCube: `pickcube.py::render_episode` demo block (`_execute_pick(...demo_frames...)`)
- StackCube: `stackcube.py::render_episode` demo block (`_execute_stack(...demo_frames...)`)

Hard rules: demo stays **third-person**; do **not** touch attempt/retry phases;
preserve output naming exactly (`<prefix>_seed_%04d__1_demo.mp4` under the
existing fresh-timestamp dir). Env kwargs already correct in the driver:
`obs_mode="state_dict", control_mode="pd_ee_delta_pose", sim_backend="cpu",
render_mode="rgb_array"` (wristcam only on PushCube execution).

## TDD — Scope A

### Sim-free firewall test: `tests/test_official_demo_firewall.py` — **DONE (P1)**

Mirrors the `inspect.getsource` idiom in `tests/test_stage4_features.py`. Flat in
`tests/`. **7 tests, all green** (the `official_demo` checks run torch-free on
the login node; the encoder handoff `importorskip("torch")` since
`vision_features` imports torch unconditionally and torch is absent on the login
node):

1. **Encoder source scan** — `inspect.getsource(vision_features).lower()`
   contains none of `qpos qvel actions set_state env_state tcp_pose .h5 h5py
   pd_joint mplib`. (Note: the encoder docstring legitimately names
   `initial_intent`/`revision`/`retry`, so those stay in the *existing*
   `test_stage4_features.py` firewall, not this token list.)
2. **Encoder signature** — params ⊆ `{demo_frames,encoder,pool,device,
   resolution,_encoder}` and disjoint from a privileged set.
3. **Safe/privileged keys disjoint** — `SAFE_STATE_KEY == "env_states"`,
   `"actions" in PRIVILEGED_H5_KEYS`, sets disjoint.
4. **No privileged bracket-index** — `official_demo` source contains neither
   `["actions"]` nor `['actions']` (etc.) for every privileged key.
5. **Import sim-free** — `mani_skill`/`sapien`/`h5py`/`mplib`/`gymnasium` not in
   `sys.modules` after import (lazy-import contract).
6. **Public surface** — the four functions exist; `OFFICIAL_CONTROL_MODE ==
   "pd_joint_pos"`.
7. **Fake-h5 handoff** — dict with `actions/qpos/qvel/env_states` + frames; pass
   **only frames** to `extract_vision_features(..., _encoder=_FakeEncoder())`;
   assert `(768,)` float32.

Gate: `python -m pytest tests/ -q` → **483 passed in ~16s** (was 476; +7).

### Phases / gates

- **P1 — helper + firewall tests (sim-free, no GPU).** **DONE.**
  `babysteps/render/official_demo.py` + `tests/test_official_demo_firewall.py`.
  Gate met: `pytest tests/ -q` → 483 passed. Pure addition.
- **Fetch + validate (login node).** **DONE.** `download_demo` for all three →
  `~/.maniskill/demos/<ENV_ID>/motionplanning/trajectory.{h5,json}` (Push 26MB,
  Pick 29MB, Stack 38MB, 1000 episodes each). Validated the `.h5` structure
  sim-free against the helper's assumptions: `traj_N` groups; `env_states` =
  nested `{actors, articulations}`, length T+1; `dict_to_list_of_dicts` works;
  `reset_kwargs = {'options': {}, 'seed': N}`; recorded `control_mode =
  pd_joint_pos`, `render_mode = rgb_array`. **Note:** StackCube's articulation
  is named `panda_wristcam` (Push/Pick: `panda`) — `set_state_dict` matches by
  name and the helper passes no `robot_uids`, so names align; the GPU smoke
  confirms.
- **P2 — Scope A wiring.** **DONE.** `--demo-source {scripted,official}` (+
  `--official-source {state_replay,solver}`, `--official-stride`) on
  `scripts/render_stage0_maniskill.py`. When `official`, only `frames["demo"]`
  is swapped for `official_demo_frames(...)` (its own isolated `pd_joint_pos`
  env); attempt/retry untouched. Unsupported-task guard returns 2 before any
  sim import. Driver imports sim-free; suite 483 green.
- **P0 — GPU smoke (a100-40gb, `handover` env).** **DONE** (job 10908679,
  COMPLETED 22s). `scripts/smoke_official_demo_replay.py` +
  `slurm/smoke_official_demo_replay.sbatch` rendered official demos via
  `replay_official_state_frames` for all three tasks: Push 72 states→36 frames,
  Pick 75→38, Stack 108→54, all (512,512,3) uint8 → MP4 + mid-solve PNG under
  `/home/wang4433/scratch/babysteps/renders/official_demo_smoke/`. PNGs verified
  real (full 0–255 range, std ~56, ~1300 distinct colors). *Gate met: saved PNG
  of the Panda mid-solve.* Run-live (`solver`) still needs the `numpy<2` /
  `mplib` fix first. (NOTE: headless server — renders save under
  `/home/wang4433/scratch/babysteps/renders`.)
- **P2-verify — render `1_demo` clips via the driver.** **DONE** (job 10908722,
  COMPLETED 35s). `render_stage0_maniskill.py --demo-source official
  --official-source state_replay` for all three → 9 MP4s (3 tasks × 3 phases)
  under `renders/official_demo_verify/<task>/`. The `1_demo` clip is the
  official Panda oracle (banner: "official ManiSkill oracle demo (state_replay,
  third-person)"); attempt/retry unchanged from the babysteps env. Visually
  confirmed: Push (cube+bullseye), Pick (cube+green goal), Stack (red+green
  cubes). *Gate met.* `slurm/render_official_demo_verify.sbatch` records the run.
  Known cosmetic warning (pre-existing, not from this change): the babysteps env
  logs `panda_wristcam is not in the task's list of supported robots` on
  PushCube — the official-demo path is unaffected (it builds its own env from
  the recorded env_kwargs, so StackCube's `panda_wristcam`-named articulation
  state matches correctly).

**Scope A is complete.** Helper + firewall tests (P1), driver wiring (P2),
GPU smoke (P0), and end-to-end driver verify (P2-verify) all done and green;
suite 483 passing. Remaining: optionally commit; then P3 (Scope B decision).
- **P3 — Scope B decision gate.** Stop and decide (see below).

## Scope B (gated) — the one real gotcha

Official demos carry **their own** `episode_seed` and scene layout. The current
encoder cut keys `seed_NNNN.npz` to `samples.jsonl` and, for PushCube, **injects
the goal per seed** keyed to `object_motion` (stratified plan in
`scripts/stage5_render_demo_frames.py`). You **cannot** swap official frames
under existing labels — the labels were computed for the babysteps-injected
scene. Adopting official demos for the encoder = **regenerating the whole
varied-intent cut**: derive each label from the official demo's *observed object
trajectory* (object evidence — allowed) and rebuild `samples.jsonl`. Then add
**PickCube** to the frame producer (currently raises `NotImplementedError`;
supports PushCube/StackCube only), re-cache DINOv2, and **re-run P1 validation**
(invalidates the locked PushCube 48/50). This is a data-cut regeneration, not a
render tweak — hence gated.

When wired, B reuses the same helper: add `--demo-source` to
`scripts/stage5_render_demo_frames.py` routing to a `_capture_replay_demo`, and
keep the `seed_NNNN.npz` / key `"frames"` / `(T,H,W,3)` uint8 contract
byte-identical so `stage5_cache_dinov2.py` is unchanged.

## Boundaries

- **Out of scope:** TurnFaucet-D (scoped separately) and the 47-dim Stage-4
  vocab freeze (this changes render *source*, not schema).
- **Invariants preserved:** single-factor revision, additive schema, demo-as-proxy
  all untouched — the demo gets a better *origin*, the intent path stays
  pixels-only.
