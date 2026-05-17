# Stage 0 PushCube Blocked-Approach — Implementation Plan

**Source spec:** `docs/superpowers/specs/2026-05-15-stage0-pushcube-blocked-design.md`
**Goal:** Smallest data-prep loop producing JSONL episodes matching `goal.md`
"Episode Data Format" — demo proxy → intent → execute (planner_failed=approach_blocked) → revise → retry succeeds.
**Acceptance:** §13 of the spec.
**Env:** `conda activate handover` on Gilbreth. CPU sim, `obs_mode="state_dict"`.
**TDD discipline:** for every pure module — write the failing test, watch it fail with the expected import/AttributeError, write the minimum implementation, run again, mark complete. Same lockstep pattern as Pick4Pass's plan.

---

## Prerequisites — verify before Task 1

- [x] **Step P1: Conda env exists** — `conda env list` shows `handover`.

- [ ] **Step P2: ManiSkill loads in the handover env** — *(smoke test, not a blocker; see Task 8)*

```bash
cd /home/wang4433/scratch/babysteps
conda activate handover
python -c "import gymnasium as gym, mani_skill.envs;\
 env = gym.make('PushCube-v1', obs_mode='state_dict', control_mode='pd_ee_delta_pose');\
 obs, info = env.reset(seed=0);\
 print('OK; obs keys:', sorted(obs.keys()));\
 env.close()"
```
Expected: `OK; obs keys: ['agent', 'extra']` (or similar). Failure here delays Task 8 only; Tasks 1–7 do not need the simulator.

- [ ] **Step P3: pyproject.toml + package marker** — minimum editable-install scaffolding so tests can `import babysteps.*`.

  Files:
  - Create `babysteps/pyproject.toml`
  - Create `babysteps/babysteps/__init__.py` (empty)
  - Create `babysteps/babysteps/envs/__init__.py` (empty)
  - Create `babysteps/tests/__init__.py` (empty)
  - Create `babysteps/tests/conftest.py` (initial: `import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))`)

  Verify: `cd /home/wang4433/scratch/babysteps && python -c "import babysteps"` returns 0.

---

## Task 1: `babysteps/schemas.py` — all data contracts

**Files:**
- Create: `babysteps/schemas.py`
- Create: `tests/test_schemas.py`

- [ ] **1.1** Write `tests/test_schemas.py` covering:
  - `Intent` round-trip (`from_dict(to_dict)`), frozen, equality.
  - `Intent` validation: rejects unknown `contact_region`, `approach_direction`, `object_motion`, `embodiment_mapping` (whitelists per the spec).
  - `INTENT_FIELDS` constant lists exactly the six factors.
  - `DemoEvidence` round-trip + `rgbd_video_path=None` round-trips as `None`.
  - `SceneState` round-trip; `blocked_sides` tuple-of-str (not list).
  - `AttemptResult` round-trip; required fields exist.
  - `FailurePacket` round-trip; nested `chosen_intent` preserved.
  - `Revision` round-trip; `frozen_factors` is a tuple.
  - `EpisodeRecord.to_jsonl_line()` is one line of valid JSON; round-trips.
  - **Snapshot:** an EpisodeRecord built from canned values dumps to JSON matching the goal.md §"Episode Data Format" example structure (same keys, same nesting).

- [ ] **1.2** Run pytest, see all fail with `ImportError`.

- [ ] **1.3** Implement `babysteps/schemas.py`:
  - Frozen `@dataclass`es per spec §5.
  - Module constants:
    ```python
    INTENT_FIELDS = ("goal_state","object_motion","contact_region",
                     "approach_direction","constraint_region","embodiment_mapping")
    CONTACT_REGIONS = {"minus_x_face","plus_x_face","minus_y_face","plus_y_face"}
    APPROACH_DIRECTIONS = {"from_minus_x","from_plus_x","from_minus_y",
                           "from_plus_y","from_above"}
    OBJECT_MOTIONS = {"translate_+x","translate_-x","translate_+y","translate_-y"}
    EMBODIMENT_MAPPINGS = {"proxy_contact_to_franka_push"}
    FAILURE_PREDICATES = {"none","approach_blocked","direction_error",
                          "contact_failure","no_motion","goal_not_satisfied"}
    REVISION_OPERATORS = {"approach_substitution"}
    CLAIM_BOUNDARY = "third_person_demo_proxy_not_human_demo"
    ```
  - `__post_init__` validates whitelists; on miss, `ValueError` with the offending value listed.
  - `to_dict`/`from_dict` for every class; tuples serialize as lists, deserialize back as tuples.
  - `EpisodeRecord.to_jsonl_line()` returns `json.dumps(self.to_dict(), sort_keys=True)`.

- [ ] **1.4** Pytest green: `pytest tests/test_schemas.py -q`.

---

## Task 2: `babysteps/envs/scene.py` — `SceneState` + small geometry helpers

**Files:**
- Create: `babysteps/envs/scene.py`
- (Tests for these live in `test_execution.py` since they're geometry helpers used by the compiler.)

- [ ] **2.1** `SceneState` already declared in `schemas.py` from Task 1. In `envs/scene.py`:
  - Re-export `SceneState` from `babysteps.schemas` for clarity.
  - Pure helpers:
    - `direction_to_face(goal_vec_xy: np.ndarray) -> str` — snap (dx,dy) to nearest cardinal `*_face` (e.g. goal at +x of cube → `"minus_x_face"`, because the push side is opposite the goal).
    - `face_to_approach(face: str) -> str` — `"minus_x_face" → "from_minus_x"`, etc.
    - `face_to_push_unit(face: str) -> np.ndarray` — direction of cube travel when contacting that face (e.g. `"minus_x_face" → (+1,0)`).
    - `goal_direction_to_motion(goal_vec_xy: np.ndarray) -> str` — `"translate_+x"`, etc., by argmax of |components|.
    - `OPPOSITE_APPROACH: dict[str,str]` mapping table.

- [ ] **2.2** Quick smoke test (inline in `test_execution.py`):
  - `direction_to_face((+1,0)) == "minus_x_face"` (cube pushed toward +x, contact face is -x).
  - `face_to_push_unit("minus_x_face")` ≈ `(+1, 0)`.
  - `goal_direction_to_motion((+1, 0.05)) == "translate_+x"` (snaps on the dominant axis).

---

## Task 3: `babysteps/demo.py` — demo evidence → intent (pure)

**Files:**
- Create: `babysteps/demo.py`
- Create: `tests/test_demo.py`

- [ ] **3.1** Write `tests/test_demo.py`:
  - `demo_to_intent` produces the correct `Intent` for a +x trajectory with `contact_region_label="minus_x_face"`.
  - Test that the signature is `demo_to_intent(evidence: DemoEvidence) -> Intent` and not `(evidence, scene)` — privileged firewall regression guard.
  - `trajectory_to_motion([(0,0),(0.05,0),(0.1,0)]) == "translate_+x"`.
  - `trajectory_to_motion([(0,0),(0,-0.1)]) == "translate_-y"`.
  - `demo_to_intent` raises if `contact_region_label` is not in `CONTACT_REGIONS`.

- [ ] **3.2** See test fail.

- [ ] **3.3** Implement `demo.py`:
  - `trajectory_to_motion(traj) -> str` — net (final - initial) xy, argmax sign-aware snap to one of `OBJECT_MOTIONS`.
  - `demo_to_intent(evidence)` per spec §6.2. Reads only `evidence` fields.
  - `generate_proxy_demo(env_runner, scene)` per spec §6.1 — lives in `demo.py` but takes an `env_runner` so it stays sim-agnostic. *This function is tested via `test_episode.py` with a fake env_runner.*

- [ ] **3.4** Green: `pytest tests/test_demo.py -q`.

---

## Task 4: `babysteps/execution.py` — push skill compiler + feasibility check

**Files:**
- Create: `babysteps/execution.py`
- Create: `tests/test_execution.py`

- [ ] **4.1** Write `tests/test_execution.py`:
  - **Geometry tests** for the scene helpers from Task 2 (move them here).
  - `compile_intent_to_push_skill` returns `None` when `intent.approach_direction in scene.blocked_sides`.
  - `compile_intent_to_push_skill` returns a `PushSkill` with non-empty `waypoints` when unblocked.
  - `build_push_waypoints(scene, intent)` returns a `(3, 7)` ndarray (xyz + xyzw quat).
  - Waypoint 1 is behind the cube along `-push_direction` at travel height.
  - Waypoint 3 is `cube_xy + push_unit * push_travel`, push_travel computed as `min(PUSH_TRAVEL_SCALE * dist(cube,goal), PUSH_TRAVEL_MAX_M)`.
  - The quaternion of every waypoint equals `scene.tcp_start_pose[3:7]`.

- [ ] **4.2** See tests fail.

- [ ] **4.3** Implement `execution.py`:
  ```python
  CUBE_HALF_SIZE = 0.02
  PRE_CONTACT_STANDOFF = 0.005
  PUSH_TRAVEL_SCALE = 0.6
  PUSH_TRAVEL_MAX_M = 0.15

  @dataclass(frozen=True)
  class PushSkill:
      waypoints: np.ndarray   # (3, 7)
      cube_z: float
      contact_region: str

  def build_push_waypoints(scene: SceneState, intent: Intent) -> np.ndarray: ...
  def compile_intent_to_push_skill(intent, scene) -> PushSkill | None: ...
  ```

  Note: `push_direction` is `face_to_push_unit(intent.contact_region)`, NOT
  derived from `intent.approach_direction`. The approach_direction factor is
  purely a feasibility filter — this is what makes Stage 0 revision factor-local.

- [ ] **4.4** Green: `pytest tests/test_execution.py -q`.

---

## Task 5: `babysteps/failure.py` — packet builder + attribution

**Files:**
- Create: `babysteps/failure.py`
- Create: `tests/test_failure.py`

- [ ] **5.1** Write `tests/test_failure.py`:
  - `build_failure_packet` predicate precedence: success → "none"; planner_failed → "approach_blocked"; not reached_contact → "contact_failure"; reached but not moved → "no_motion"; moved opposite → "direction_error"; otherwise "goal_not_satisfied".
  - `build_failure_packet` populates `object_displacement` and `direction_alignment` (cos sim of motion vs goal vec).
  - `attribute_failure` for each predicate returns the spec §7.2 row.
  - `attribute_failure("none")` returns `Attribution(semantic_failure=False, wrong_factor=None, …)`.

- [ ] **5.2** See tests fail.

- [ ] **5.3** Implement `failure.py` per spec §7:
  ```python
  @dataclass(frozen=True)
  class Attribution:
      semantic_failure: bool
      wrong_factor: str | None
      freeze: tuple[str, ...]
      revise: tuple[str, ...]

  FAILURE_TO_FACTOR = { ... per spec §7.2 ... }

  def build_failure_packet(intent, attempt, scene) -> FailurePacket: ...
  def attribute_failure(fp: FailurePacket) -> Attribution: ...
  ```

- [ ] **5.4** Green: `pytest tests/test_failure.py -q`.

---

## Task 6: `babysteps/revision.py` — approach_substitution operator

**Files:**
- Create: `babysteps/revision.py`
- Create: `tests/test_revision.py`

- [ ] **6.1** Write `tests/test_revision.py`:
  - `revise_intent` with `wrong_factor="approach_direction"`:
    - Returns a new Intent whose `approach_direction` differs from input.
    - All other five factors are identical to the input (frozen).
    - The chosen new value is not in `scene.blocked_sides`.
    - `Revision.operator == "approach_substitution"`.
    - `Revision.factor == "approach_direction"`.
    - `Revision.old_value` is the original; `Revision.new_value` is the new one.
    - `Revision.frozen_factors` enumerates the other five exactly.
  - Other `wrong_factor` values raise `NotImplementedError` with a clear message.
  - Edge: if every candidate is blocked, falls back to `"from_above"` (per spec §8).

- [ ] **6.2** See tests fail.

- [ ] **6.3** Implement `revision.py` per spec §8.

- [ ] **6.4** Green: `pytest tests/test_revision.py -q`.

---

## Task 7: `babysteps/episode.py` — the loop (with fake env_runner)

**Files:**
- Create: `babysteps/episode.py`
- Create: `tests/test_episode.py`
- Extend: `tests/conftest.py`

- [ ] **7.1** Add a fake env_runner fixture to `conftest.py`:
  ```python
  @pytest.fixture
  def fake_env_runner():
      """Returns a deterministic env_runner stub: a synthetic SceneState on
      reset(seed); on run(intent, scene) returns a synthetic AttemptResult
      that is success when intent.approach_direction not in scene.blocked_sides
      AND intent.contact_region is opposite the goal direction, else
      planner_failed=True (when blocked) or wrong-direction (when contact_region
      is wrong)."""
  ```

- [ ] **7.2** Write `tests/test_episode.py`:
  - **Happy path:** seeded run of `run_episode` with the fake env_runner produces an `EpisodeRecord` with:
    - `demo.demonstrator_type == "proxy_oracle"`,
    - `execution.success == False`,
    - `failure_packet.failure_predicate == "approach_blocked"`,
    - `failure_packet.wrong_factor == "approach_direction"` (oracle label),
    - `revision.operator == "approach_substitution"`,
    - `retry.success == True`,
    - `metrics.num_attempts_to_success == 2`.
  - **JSON shape:** `EpisodeRecord.to_jsonl_line()` parses back to the same record AND the resulting dict has all top-level keys from goal.md's example (`episode_id, stage, task, claim_boundary, demo, execution, failure_packet, revision, retry, metrics`).
  - **Privileged-firewall guard:** verify `demo_to_intent` was called with a `DemoEvidence` only (use a spy / `mock.patch` to assert the call signature has no `SceneState` parameter).
  - **Already-succeeds path:** if attempt 1 succeeds (e.g., fake env returns success unconditionally for one test), the record has `failure_packet.failure_predicate == "none"`, `revision is None`, `retry is None`.

- [ ] **7.3** See tests fail.

- [ ] **7.4** Implement `episode.py` per spec §6.4. Key code:
  ```python
  EnvRunner = Protocol  # .reset(seed) -> SceneState; .run(intent, scene) -> AttemptResult

  def run_episode(episode_id, seed, env_runner, *,
                  blocked_sides_factory=lambda intent: (intent.approach_direction,),
                  ) -> EpisodeRecord: ...
  ```

- [ ] **7.5** Green: `pytest tests/test_episode.py -q`.

---

## Task 8: `babysteps/envs/pushcube_runner.py` — the real ManiSkill env_runner

**Files:**
- Create: `babysteps/envs/pushcube_runner.py`
- Create: `scripts/smoke_pushcube.py`

*This is the only module that imports `mani_skill`. No unit tests — verified end-to-end by Tasks 9–10 + the smoke test.*

- [ ] **8.1** Implement `smoke_pushcube.py`:
  ```python
  import sys
  try:
      import gymnasium as gym
      import mani_skill.envs  # noqa: F401
      env = gym.make("PushCube-v1", obs_mode="state_dict",
                     control_mode="pd_ee_delta_pose")
      obs, info = env.reset(seed=0)
      print("OK; obs keys:", sorted(obs.keys()))
      env.close()
  except Exception as e:
      print(f"SMOKE FAIL: {type(e).__name__}: {e}", file=sys.stderr)
      sys.exit(1)
  ```

- [ ] **8.2** Run `conda activate handover && python scripts/smoke_pushcube.py`.
  - If OK: proceed to 8.3.
  - If fail: note the error in the report and skip 8.3 + Task 9; ship Tasks 1–7 with the pure-logic verification.

- [ ] **8.3** Implement `pushcube_runner.py`:
  - Class `PushCubeEnvRunner` with:
    - `__init__(self)` — constructs the gym env once.
    - `reset(self, seed: int) -> SceneState` — `env.reset(seed=seed)`, reads `cube_xy`, `cube_z`, `goal_xy`, `tcp_pose` from `obs["extra"]`. Returns SceneState with `blocked_sides=()`.
    - `run(self, intent, scene, *, rollout_path: Path | None) -> AttemptResult`:
      - Compile via `compile_intent_to_push_skill(intent, scene)`.
      - If `None`: return `AttemptResult(planner_failed=True, reached_contact=False, object_moved=False, …, success=False, initial_obj_xy=scene.cube_xy, final_obj_xy=scene.cube_xy, goal_xy=scene.goal_xy)`.
      - Else: replay the 3-phase proportional EE control loop (port from Pick4Pass `_run_one_push`, simplified to take a `PushSkill` directly). Save per-step poses to `rollout_path.npz` if given. Read final `info["success"]`, final `cube_xy`. Return AttemptResult.
    - `close(self)`.
  - Boundary check: `import babysteps.envs.pushcube_runner` succeeds with no `mani_skill` import side effect leaking into `babysteps.{schemas,demo,execution,failure,revision,episode,eval}` (`grep -L mani_skill babysteps/*.py` should list every file).

---

## Task 9: `babysteps/eval.py` + `scripts/stage0_summarize.py`

**Files:**
- Create: `babysteps/eval.py`
- Create: `scripts/stage0_summarize.py`
- Create: `tests/test_eval.py`

- [ ] **9.1** Write `tests/test_eval.py`:
  - `compute_metrics([record_a, record_b, …])` returns the spec §9 dataset-level dict.
  - `final_success_rate`, `retry_success_rate`, `delta_pp`, `non_regression_score` computed correctly on canned records (mix of success/fail/revised/non-regression-violating).
  - `passed_acceptance == (round(delta_pp,10) >= 10.0)`.

- [ ] **9.2** Implement `babysteps/eval.py`:
  - `compute_metrics(records: list[EpisodeRecord]) -> dict`.
  - `write_report(metrics, out_dir)` — writes `report.json` (pretty) + `report.md` (Pick4Pass table style).

- [ ] **9.3** Implement `scripts/stage0_summarize.py`:
  - `python scripts/stage0_summarize.py --samples PATH --out_dir PATH`.
  - Reads JSONL, deserializes each line to `EpisodeRecord.from_jsonl_line`, calls `compute_metrics`, calls `write_report`.

- [ ] **9.4** Green: `pytest tests/test_eval.py -q`.

---

## Task 10: `scripts/stage0_collect.py` — wire it all up

**Files:**
- Create: `scripts/stage0_collect.py`

- [ ] **10.1** Implement:
  ```python
  parser.add_argument("--out_dir", type=Path, required=True)
  parser.add_argument("--n_episodes", type=int, default=5)
  parser.add_argument("--seed_start", type=int, default=0)
  ```
  - Constructs `PushCubeEnvRunner()`.
  - For each `seed in range(seed_start, seed_start + n_episodes)`:
    - `episode_id = f"pushcube_blocked_approach_seed_{seed:04d}"`.
    - `record = run_episode(episode_id, seed, env_runner, …)`.
    - Append `record.to_jsonl_line() + "\n"` to `out_dir / "samples.jsonl"`.
  - After loop, run `compute_metrics` and `write_report`. Print the metrics JSON to stdout (Pick4Pass pattern).

- [ ] **10.2** Smoke run (only if Task 8.2 passed):
  ```bash
  conda activate handover
  python scripts/stage0_collect.py --out_dir datasets/stage0_pushcube_blocked --n_episodes 5 --seed_start 0
  ```
  Expected: `samples.jsonl` with 5 lines; `report.md` shows
  `passed_acceptance == True`; runtime ≤ 90 s.

---

## Task 11: README + housekeeping

**Files:**
- Create: `babysteps/README.md`
- Update: `babysteps/CLAUDE.md` — add a Stage 0 section pointing to `goal.md` as authoritative and noting the spec/plan paths.
- Create: `babysteps/.gitignore` — ignore `datasets/stage0_pushcube_blocked/rollouts/`, `__pycache__/`, `.pytest_cache/`.

- [ ] **11.1** README: project description; quickstart (`conda activate handover; pip install -e .; pytest; python scripts/stage0_collect.py …`); link to the spec.

- [ ] **11.2** CLAUDE.md addendum (only a small Stage 0 footer; keep the existing content untouched to preserve the older design context).

- [ ] **11.3** `.gitignore`.

---

## Final acceptance gate

- [ ] `pytest tests/` exits 0.
- [ ] `python scripts/smoke_pushcube.py` exits 0 — *if and only if* ManiSkill is importable in the handover env. If not, log the failure and proceed.
- [ ] `python scripts/stage0_collect.py --out_dir datasets/stage0_pushcube_blocked --n_episodes 5 --seed_start 0` produces a 5-line `samples.jsonl` with the validation criteria in spec §13.3.
- [ ] `python scripts/stage0_summarize.py --samples datasets/stage0_pushcube_blocked/samples.jsonl --out_dir datasets/stage0_pushcube_blocked` writes `report.md` showing `passed_acceptance: True`.

---

## Notes on dependencies between tasks

```
P1 ─┐
P2 ─┼──> Task 1 ──> Task 3 ──┐
P3 ─┘     │                  ├──> Task 7 ──> Task 8 ──> Task 9 ──> Task 10 ──> Task 11
          └──> Task 2 ──> Task 4 ──> Task 5 ──> Task 6 ──┘                    
```

Tasks 1, 2, 3, 4, 5, 6 are mostly independent once their predecessors are
done — they can be implemented sequentially with no parallelism gain since
they are all small.

Tasks 8 (real env_runner) and 9 (eval) can in principle proceed in parallel
once Task 7 lands, but the smoke run in 8.2 is the only thing that gates
Task 10.

The plan deliberately interleaves tests and implementation per task — TDD
discipline, same as Pick4Pass's `2026-05-14-pushcube-babysteps-loop.md`.
