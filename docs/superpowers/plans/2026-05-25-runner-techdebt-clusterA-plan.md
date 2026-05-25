# Cluster A — Runner & CLI Tech-Debt Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove three dead/duplicated surfaces (`adapter.compile_skill`, `--rollouts-subdir`, the per-runner `rollout_seed` boilerplate) and document the deliberate render/collection divergence for blocked-approach handling. Zero behavior change; sim-free validation only.

**Architecture:** Each of the four cleanups lands as its own commit, ordered smallest-risk-first (A4 → A2 → A3 → A1). The full 302-test sim-free pytest suite (`tests/CLAUDE.md` is the source of truth; CODE_MAP.md may say 343 — confirm with the baseline run in Task 1) is the only validator needed.

**Tech Stack:** Python 3, pytest, git.

---

## File Structure

Files this plan touches (all already exist; nothing is created):

| File | Change | Why |
| --- | --- | --- |
| `babysteps/envs/pushcube_runner.py` | A4 comment at L159 block; A3 shrink at L151-158 | divergence note + dedup |
| `babysteps/render/pushcube.py` | A4 comment at L294 | reciprocal divergence note |
| `scripts/stage0_collect.py` | A2 delete L74-78 | dead argparse |
| `babysteps/envs/task_adapter.py` | A3 expand docstring at L32-34; A1 delete abstract method at L120-124 | canonical home for `rollout_seed` doc; abstract method dead |
| `babysteps/envs/pickcube_runner.py` | A3 shrink at L130-133 | dedup |
| `babysteps/envs/stackcube_runner.py` | A3 shrink at L152-160 | dedup |
| `babysteps/envs/turnfaucet_runner.py` | A3 shrink at L193-195 | dedup |
| `babysteps/envs/pushcube_adapter.py` | A1 delete `compile_skill` method + docstring bullet | dead surface |
| `babysteps/envs/pickcube_adapter.py` | A1 delete `compile_skill` method + docstring bullet | dead surface |
| `babysteps/envs/stackcube_adapter.py` | A1 delete `compile_skill` method + docstring bullet | dead surface |
| `babysteps/envs/turnfaucet_adapter.py` | A1 delete `compile_skill` method + docstring bullet | dead surface |
| `babysteps/envs/crossview_adapter.py` | A1 delete `compile_skill` method (L117-119) + docstring bullet | dead surface; runner already does world-resolution |
| `tests/test_pushcube_adapter.py` | A1 delete L160-193 parity tests | tests dead surface |
| `tests/test_pickcube_adapter.py` | A1 delete L146-172 parity tests | tests dead surface |
| `tests/test_stackcube_adapter.py` | A1 delete `compile_skill` test section | tests dead surface |
| `tests/test_turnfaucet_adapter.py` | A1 delete `compile_skill` test (~L137) | tests dead surface |
| `tests/test_task_adapter.py` | A1 update comment L57; delete `def compile_skill` from test-stub adapters at L85, L137, L254, L267 | stubs no longer need to override |

Historical specs/plans in `docs/superpowers/specs/` and `docs/superpowers/plans/` reference `compile_skill` and `--rollouts-subdir`. **Do not edit them** — they are dated records and remain accurate to their era.

---

## Task 1: Set up branch and baseline

**Files:**
- No file changes; environment / git setup only.

- [ ] **Step 1.1: Inspect working-tree state**

Run:
```bash
cd /scratch/gilbreth/wang4433/babysteps
git status --short
git rev-parse --abbrev-ref HEAD
```

Expected: branch is `master`. There may be modified files (Stage 5 work-in-progress) — record them. Do NOT discard.

- [ ] **Step 1.2: Stash any work-in-progress to keep Cluster A separable**

If `git status --short` shows modified files, stash them so Cluster A lands on a clean state:
```bash
git stash push -u -m "WIP: pre-clusterA stash (stage 5 docs / scripts)"
git status --short
```

Expected after stash: working tree clean. If there were no WIP files, skip the stash.

- [ ] **Step 1.3: Create the Cluster A branch**

```bash
git checkout -b chore/runner-techdebt-clusterA
git rev-parse --abbrev-ref HEAD
```

Expected: branch is `chore/runner-techdebt-clusterA`.

- [ ] **Step 1.4: Baseline the test suite**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: all tests pass. Record the exact pass count (`N passed`) — this is the baseline. Subsequent tasks compare against it.

If any tests fail at baseline, **STOP** and investigate before proceeding — Cluster A is a no-behavior-change refactor and must start from green.

---

## Task 2: A4 — Document the render/collection divergence (pure additive comments)

**Files:**
- Modify: `babysteps/envs/pushcube_runner.py` (add comment at L159 block)
- Modify: `babysteps/render/pushcube.py` (add comment at L294)

- [ ] **Step 2.1: Add the collection-side divergence comment**

In `babysteps/envs/pushcube_runner.py`, locate the `if skill is None:` block (currently line 160). Insert a comment immediately above the `return AttemptResult(` line inside that block.

Find this code:
```python
        skill = compile_intent_to_push_skill(intent, scene)
        if skill is None:
            return AttemptResult(
                initial_obj_xy=scene.cube_xy,
```

Replace with:
```python
        skill = compile_intent_to_push_skill(intent, scene)
        if skill is None:
            # Deliberate divergence from the render path: collection labels
            # this as planner_failed=True without stepping the env — fast,
            # and the right attribution for the schema. The render-path
            # equivalent (babysteps/render/pushcube.py) spawns a physical
            # red wall and steps until the arm stalls, because reviewers
            # see the MP4. Do not unify.
            return AttemptResult(
                initial_obj_xy=scene.cube_xy,
```

- [ ] **Step 2.2: Add the render-side reciprocal comment**

In `babysteps/render/pushcube.py`, locate the `# === Phase 2 — ATTEMPT (approach physically obstructed) ===` block (currently around line 290).

Find this code:
```python
    # === Phase 2 — ATTEMPT (approach physically obstructed) ===
    # Move the wall onto the demo's approach side, then drive the
    # demo-derived waypoints. The arm reaches the approach standoff,
    # hits the wall, and the no-progress break ends the clip.
    _move_obstacle_to_block(
```

Replace with:
```python
    # === Phase 2 — ATTEMPT (approach physically obstructed) ===
    # Move the wall onto the demo's approach side, then drive the
    # demo-derived waypoints. The arm reaches the approach standoff,
    # hits the wall, and the no-progress break ends the clip.
    #
    # Deliberate divergence from the collection path: render needs a
    # visible failure (MP4 for reviewers), so we spawn a physical wall
    # and step the env. The collection path in
    # babysteps/envs/pushcube_runner.py instead returns
    # planner_failed=True without stepping — fast for 1k-episode runs.
    # Do not unify.
    _move_obstacle_to_block(
```

- [ ] **Step 2.3: Verify tests still pass**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: same pass count as baseline (Task 1 Step 1.4).

- [ ] **Step 2.4: Commit**

```bash
git add babysteps/envs/pushcube_runner.py babysteps/render/pushcube.py
git diff --cached --stat
git commit -m "docs(stage0): note deliberate render/collection divergence

Both pushcube_runner (collection) and render/pushcube (render) handle a
blocked approach, but differently — runner returns planner_failed=True
without stepping (fast schema label); render spawns a physical wall and
steps until stall (visible failure for the MP4). Add reciprocal comments
on both sides so a reader changing one path sees why the other diverges.

No behavior change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: A2 — Delete `--rollouts-subdir` from `stage0_collect.py`

**Files:**
- Modify: `scripts/stage0_collect.py` (delete L74-78)

Background: confirmed via `git grep` that `args.rollouts_subdir` is never read after `parse_args` in any production code or test.

- [ ] **Step 3.1: Delete the argparse declaration**

In `scripts/stage0_collect.py`, find:
```python
    p.add_argument(
        "--rollouts-subdir", type=str, default="rollouts",
        help="Sub-directory of out_dir to hold per-episode rollout .npz "
             "files. Only the real env_runner writes these.",
    )
    args = p.parse_args(argv)
```

Replace with:
```python
    args = p.parse_args(argv)
```

(Delete the 5 `add_argument` lines; keep the `args = p.parse_args(argv)` line that follows.)

- [ ] **Step 3.2: Confirm no remaining references**

```bash
git grep -nE 'rollouts[-_]subdir' -- babysteps/ scripts/ tests/
```

Expected: empty output.

- [ ] **Step 3.3: Run CLI tests + full suite**

```bash
python -m pytest tests/test_stage0_collect_cli.py -q
python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: CLI tests pass; full-suite pass count matches baseline.

- [ ] **Step 3.4: Commit**

```bash
git add scripts/stage0_collect.py
git diff --cached --stat
git commit -m "chore(stage0): drop unused --rollouts-subdir argparse flag

The flag was parsed but never read after parse_args — no production
code or test consumes args.rollouts_subdir. Documented-but-unimplemented
feature. Deleting the dead surface to avoid confusing future readers.

No behavior change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: A3 — Consolidate `rollout_seed` documentation in one place

**Files:**
- Modify: `babysteps/envs/task_adapter.py` (expand EnvRunner Protocol docstring at L23-35)
- Modify: `babysteps/envs/pushcube_runner.py` (shrink L151-158 explainer)
- Modify: `babysteps/envs/pickcube_runner.py` (shrink L130-133 explainer)
- Modify: `babysteps/envs/stackcube_runner.py` (shrink L152-160 explainer)
- Modify: `babysteps/envs/turnfaucet_runner.py` (shrink L193-195 explainer)

`crossview_runner.py` has no inline explainer — leave it alone.

- [ ] **Step 4.1: Expand the EnvRunner Protocol docstring (canonical home)**

In `babysteps/envs/task_adapter.py`, find:
```python
class EnvRunner(Protocol):
    """Minimal env_runner contract. Implementations: the fake in
    tests/conftest.py and the real ManiSkill PushCubeEnvRunner. Future tasks
    add their own runners; the adapter constructs them in make_env_runner.

    This is the canonical EnvRunner Protocol; episode.run_episode imports
    it from here. (It was relocated from babysteps.episode in Plan Task 6.)"""

    def reset(self, seed: int) -> SceneState: ...
    def run(
        self, intent: Intent, scene: SceneState, *, rollout_seed: int | None = None
    ) -> AttemptResult: ...
    def close(self) -> None: ...
```

Replace with:
```python
class EnvRunner(Protocol):
    """Minimal env_runner contract. Implementations: the fake in
    tests/conftest.py and the real ManiSkill PushCubeEnvRunner. Future tasks
    add their own runners; the adapter constructs them in make_env_runner.

    This is the canonical EnvRunner Protocol; episode.run_episode imports
    it from here. (It was relocated from babysteps.episode in Plan Task 6.)

    On `rollout_seed`: kwarg accepted by `run` for fresh-seed-per-attempt
    protocol conformance. All current concrete runners reset from the
    captured episode seed (recorded in `reset`) to hold the scene layout
    fixed across the attempt — the controllers are deterministic, so the
    rollout is a function of (layout, waypoints) alone, and a distinct
    intent changes the waypoints and therefore the outcome. Using
    `rollout_seed` to re-seed the reset would desynchronise the layout
    from the passed-in `scene`. Runners therefore accept the kwarg and
    intentionally ignore it. (Same-intent retry is provably 0% under this
    determinism — that is intentional and documented in the M3 spec.)"""

    def reset(self, seed: int) -> SceneState: ...
    def run(
        self, intent: Intent, scene: SceneState, *, rollout_seed: int | None = None
    ) -> AttemptResult: ...
    def close(self) -> None: ...
```

- [ ] **Step 4.2: Shrink pushcube_runner.py explainer**

In `babysteps/envs/pushcube_runner.py`, find:
```python
    def run(
        self,
        intent: Intent,
        scene: SceneState,
        *,
        rollout_log_path: Optional[Path] = None,
        rollout_seed: Optional[int] = None,
    ) -> AttemptResult:
        """Execute one push attempt for `intent` under `scene` (`scene` carries
        blocked_sides). If the intent is blocked, returns planner_failed without
        stepping the env.

        `rollout_seed` is part of the EnvRunner fresh-seed-per-attempt protocol.
        PushCube resets from the episode seed to hold the scene layout fixed,
        and the prop controller is deterministic, so the rollout is a function
        of (layout, waypoints) alone: a distinct intent changes the waypoints
        and therefore the outcome, while an identical intent reproduces the
        attempt exactly (the spec's "same_intent_retry is provably 0%" caveat).
        It is accepted for protocol conformance and intentionally not used to
        re-seed the reset, which would desynchronise the layout from `scene`."""
```

Replace with:
```python
    def run(
        self,
        intent: Intent,
        scene: SceneState,
        *,
        rollout_log_path: Optional[Path] = None,
        rollout_seed: Optional[int] = None,  # see EnvRunner protocol docstring
    ) -> AttemptResult:
        """Execute one push attempt for `intent` under `scene` (`scene` carries
        blocked_sides). If the intent is blocked, returns planner_failed without
        stepping the env."""
```

- [ ] **Step 4.3: Shrink pickcube_runner.py explainer**

In `babysteps/envs/pickcube_runner.py`, find:
```python
    def run(
        self,
        intent: Intent,
        scene: SceneState,
        *,
        rollout_log_path: Optional[Path] = None,
        rollout_seed: Optional[int] = None,
    ) -> AttemptResult:
        # rollout_seed: EnvRunner fresh-seed-per-attempt protocol. PickCube
        # resets from the episode seed (layout fixed) and the controller is
        # deterministic, so accepting it for protocol conformance is enough;
        # see PushCubeEnvRunner.run for the rationale.
        skill = compile_intent_to_pick_skill(intent, scene)
```

Replace with:
```python
    def run(
        self,
        intent: Intent,
        scene: SceneState,
        *,
        rollout_log_path: Optional[Path] = None,
        rollout_seed: Optional[int] = None,  # see EnvRunner protocol docstring
    ) -> AttemptResult:
        skill = compile_intent_to_pick_skill(intent, scene)
```

- [ ] **Step 4.4: Shrink stackcube_runner.py explainer**

In `babysteps/envs/stackcube_runner.py`, find (lines 144-156):
```python
    def run(
        self,
        intent: Intent,
        scene: SceneState,
        *,
        rollout_log_path: Optional[Path] = None,
        rollout_seed: Optional[int] = None,
    ) -> AttemptResult:
        # rollout_seed: EnvRunner fresh-seed-per-attempt protocol. StackCube
        # resets from the episode seed (layout fixed) with a deterministic
        # controller; accepted for protocol conformance — see
        # PushCubeEnvRunner.run for the rationale.
        skill = compile_intent_to_stack_skill(intent, scene)
```

Replace with:
```python
    def run(
        self,
        intent: Intent,
        scene: SceneState,
        *,
        rollout_log_path: Optional[Path] = None,
        rollout_seed: Optional[int] = None,  # see EnvRunner protocol docstring
    ) -> AttemptResult:
        skill = compile_intent_to_stack_skill(intent, scene)
```

- [ ] **Step 4.5: Shrink turnfaucet_runner.py explainer**

In `babysteps/envs/turnfaucet_runner.py`, find:
```python
    def run(self, intent: Intent, scene: SceneState, *,
            rollout_log_path: Optional[Path] = None,
            rollout_seed: Optional[int] = None) -> AttemptResult:
        # rollout_seed: EnvRunner fresh-seed-per-attempt protocol. TurnFaucet
        # resets from the episode seed (layout fixed); accepted for protocol
        # conformance — see PushCubeEnvRunner.run for the rationale.
        seed = self._last_seed
```

Replace with:
```python
    def run(self, intent: Intent, scene: SceneState, *,
            rollout_log_path: Optional[Path] = None,
            rollout_seed: Optional[int] = None,  # see EnvRunner protocol docstring
            ) -> AttemptResult:
        seed = self._last_seed
```

- [ ] **Step 4.6: Verify tests still pass**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: pass count matches baseline.

- [ ] **Step 4.7: Confirm the in-runner explainers are gone**

```bash
git grep -nE 'EnvRunner fresh-seed-per-attempt protocol' -- babysteps/
```

Expected: zero matches in `babysteps/envs/*_runner.py`. The phrase should only appear (if at all) in `task_adapter.py` — and even there, the new docstring uses different wording.

- [ ] **Step 4.8: Commit**

```bash
git add babysteps/envs/task_adapter.py babysteps/envs/pushcube_runner.py babysteps/envs/pickcube_runner.py babysteps/envs/stackcube_runner.py babysteps/envs/turnfaucet_runner.py
git diff --cached --stat
git commit -m "refactor(envs): consolidate rollout_seed docs in EnvRunner protocol

The 'kwarg accepted, intentionally not used to re-seed' rationale was
duplicated in 4 runners with mild variation. Move the canonical explainer
to the EnvRunner Protocol docstring in task_adapter.py and shrink each
in-runner comment to a one-line pointer at the kwarg declaration.

Net: ~25 lines of duplicate comments removed, ~12 lines of canonical
docstring added. No behavior change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: A1 — Delete `compile_skill` from the adapter surface

**Files (atomic commit — single coherent state):**
- Modify: `babysteps/envs/task_adapter.py` (delete abstract method L120-124)
- Modify: `babysteps/envs/pushcube_adapter.py` (delete impl + docstring bullet)
- Modify: `babysteps/envs/pickcube_adapter.py` (delete impl + docstring bullet)
- Modify: `babysteps/envs/stackcube_adapter.py` (delete impl + docstring bullet)
- Modify: `babysteps/envs/turnfaucet_adapter.py` (delete impl + docstring bullet)
- Modify: `babysteps/envs/crossview_adapter.py` (delete impl L117-119 + docstring bullet)
- Modify: `tests/test_pushcube_adapter.py` (delete parity tests L160-193)
- Modify: `tests/test_pickcube_adapter.py` (delete parity tests L146-172)
- Modify: `tests/test_stackcube_adapter.py` (delete compile_skill test section)
- Modify: `tests/test_turnfaucet_adapter.py` (delete compile_skill test ~L137)
- Modify: `tests/test_task_adapter.py` (update L57 comment; delete 4 test-stub `def compile_skill` at L85, L137, L254, L267)

This is one atomic commit because intermediate states (delete surface without deleting tests, or vice versa) would leave the test suite broken — and our discipline is "every commit is green."

- [ ] **Step 5.1: Delete the abstract method declaration**

In `babysteps/envs/task_adapter.py`, find and delete:
```python
    @abstractmethod
    def compile_skill(self, intent: Intent, scene: SceneState) -> Any:
        """Compile the intent + scene into an executable skill object. The
        env_runner consumes whatever this returns. Returns None when the
        intent is infeasible (e.g., approach_direction in blocked_sides) —
        None propagates as planner_failed=True downstream."""

```

After deletion, the `# ---- overridable hooks: default delegates to shared modules ----- #` section should immediately follow `scripted_demo_to_intent`'s abstract declaration.

- [ ] **Step 5.2: Delete the 5 concrete `compile_skill` implementations**

For each of these adapters, locate the `def compile_skill(self, intent: Intent, scene: SceneState):` method body and delete it (typically 3-5 lines including the body):

- `babysteps/envs/pushcube_adapter.py` (around L76)
- `babysteps/envs/pickcube_adapter.py` (around L87)
- `babysteps/envs/stackcube_adapter.py` (around L85)
- `babysteps/envs/turnfaucet_adapter.py` (around L74)
- `babysteps/envs/crossview_adapter.py` (L117-119)

In each module's top-of-file docstring, also remove the `* compile_skill →` bullet:
- `pushcube_adapter.py:10` — line starting `* compile_skill         → wraps skills.push.compile_intent_to_push_skill`
- `pickcube_adapter.py:14` — same shape
- `stackcube_adapter.py:15` — same shape
- `turnfaucet_adapter.py:19` — same shape
- `crossview_adapter.py:6` — `* compile_skill       → resolve observer-relative intent to world via direction_grounding`

For crossview specifically: the runner (`crossview_runner.py:30-32`) already calls `world_resolved_intent(intent, yaw)` and delegates to the inherited PushCube runner. Deleting the adapter's `compile_skill` removes duplicate logic, not active behavior.

- [ ] **Step 5.3: Delete the dead parity tests**

In `tests/test_pushcube_adapter.py`, delete the entire `# ---------- compile_skill parity --------------------------------------- #` section starting at L160 through L193 (inclusive), comprising:
- The `_correct_push_intent()` helper (if not used elsewhere in the file — verify with a quick grep).
- `test_compile_skill_unblocked_returns_pushskill`.
- `test_compile_skill_blocked_returns_none`.

In `tests/test_pickcube_adapter.py`, delete the entire `# ---------- compile_skill ----- #` section starting at L146 through L172 (inclusive), comprising:
- The `_correct_pick_intent()` helper (verify it isn't used in remaining tests).
- `test_compile_skill_returns_pickskill_when_unblocked`.
- `test_compile_skill_returns_pickskill_even_when_contact_blocked`.

In `tests/test_stackcube_adapter.py`, delete `test_compile_skill_delegates_to_stack_skill` (starts at L171, runs until the next test `test_adapter_inherits_default_hooks` at L191) — delete lines 171 through 190 inclusive, plus any blank-line separator before L171.

In `tests/test_turnfaucet_adapter.py`, delete `test_compile_skill_delegates_to_turn_skill` (starts at L128, runs until the next test `test_adapter_inherits_default_hooks` at L141) — delete lines 128 through 140 inclusive, plus any blank-line separator. **Also check** the helper `_scene_with_extra()` (defined at L15) — grep the file for its uses:
```bash
grep -n "_scene_with_extra" tests/test_turnfaucet_adapter.py
```
If `_scene_with_extra()` is used only by the deleted compile_skill test, also delete its definition at L15. If it is used by surviving tests, leave the definition in place.

**Before deleting any helper, grep within each file** to confirm it isn't reused by surviving tests. The `_correct_push_intent`, `_correct_pick_intent`, `_scene_with_extra` helpers may have other consumers — leave the helper in place if it does; delete only the test functions otherwise.

- [ ] **Step 5.4: Update `tests/test_task_adapter.py` test stubs**

In `tests/test_task_adapter.py`:

- Line 57 has a comment listing abstract methods: `# oracle_wrong_factor, scripted_demo_to_intent, compile_skill.` — delete `, compile_skill` from this comment.
- Line 85: `    def compile_skill(self, intent, scene): return None` — delete this line from its test stub class.
- Line 137: `    def compile_skill(self, intent, scene):` (followed by body) — delete this method from its test stub.
- Line 254: `            def compile_skill(self, intent, scene): return None` — delete from inner test class.
- Line 267: `        def compile_skill(self, i, s): return None` — delete from test stub.

These stubs were satisfying the abstract method contract; with the contract gone, they are vestigial.

**Important**: also check whether any test in this file asserts that `compile_skill` is abstract (e.g., "instantiating a class without compile_skill should raise TypeError"). If such a test exists, delete it — the contract no longer exists. Use:
```bash
grep -n "compile_skill" tests/test_task_adapter.py
```
to enumerate every remaining reference after the deletions above.

- [ ] **Step 5.5: Verify zero remaining `compile_skill` references in non-historical code**

```bash
git grep -nF 'compile_skill' -- babysteps/ scripts/ tests/
```

Expected: empty output. (Historical docs in `docs/superpowers/specs/` and `docs/superpowers/plans/` still contain the term — that is fine and deliberate.)

- [ ] **Step 5.6: Run the full test suite**

```bash
python -m pytest tests/ -q 2>&1 | tail -10
```

Expected: all tests pass. Pass count = baseline minus (# of deleted compile_skill tests). Record the delta and confirm it matches the count of deleted test functions (roughly 6-10, depending on exact counts in stackcube/turnfaucet).

If anything else fails — STOP and inspect. Likely cause: a single-use helper (`_correct_push_intent`, `_correct_pick_intent`, `_scene`) was deleted but is actually used by a surviving test.

- [ ] **Step 5.7: Commit**

```bash
git add babysteps/envs/task_adapter.py \
        babysteps/envs/pushcube_adapter.py \
        babysteps/envs/pickcube_adapter.py \
        babysteps/envs/stackcube_adapter.py \
        babysteps/envs/turnfaucet_adapter.py \
        babysteps/envs/crossview_adapter.py \
        tests/test_pushcube_adapter.py \
        tests/test_pickcube_adapter.py \
        tests/test_stackcube_adapter.py \
        tests/test_turnfaucet_adapter.py \
        tests/test_task_adapter.py
git diff --cached --stat
git commit -m "refactor(envs): delete BaseTaskAdapter.compile_skill (dead surface)

The abstract method was declared on BaseTaskAdapter and implemented by
all 5 task adapters, but NO production code called it: every runner
imports the underlying compile_intent_to_*_skill function directly.
CrossView's compile_skill duplicated world-resolution logic that
crossview_runner.run already performs via world_resolved_intent.

Removing:
  - the abstract method on BaseTaskAdapter
  - 5 concrete adapter implementations (+ 5 docstring bullets)
  - the corresponding parity tests in test_*_adapter.py
  - the test-stub overrides in test_task_adapter.py

If Stage-5 P4 (learned action decoder) later wants a swappable compile
boundary, it can re-introduce the abstraction at that point with a
purpose. Until then, the surface is dead and confusing.

No production behavior change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Final validation and branch summary

- [ ] **Step 6.1: Re-run the full test suite from a clean state**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: all remaining tests pass. Pass count = baseline minus (# deleted compile_skill tests in Task 5).

- [ ] **Step 6.2: Summarise the branch**

```bash
git log master..HEAD --oneline
git diff --stat master..HEAD
```

Expected: 4 commits (one per Task 2/3/4/5). Net line delta should be ~negative (deletions dominate).

- [ ] **Step 6.3: Sanity-grep for stale references**

```bash
echo "--- compile_skill in production code ---"
git grep -nF '.compile_skill(' -- babysteps/ scripts/ || echo "(none)"
echo "--- compile_skill in tests ---"
git grep -nF '.compile_skill(' -- tests/ || echo "(none)"
echo "--- rollouts-subdir anywhere live ---"
git grep -nE 'rollouts[-_]subdir' -- babysteps/ scripts/ tests/ || echo "(none)"
echo "--- duplicated 'fresh-seed-per-attempt protocol' phrase in runners ---"
git grep -nE 'fresh-seed-per-attempt' -- babysteps/envs/*_runner.py || echo "(none)"
```

Expected: all four "live code" greps return `(none)`. Historical docs in `docs/` may still match — that is intentional.

- [ ] **Step 6.4: Restore stashed work-in-progress (if applicable)**

If Step 1.2 stashed any WIP, switch back and pop:
```bash
git checkout master
git stash pop
git status --short
git checkout chore/runner-techdebt-clusterA
```

Expected: the WIP files are back on master; the Cluster A branch is untouched.

- [ ] **Step 6.5: Report**

Print to the user:
- Baseline test count (from Task 1).
- Final test count (from Task 6 Step 1).
- Delta (should equal the # of deleted compile_skill tests, ~6-10).
- 4 commits landed on `chore/runner-techdebt-clusterA`.
- Recommended next step: push the branch and open a PR, or merge directly to master if the project doesn't gate on PRs.

---

## Out-of-scope (deferred)

The following were considered and explicitly deferred (Cluster B / not doing):

1. **Promoting `turnfaucet_runner._execute_skill` to a shared `phase_executor.py`** — high-value but needs GPU re-validation per runner. Write a separate spec when there's appetite.
2. **Adding an `on_step` callback to runners** — YAGNI until Stage-5 vision pipeline asks for it.
3. **Unifying render modules with runners** — different jobs (visual evidence vs schema labels); see the divergence comments added in Task 2.
4. **Editing historical specs/plans in `docs/superpowers/`** — they are dated records of decisions at the time and should stay accurate to their era.

---

## Notes for the executor

- **No GPU node required.** All validation is the sim-free pytest suite.
- **Tests must be green at every commit.** If a commit produces a red suite, fix-or-revert before moving on.
- **Do not edit dated docs in `docs/superpowers/`.** Historical specs/plans referencing `compile_skill` or `--rollouts-subdir` remain accurate to their era.
- **If `rollout_seed` is referenced in render modules (`babysteps/render/*.py`), leave those untouched** — render modules don't call `EnvRunner.run`, they have their own control loops. Cluster A doesn't touch render except for the A4 comment.
- **If you discover the WIP stash conflicts on pop** (Step 6.4), the most likely cause is that the stash modified `CLAUDE.md` or a doc file that Cluster A didn't touch — you should be able to pop cleanly. If not, resolve manually; this is outside the Cluster A scope.
