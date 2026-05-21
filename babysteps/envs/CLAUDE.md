# babysteps/envs/ — the simulation boundary

This package is where pure loop logic meets ManiSkill. It is split so that the
GPU/Vulkan dependency lives only in the `*_runner.py` files; everything else
(adapters, scene geometry, dispatch) is sim-free and unit-tested.

## Layout

| File | Job | Sim? |
| --- | --- | --- |
| `task_adapter.py` | The `TaskAdapter` interface — the boundary the episode loop and CLI scripts call against. Defines how a task builds an initial intent, attributes failures, and lists frozen factors. | no |
| `task_registry.py` | `--task` dispatch table mapping task ids to adapters/runners. | no |
| `scene.py` | Pure, sim-agnostic scene geometry helpers (poses, sides, grounding-resolution math). | no |
| `<task>_adapter.py` | Per-task adapter (pushcube, pickcube, stackcube, turnfaucet, crossview). Pure logic: intent construction, attribution, frozen-factor sets. | no |
| `<task>_runner.py` | Real ManiSkill env wrapper for that task. Touches Vulkan/GPU. | **yes** |

## Notable structure

- **CrossViewPush** (`crossview_adapter.py` + `crossview_runner.py`) is
  Sub-project E. The adapter adds `observe_demo`, the `actor_frame` egocentric
  mis-grounding, and grounding attribution; it diffs intents over 7 fields.
  The runner is a **thin wrapper over `PushCubeEnvRunner`** — CrossViewPush runs
  on the real `PushCube-v1` gym env (`adapter.gym_env_id`). The observer yaw
  lives in the grounding math, not in a separate sim.
- **TurnFaucet** (`turnfaucet_runner.py`) is the embodiment_substitution
  version: privileged-qpos demo, failing grasp attempt, poke-turn retry with
  auto-sign two-trial dispatch. It handles real ManiSkill merged-link faucet
  joints via the physx articulation array (`_set_faucet_qpos`).

## Rules

- New task → add an adapter (sim-free, with tests) **and** a runner. Register
  both in `task_registry.py`.
- Adapters must not import a runner or ManiSkill — keep them testable on the
  login node.
- Runners are the only place GPU/Vulkan setup belongs.
