# Cold-Start Handover (ARCHIVE)

> **Historical reference only.** This is the original long-form handover that
> used to be the root `CLAUDE.md`. It describes a fuller *post-Stage-0* system
> whose **schema and module names do not match the current code**. It is kept
> because it motivated the modular boundaries. For anything authoritative, read
> `goal.md` and the current `CLAUDE.md` / `CODE_MAP.md`.
>
> Key drift to be aware of:
> - Old schema: `target / goal / interaction.{mode,contact_site} /
>   reference_frame / constraints`.
> - Current Stage-0 schema (object-centric): `goal_state / object_motion /
>   contact_region / approach_direction / constraint_region /
>   embodiment_mapping (+ direction_grounding)`.

---

## 1. Project Goal

Build a **failure-guided intent revision framework** for Franka manipulation:

```text
ambiguous demo/instruction
→ structured intent hypothesis
→ action execution
→ structured failure signal
→ revise only the wrong intent factor
→ re-execute
```

The novelty is **factor-level intent revision**, not generic retry.

## 2. What This Project Is Not

```text
- cross-embodiment imitation
- direct language-to-torque control
- end-to-end VLA training
- reinforcement learning
- generic retry after failure
- asking a VLM/LLM to simply "try again"
```

The robot is always Franka/Panda. The low-level controller is a fixed skill
primitive, motion planner, or Cartesian controller.

## 3. Core Research Question

> Can execution failure diagnose which part of a structured task intent was wrong?

Measure not only final success rate, but: did the system revise the *correct*
factor, preserve the correct factors, and improve re-execution?

## 4. Main Difference From Related Work

- **Inner Monologue** updates the planner's *context* and replans. BABYSTEPS
  updates the robot's *belief about what the task meant* and revises one factor.
- **Diffusion Policy** generates actions. BABYSTEPS (optionally) uses
  diffusion-generated counterfactuals to *diagnose* which intent factor to revise.

## 5. System Boundary (five modules)

```text
1. Intent proposer   2. Intent compiler   3. Skill executor
4. Failure detector  5. Intent reviser
```

## 6. Structured Intent (original schema)

```json
{
  "target": "red_block",
  "goal": {"predicate": "left_of", "args": ["red_block", "bowl"]},
  "interaction": {"mode": "push", "contact_site": "right_side"},
  "reference_frame": "object_centric",
  "constraints": ["do_not_move(bowl)"]
}
```

Do not create task-specific fields (`drawer_axis_correct`, `push_side_correct`,
`peg_depth_correct`) — they make the method look hand-designed per task.

## 7. Structured Failure (original)

| Failure type | Likely wrong factor |
| --- | --- |
| `direction_error` | `reference_frame` or `goal` |
| `contact_failure` | `interaction.mode` or `contact_site` |
| `wrong_object_moved` | `target` |
| `goal_not_satisfied` | `goal` |
| `constraint_violation` | `constraints` |
| `planning_failure` | `interaction.mode`, `goal`, or `constraints` |

## 8–14. Compiler, VLM, simulator, baselines, metrics, interfaces

- Intent compiles through a deterministic skill compiler, not directly to torque.
- Default VLM: `Qwen/Qwen2.5-VL-7B-Instruct`, frozen, few-shot. Backups:
  InternVL3-8B, InternVL2.5-8B, LLaVA-OneVision-Qwen2-7B.
- Simulator: **ManiSkill 3** (Franka, tabletop tasks, success predicates,
  RGB-D/segmentation, GPU parallel rollout).
- Baselines: one-shot, failure-agnostic retry, full replanning,
  Inner-Monologue-style feedback replanning, BABYSTEPS-no-diffusion,
  BABYSTEPS-with-diffusion, oracle intent. **Key comparison: BABYSTEPS vs full
  replanning** (full replanning accidentally changes correct fields).
- Primary metrics: `final_success_rate`, `retry_success_rate`,
  `num_attempts_to_success`, `correct_factor_revision_accuracy`,
  `unnecessary_factor_change_rate`.

## 15–17. Milestones

- M1: BABYSTEPS without diffusion on PushCube-v1 (controlled pushing first).
- M2: multiple failure types across PushCube / PickCube / Drawer.
- M3: diffusion counterfactual *scoring* (edit ranking), never as the controller.

## 18. Paper Claim

> Robot execution failure can be used as evidence to revise structured task
> intent. BABYSTEPS revises only the factor likely responsible for failure.
>
> Reviewer one-liner: Inner Monologue replans after feedback; BABYSTEPS
> diagnoses and edits the latent task intent that made the previous plan fail.

## 19. Risks (and mitigations)

1. *Looks like prompt engineering* → VLM is only a proposer; report factor-revision accuracy.
2. *Looks like generic retry* → compare against failure-agnostic retry + full replanning.
3. *Looks task-specific* → same intent schema across all tasks, no diagnostic fields.
4. *Diffusion feels unnecessary* → make it an optional scorer; show it improves edit ranking.

## 20–21. Working Definition

BABYSTEPS is a **test-time structured intent revision framework** for robot
manipulation. It is not a VLA model, a diffusion policy, an RL algorithm, or a
generic LLM retry loop. The implementation uses JSON; the paper describes it as
a structured latent intent hypothesis.

### Original proposed file structure (aspirational — not the current layout)

The original handover proposed `schemas/ vlm/ envs/ skills/ execution/ failure/
revision/ evaluation/ utils/` packages plus a `configs/` tree. The current
Stage-0 code collapses this into a flatter `babysteps/` package (see
`CODE_MAP.md`). The proposal is retained here only as design motivation.
