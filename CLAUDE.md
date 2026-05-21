# BABYSTEPS — project instructions

> **Read `goal.md` first.** It is the Stage-0 authority. If anything here or
> in a subdirectory disagrees with `goal.md`, **`goal.md` wins for Stage 0.**

## What this project is

BABYSTEPS is a **failure-guided structured-intent revision** framework for
Franka manipulation. The core loop:

```text
third-person demonstration proxy
  → structured intent factors
  → Franka first-person execution
  → structured failure packet
  → revise ONLY the implicated intent factor (freeze the rest)
  → retry
```

The research claim is **factor-level intent revision**, not generic retry.
Failure is treated as evidence about *which part of the task intent was
misunderstood*, not just as an action-recovery signal.

## What this project is NOT

- not cross-embodiment imitation, end-to-end VLA, RL, or diffusion-policy control
- not "ask a VLM/LLM to replan the whole thing after failure"
- the robot is always **Franka/Panda**; the low-level controller is a fixed
  skill primitive / motion planner, never learned from scratch

## Stage-0 intent schema (object-centric)

Every task uses the same six factors — no task-specific fields:

| Factor | Meaning |
| --- | --- |
| `goal_state` | desired final object relation or pose |
| `object_motion` | observed / intended object movement |
| `contact_region` | demonstrated or inferred contact site |
| `approach_direction` | route or side used to reach contact |
| `constraint_region` | scene region / object state to preserve |
| `embodiment_mapping` | how a proxy contact maps to a Franka action |
| `direction_grounding` | (Sub-project E) frame the demo direction is read in |

A revision must change **exactly one** factor and preserve the rest. This
single-factor invariant is the headline reviewer-facing property — guard it.

## Sub-projects (Stage-0 task families)

| ID | Task | Revised factor | Failure |
| --- | --- | --- | --- |
| A | PushCube-v1 | `approach_direction` | approach blocked |
| B | PickCube-v1 | `contact_region` | grasp slip |
| C | StackCube-v1 | `goal_state` | underspecified goal |
| D | TurnFaucet-v1 | `embodiment_mapping` | handle not graspable → poke-turn |
| E | CrossViewPush-v1 | `direction_grounding` | observer/actor frame mismatch |

All five are implemented and pass the acceptance gate (`delta_pp >= 10`
between revised-retry and initial-attempt success rates).

## Working invariants (do not violate)

1. **Single-factor revision.** One factor per revision; multi-factor operators
   are a reviewer-visible weakness.
2. **Additive schema changes.** Add new tokens and stop emitting old ones;
   defer token *removal* to a cleanup pass after tests prove no references.
3. **Demo captions describe object evidence**, never an executable Franka motor
   program (the demo is a proxy, not privileged action data).
4. **Keep simulator privilege out of the demo→intent path.** Use it only for
   labels, success checks, and evaluation.
5. **Sim-free tests stay sim-free.** The `tests/` suite must run on the login
   node with no GPU/Vulkan.

## Where to find things

- `goal.md` — Stage-0 boundary, data contract, success criteria (**authority**).
- `CODE_MAP.md` — one-screen map of every directory.
- `RUNBOOK.md` — copy-paste commands (render, collect, summarize, tests).
- `docs/` — design specs + TDD plans (one pair per sub-project).
- Each subdirectory has its own `CLAUDE.md` describing its job.

The older long-form handover that used to live here (richer post-Stage-0
schema and module names) is **archived in `docs/`** as historical motivation.
Its schema does not match the current code; `goal.md` is authoritative.
