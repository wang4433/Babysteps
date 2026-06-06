# BABYSTEPS — project instructions

> **Read `goal.md` first.** It is the Stage-0 authority. If anything here or
> in a subdirectory disagrees with `goal.md`, **`goal.md` wins for Stage 0.**

## What this project is

BABYSTEPS is a **failure-guided structured-intent revision** framework for
Franka manipulation. The core loop:

```text
demo video frames → [vision encoder] → slot intents G
  → Franka first-person execution
  → structured failure packet
  → [VLM diagnosis] → which slot failed?
  → [learned ReviseHead] → edit ONLY that slot (freeze the rest)
  → retry
```

The research claim is **slot-local intent revision in a vision-grounded
latent space**, not generic retry or free-form VLM replanning. Failure is
treated as evidence about *which latent intent factor was misgrounded*,
not just as an action-recovery signal.

## Current priority: Stage 5 (ICLR submission track)

The active development track is **Stage 5** (`goal.md` §"Stage 5"):

1. **P1 — Vision encoder swap** (critical): frozen DINOv2/R3M on demo
   frames → IntentHead → vision-grounded slot intents.
2. **P2 — VLM attribution**: a VLM (InternVL3.5-8B; the step is
   VLM-agnostic) for failure diagnosis (constrained to one factor name,
   never free-form replanning).
3. **P3 — World model counterfactual**: learned dynamics for G3
   selectivity certification.
4. **P4 — Learned action decoder** (optional): replace skill compiler.

Stage 0–4 are complete; their discrete schema and episode data serve as
supervision and certification scaffold for Stage 5.

**Camera setup (Stage 5).** The demo is read from **two external camera
views** — a global high-oblique view for final-state/relational factors and a
closer view for contact factors — with each factor decoded from the view that
sees it (`DualViewIntentExtractor`; presets in `babysteps/render/camera_presets.py`).
The **wrist camera is execution-only** and never feeds the demo→intent path.
(The dual-stream reader is built and sim-free-tested; its payoff is per-factor
observability such as PickCube `contact_region`. A high-oblique camera was
*falsified* as the lever for StackCube `goal_state` — that factor is grounded
by a retract-gripper demo render, not by camera placement.)

## What this project is NOT

- not cross-embodiment imitation, end-to-end VLA, RL, or diffusion-policy control
- not "ask a VLM/LLM to replan the whole thing after failure" — the VLM
  does *diagnosis only* (which factor?); the learned ReviseHead does the edit
- the robot is always **Franka/Panda**; the low-level controller is a fixed
  skill primitive / motion planner (until P4 adds a learned decoder)

## Stage-0 intent schema (object-centric)

Every task uses the same six core factors (plus an additive seventh,
`direction_grounding`, added for Sub-project E) — no task-specific fields:

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

| ID | Task | Revised factor | Inference failure mode |
| --- | --- | --- | --- |
| A | PushCube-v1 | `contact_region` (paper-figure render) / `approach_direction` (Stage-0 clutter ablation) | demo viewpoint conflates which cube face the proxy contacted |
| B | PickCube-v1 | `contact_region` | grasp face occluded in demo video |
| C | StackCube-v1 | `goal_state` | demo goal-ambiguous (place near vs stack on) |
| D | TurnFaucet-v1 | `embodiment_mapping` | demo strategy (grasp-turn) infeasible on execution object |
| E | CrossViewPush-v1 | `direction_grounding` | demo camera ≠ robot camera → direction-frame mismatch |

All five are implemented and pass the acceptance gate (`delta_pp >= 10`
between revised-retry and initial-attempt success rates). In Stage 0,
failures are simulated via the `blocked_sides` mechanism for controlled
ablation. **In Stage 5 the loops drop `blocked_sides` entirely**: failures
arise naturally from the vision encoder's imperfect intent inference, driven
by **seed variation** — demo↔execution seed decoupling (PushCube: the demo
grounds one goal geometry, the execution scene is a different seed) or demo
goal under-specification (StackCube: a whole-clip demo is ambiguous between
place-near and stack-on). See `redesign_failure_paradigm.md`.

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
