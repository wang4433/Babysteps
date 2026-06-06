# Paper Narrative: Vision-Grounded Intent Inference & Revision

## The Core Story

```
Demo video (3rd-person) → [Vision Encoder] → [IntentHead] → slot intents G
                                                                 ↓
                                                  some factor naturally wrong
                                                                 ↓
                                               Franka 1st-person execution
                                                                 ↓
                                                              Failure
                                                                 ↓
                                                [VLM Attribution] → "which slot?"
                                                                 ↓
                                              [ReviseHead] → edit ONLY that slot
                                                                 ↓
                                                         retry → success
```

The paper's claim is **not** about how failures happen at the control level.
It is about the **learning loop**: a vision encoder infers structured intent
from a demo video; that inference is naturally imperfect; BABYSTEPS diagnoses
which factor is wrong and revises exactly that one.

## Why Intent Inference Is Naturally Imperfect

A third-person demo video does not perfectly specify every intent factor:

| Source of ambiguity | Affected factor(s) | Example |
| --- | --- | --- |
| Viewpoint occlusion | `contact_region` | Camera angle hides which cube face the gripper contacts |
| Demo underspecification | `goal_state` | "Place near" vs "stack on" indistinguishable in video |
| Affordance gap | `embodiment_mapping` | Demo shows grasp-turn, but execution handle is too thick |
| Cross-view frame | `direction_grounding` | "Push left" in demo camera ≠ "push left" in robot camera |
| Encoder capacity | any factor | Finite-dim latent cannot losslessly encode all factors |

These are not artificial failures. They are inherent limitations of
vision-based intent inference from a single third-person demonstration.

## Five Tasks, Five Inference Failure Modes

| Task | Factor naturally hard to infer | Why inference fails | What happens on execution | Revised factor |
| --- | --- | --- | --- | --- |
| PushCube | `contact_region` | Demo viewpoint conflates which cube face the proxy contacted (same-axis faces project alike) | Robot pushes the wrong face → cube moves away from goal | `contact_region` |
| PickCube | `contact_region` | Demo camera occludes which face was grasped | Robot grasps wrong face → grasp fails or object slips | `contact_region` |
| StackCube | `goal_state` | Demo is ambiguous: "place near" vs "stack on" | Robot places at wrong pose → goal not satisfied | `goal_state` |
| TurnFaucet | `embodiment_mapping` | Demo shows grasp-turn strategy | Robot tries grasp-turn → handle too thick → grasp infeasible | `embodiment_mapping` |
| CrossViewPush | `direction_grounding` | Demo camera ≠ robot camera | Robot interprets "left" in wrong frame → pushes wrong direction | `direction_grounding` |

In every case, the failure is a **single-factor inference error**. The other
five factors are correct. BABYSTEPS diagnoses which one is wrong and revises
only that slot — the headline single-factor invariant.

## Two-Layer Validation

The paper presents two complementary layers of evidence:

### Layer 1: Controlled experiments (Stage 0)

The `blocked_sides` mechanism in the codebase simulates specific factor-level
inference errors. For each seed, we set exactly one factor to a wrong value
and measure:

- Attribution accuracy (does the diagnosis identify the right factor?)
- Revision precision (does ReviseHead fix it?)
- Frozen-factor preservation (are the other 5 factors untouched?)
- Recovery rate (does the retry succeed?)

This is the controlled ablation. We know ground truth (oracle labels), so we
can measure everything exactly. The M3 baselines (one_shot, same_intent_retry,
random_factor_revision, babysteps_selective, full_replan_analogue,
oracle_factor_revision) are evaluated here.

**Paper framing**: "We validate the revision loop under controlled
single-factor errors that simulate vision-encoder inference mistakes."

### Layer 2: End-to-end vision pipeline (Stage 5)

DINOv2/R3M → IntentHead naturally produces wrong intents from demo video
frames. The same revision loop (VLM attribution → ReviseHead) corrects them.

**The Stage-5 failure mechanism is seed variation, NOT `blocked_sides`.**
The `blocked_sides` injection (Layer 1) is gone from the Stage-5 loops. Failures
arise from the natural mismatch between an imperfectly-grounded demo intent and
the execution scene:

- **PushCube — seed-decoupled** (`scripts/stage5_natural_loop_eval.py`): the
  demo clip and the execution scene use *different* seeds, so the goal geometry
  the demo grounds need not match the scene the robot faces. On mismatched
  episodes the open-loop retry recovers 0%; displacement-vector feedback +
  slot-local revision recovers it. No block is applied.
- **StackCube — demo under-specification** (`scripts/stage5_goalstate_loop_eval.py`):
  the natural failure is demo *ambiguity*, not an instance mismatch (a
  whole-clip demo grounds `goal_state` only ~0.63). The under-specified
  `cube_at_target` executes, `goal_not_satisfied` fires, and the
  `goal_refinement` operator lifts it to `cubeA_on_cubeB`. A disambiguating
  (retract) demo grounds it ~0.92, so grounding carries it and revision barely
  fires — the same loop spans a grounding↔revision spectrum.

(Historical note: an earlier Stage-5 latent loop still re-applied
`default_blocked_factory` under `--latent-initial`; that artificial block was
removed in favor of the seed-decoupled / ambiguity-driven failures above.)

**Paper framing**: "We then show the same loop works end-to-end when the
intent is inferred from raw demo video by a frozen vision encoder, without
injecting controlled errors — the failure arises naturally from the
demo↔execution seed mismatch (PushCube) or demo goal under-specification
(StackCube)."

The two layers together make the paper complete:
- Layer 1 proves the mechanism works (controlled, measurable)
- Layer 2 proves it works on real inference errors (end-to-end, natural)

## Render / Figure Plan

### Three-phase figure (per task)

| Panel | Camera | What it shows |
| --- | --- | --- |
| Phase 1: Demo | Third-person external view(s) — **dual-stream**: a global high-oblique view + a closer contact view (`babysteps/render/camera_presets.py`) | Oracle Franka performs the task correctly — this is the input to the vision encoder. Each factor is decoded from the view that sees it (`DualViewIntentExtractor` routing). |
| Phase 2: Attempt | First-person (wrist cam) | Robot executes with the inferred (wrong) intent → natural failure |
| Phase 3: Retry | First-person (wrist cam) | Robot executes with the revised intent → success |

**Camera roles.** Two external cameras feed the *demo* (intent grounding);
the wrist camera is *execution-only* and never enters the demo→intent path.
The dual-stream demo is built and sim-free-tested (`camera_presets.py`,
`DualViewIntentExtractor`); its payoff is the per-factor-observability story
(e.g. PickCube `contact_region`). Note: a high-oblique camera was *falsified*
as the lever for StackCube `goal_state` — that factor is grounded by a
retract-gripper demo render, not by where the camera points (the camera move
is the control that rebuts "your boundary is just the camera placement").

**No obstacles, walls, or clutter needed.** The failure is visible in the
execution itself:
- PushCube: cube moves away from goal (wrong contact face — opposite-face flip)
- PickCube: grasp fails or object slips (wrong contact face)
- StackCube: cube placed at wrong location (wrong goal interpretation)
- TurnFaucet: grasp attempt fails on thick handle (wrong strategy)
- CrossViewPush: cube moves in wrong direction (wrong frame interpretation)

### Figure caption template

> "The demo video (left) shows a successful push toward the goal. The vision
> encoder infers `contact_region=plus_x_face`, but the proxy actually
> contacted the opposite face (the two are ambiguous from the demo viewpoint).
> The first attempt (center) executes the misgrounded intent and pushes the
> cube away from the goal. BABYSTEPS diagnoses `contact_region` as the
> misgrounded factor and revises it to `minus_x_face`. The retry (right)
> succeeds. The other five factors — `goal_state`, `object_motion`,
> `approach_direction`, `constraint_region`, `embodiment_mapping` — are
> preserved unchanged."

## Relationship to Existing Codebase

### What stays the same

The Stage 0 codebase mechanics are unchanged:

- `babysteps/envs/*_runner.py` — runners still *support* `blocked_sides` for
  the Layer-1 controlled ablation, but the **Stage-5 loops do not use it**:
  they create natural failures by seed variation (demo↔execution decoupling)
  or demo under-specification (see Layer 2 above).
- `babysteps/failure.py` — failure predicates and `FAILURE_TO_FACTOR` mapping.
- `babysteps/revision.py` — single-factor revision operators.
- `babysteps/envs/*_adapter.py` — task adapters, attribution logic.
- `babysteps/schemas.py` — intent schema, discrete tokens.
- `babysteps/policies.py` — M3 baseline policies.
- All existing M3 and P2 evaluation results remain valid.

### What changes (narrative only)

- **Paper prose**: Failures are framed as "vision-encoder inference errors,"
  not "scene modifications" or "context drift."
- **Render code**: For paper figures, Phase 2 shows the robot executing the
  wrong intent naturally (no obstacle). The existing render code with
  clutter is kept for reference but is not the paper figure.
- **`CLAUDE.md` sub-projects table**: Updated to reflect inference-error
  framing.

### What the clutter render work achieved

Phase 1 (PushCube clutter render, job 10848944) is complete and the clutter
looks natural. This work is NOT wasted — it can serve as a supplementary
figure showing how the controlled failure is set up in simulation. But the
**main paper figure** uses the new framing: no obstacle, just wrong intent
→ natural failure → revision → success.

## TurnFaucet Status

The Phase 3 diagnostic (job 10848995) completed:

| Category | Count | Rate |
| --- | --- | --- |
| `contact_no_motion` | 21 | 42% |
| `no_contact` | 14 | 28% |
| `partial_rotation` | 11 | 22% |
| `success` | 2 | 4% |
| `mostly_rotated` | 2 | 4% |

Decision: PROCEED with poke_turn fix (port v1 single-sweep from
`scripts/_diag_tf_poke5.py`), 3-day cutoff. If held-out success < 30%,
drop to appendix.

TurnFaucet is the strongest natural failure: grasp-turn genuinely fails on
PartNet handles, and the revision to poke-turn is a real embodiment_mapping
change. This task needs no reframing — its failure was always natural.

If TurnFaucet drops to appendix, frame it as: "VLM attribution correctly
identifies `embodiment_mapping` (100% accuracy), but execution success of the
revised policy remains low due to motion-planning limitations — disentangling
intent revision from motor execution is a key design principle."

## Paper Narrative (final versions)

### Five-task version

> "We evaluate on five manipulation tasks where the vision encoder's intent
> inference from a third-person demo is naturally imperfect:
> (1) the inferred contacted cube face is ambiguous from the demo viewpoint,
> (2) the inferred grasp face is occluded in the demo video,
> (3) the demo is goal-ambiguous between two valid interpretations,
> (4) the inferred manipulation strategy does not transfer to the execution
> object's affordance, and
> (5) the demo viewpoint differs from the robot's egocentric view, causing
> a direction-frame mismatch.
> In each case, exactly one structured intent factor is misgrounded, and our
> method diagnoses and revises only that factor while preserving the other
> five."

### Four-task version (if TurnFaucet drops to appendix)

> "We evaluate on four manipulation tasks where the vision encoder's intent
> inference from a third-person demo is naturally imperfect:
> (1) the inferred contacted cube face is ambiguous from the demo viewpoint,
> (2) the inferred grasp face is occluded in the demo video,
> (3) the demo is goal-ambiguous between two valid interpretations, and
> (4) the demo viewpoint differs from the robot's egocentric view, causing
> a direction-frame mismatch.
> In each case, exactly one structured intent factor is misgrounded, and our
> method diagnoses and revises only that factor. We additionally evaluate on
> TurnFaucet (Appendix X), where the VLM attribution correctly identifies the
> misgrounded factor (`embodiment_mapping`) but execution success remains low
> due to motor-execution limitations rather than the attribution/revision
> pipeline."

## Execution Priority

This reframing is a **narrative change**, not a mechanism change. The codebase
stays the same. The priority ordering:

1. **Update docs** (~1 hour): this file + `CLAUDE.md` sub-projects table.
   Already done.
2. **TurnFaucet poke_turn fix** (~2-3 days): port v1 single-sweep into
   `babysteps/skills/turn.py`. This is the same work regardless of framing.
3. **New render for paper figures** (~1 day GPU): render Phase 2 with the
   robot executing the wrong intent naturally (no obstacle). This is a new
   render mode in `babysteps/render/pushcube.py` etc.
4. **Stage 5 P1 vision encoder** (ongoing): DINOv2 → IntentHead → G1 probe.
   This is the Layer 2 evidence.

The clutter render (job 10848944) and Phase 3 diagnostic (job 10848995) are
done and remain valid as supplementary material.
