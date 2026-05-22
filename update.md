# Stage-0 framing update — single-Franka cross-view

> **Status:** locked 2026-05-21. Supersedes the earlier "Robot A / Robot
> B" two-robot framing that briefly lived in this file. The Stage-0
> setup is, and always was in code, **one Franka, two cameras**.
> `goal.md` is the authoritative data contract; this file records the
> framing decision.

## The setup, in one sentence

> One Franka demonstrates the task on the desk and is recorded from a
> fixed third-person external camera. The same Franka then attempts the
> task and is observed from its own first-person camera (wrist /
> robot-front). BABYSTEPS uses the failure of the first-person attempt
> to revise one structured intent factor, then retries.

## The claim we defend

Use this claim:

> Existing cross-view imitation methods aim to transfer a demonstrated
> behavior across viewpoints. BABYSTEPS studies what happens when that
> transfer fails: the failure trace from the executing Franka's
> first-person view provides evidence about which latent imitation
> target was misgrounded, enabling selective single-factor correction
> and retry.

That is stronger than:

> "Franka can imitate itself across cameras."

because the novelty is **failure-guided correction of latent intent
under cross-view ambiguity**, not the cross-view transfer itself. Prior
imitation-from-observation work studies the cross-view transfer
problem; what is new here is using execution failure as evidence about
which latent intent factor was misinterpreted. ([NeurIPS Papers][1],
[Google Sites][2])

```text
not the cross-view transfer itself
but failure-guided correction of the latent target after cross-view imitation fails
```

## The two cameras

Call the two views explicitly:

```text
demo view : third-person external camera (fixed, in front of the desk)
exec view : first-person camera on the executing Franka (wrist / robot-front)
```

Both are simulated cameras attached to the same ManiSkill scene; the
demonstrator and the executor are the **same Franka kinematic chain**.
We never put two arms in the scene; we never use a non-Franka
demonstrator; we never use a human or a hand proxy.

The clean experimental condition is:

```text
demo view ≠ exec view  (cross-view stressor)
demo arm  = exec arm   (no embodiment confound)
demo task = exec task  (no goal confound)
```

This is what makes the cross-view ambiguity, when it occurs, *attributable
to a single latent intent factor* — the central claim of BABYSTEPS.

## Why same embodiment

Keep the demo and exec arms identical. If they differed (Franka-vs-other
arm, or Franka-vs-hand-proxy), reviewers would ask whether the method
solves view transfer, embodiment transfer, or both. Stage 0 deliberately
removes the embodiment axis so the method is evaluated only on
cross-view ambiguity and on the failure-guided correction loop.

Stage 0 → Stage 3 all stay Franka-to-Franka. Cross-embodiment is a
deferred-and-separately-justified follow-on, not a Stage-0 confound. See
`goal.md` §"Later Stages".

## Add this factor to the schema

The cross-view condition motivates one additional factor over the base
six-factor schema. Use the locked Stage-0 name `direction_grounding`
(see `babysteps/schemas.py`):

```python
direction_grounding = [
    "observer_left",   # left in the third-person demo camera frame
    "observer_right",
    "actor_left",      # left in the executing Franka's first-person frame
    "actor_right",
    "object_left",     # left relative to the object's own frame
    "object_right",
    "world_left",      # left in the world / table frame
    "world_right",
]
```

`direction_grounding` is what gets revised in the CrossViewPush sub-project
when the executing Franka pushes "left" but resolves "left" in the wrong
frame.

## Concrete example

```text
Demo (third-person camera):
the Franka pushes the cube to the LEFT of the goal marker, as seen by
the desk-front camera.

Initial intent inferred by BABYSTEPS:
target_relation     = left_of_goal
direction_grounding = observer_frame  (third-person camera's left)

Execution (first-person camera, executing Franka):
the Franka resolves "left" in its own first-person frame, which is
rotated relative to the demo camera. The push goes the wrong physical
way.

Failure trace:
direction_error = true

BABYSTEPS revision (single factor):
direction_grounding : observer_frame → world_frame
target_relation     : preserved
object_motion       : preserved
contact_region      : preserved

Retry:
the cube ends up on the correct physical side.
```

This is the core story for Sub-project E (CrossViewPush).

## What is novel

The novelty is **not**:

```text
A Franka imitates itself across cameras.
A Franka transfers a demonstrated behavior across viewpoints.
```

Those framings are crowded. The novelty is:

```text
The executing Franka uses its own failed first-person attempt to infer
which latent factor of the demo-derived intent was misgrounded, and
revises only that factor.
```

That is closer to an ICLR-style representation/inference claim:

```text
q(z_imitation | demo_view)
→ q(z_imitation | demo_view, failed_execution)
```

where `z_imitation` is the structured 6+1 factor intent.

## Data we collect

Stay deliberately small and Franka-to-Franka:

```text
1 robot family : Franka / Panda
2 camera roles : third-person external (demo), first-person (exec)
5 task families: PushCube, PickCube, StackCube, TurnFaucet, CrossViewPush
seeds          : 24+ per task for the main table; more for ablations
```

Each episode stores:

```python
episode = {
    "task": "...",
    "demo_view": "third_person_desk_front",
    "exec_view": "first_person_wrist_or_front",
    "initial_intent": {...},
    "first_attempt": trajectory_in_exec_view,
    "failure_trace": {
        "success": False,
        "direction_error": bool,
        "relation_error": bool,
        "contact_miss": bool,
        "visibility_failure": bool,
        "terminal_state_error": bool
    },
    "wrong_factor": "direction_grounding",   # for CrossViewPush; varies per task
    "revised_intent": {...},
    "retry_success": bool,
}
```

## What we compare against

Locked baselines (`docs/milestone1_locked_claim.md` §4):

| Method                     | What it tests                                   |
| -------------------------- | ----------------------------------------------- |
| `one_shot`                 | initial intent only, no retry                   |
| `same_intent_retry`        | retry the same intent (luck-only recovery)      |
| `random_factor_revision`   | revise a random factor; no diagnosis            |
| `babysteps_selective`      | **ours** — diagnose, revise one factor, retry   |
| `text_feedback_replan`     | Inner-Monologue-style feedback replan           |
| `full_replan_analogue`     | full intent regeneration                        |
| `oracle_factor_revision`   | upper bound (revise the GT wrong factor only)   |

Headline metrics: final / retry success, factor-attribution accuracy,
**frozen-factor preservation**, **unnecessary-factor-change rate**, harmful
revision rate.

The result we want:

```text
BABYSTEPS recovers as well as full replanning while changing far fewer
already-correct factors.
```

## Best paper framing

> We study failure-guided revision of structured intent under
> cross-view imitation ambiguity. The same Franka demonstrates a task
> from a third-person desk-front camera and then attempts the task from
> its own first-person camera. When the first-person attempt fails,
> BABYSTEPS uses the structured failure trace to diagnose which latent
> factor — particularly view / direction grounding — was misinterpreted,
> revises only that factor, and retries.

Not:

> "Cross-view imitation."

The first is novel enough to build a paper around. The second is too
broad and already crowded.

[1]: https://papers.neurips.cc/paper/8528-third-person-visual-imitation-learning-via-decoupled-hierarchical-controller.pdf?utm_source=chatgpt.com "Third-Person Visual Imitation Learning via Decoupled ..."
[2]: https://sites.google.com/site/imitationfromobservation/?utm_source=chatgpt.com "Imitation from Observation"
