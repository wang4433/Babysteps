> **Stage-0 setup (this document is positioning, not a contract).** All
> demonstrations and executions in BABYSTEPS Stage 0 are produced by the
> same robot: a Franka / Panda. The cross-view stressor isolated here is
> *demo camera ≠ execution camera*: one Franka performs the task on the
> desk and is recorded from a fixed third-person external camera; the
> same Franka then attempts the task and is observed from its own
> first-person view (wrist / robot-front camera). There is no human
> demonstrator anywhere in the Stage-0 pipeline. `goal.md` is the
> authoritative data contract; this document is related-work positioning
> for the BABYSTEPS belief-revision loop.

Those papers are related, but they also show what **not** to do. Do **not** make BABYSTEPS another “LLM inner monologue replans after feedback” system. Inner Monologue already uses environment feedback for embodied planning; VoxPoser already uses LLM/VLM reasoning to compose 3D value maps; BrainBody-LLM and similar works already use closed-loop state/simulator feedback to correct LLM plans; RACER already uses VLM supervision for failure recovery in imitation learning. ([arXiv][1])

Your loop should be different:

> **Execution failure updates a structured belief over intent factors, not just the next action or the language plan.**

That is the core.

---

# Proposed BABYSTEPS correction loop

## Core state

BABYSTEPS should maintain three states:

[
E_t = \text{grounded visual evidence}
]

[
B_t(z) = p(z \mid D, X, F_{1:t})
]

[
P_t = \text{executable robot plan conditioned on } z
]

where:

* (D): third-person Franka demonstration video (one Franka, viewed from
  an external desk-front camera)
* (X): current robot scene seen from the executing Franka's first-person
  view (wrist / robot-front camera)
* (F_{1:t}): observed robot failures
* (z): structured latent intent

Use a factorized intent:

[
z = {z_g, z_m, z_c, z_a, z_s, z_e}
]

where:

| Factor | Meaning                     |
| ------ | --------------------------- |
| (z_g)  | goal state                  |
| (z_m)  | object motion               |
| (z_c)  | contact / affordance region |
| (z_a)  | approach direction          |
| (z_s)  | safety or task constraint   |
| (z_e)  | embodiment-specific mapping |

This is the representation BABYSTEPS updates.

---

# The loop

## Step 1: Parse the third-person Franka demo into grounded candidates

The demo is one Franka executing the task on the desk, recorded from a
fixed external (third-person) camera. BABYSTEPS never sees the demo
Franka's joint trace as input — only the third-person video.

Use DINO/DINOv2-style features, segmentation, optical flow, and possibly depth to extract grounded evidence:

* object mask
* object parts / regions
* end-effector / object contact region (Franka gripper-object contact in the demo)
* object motion
* before/after object state
* candidate affordance regions
* third-person demo view to first-person execution view correspondences

DINOv2 is useful here because it provides general visual features across image and pixel-level tasks, but it should be treated as a grounding module, not an intent generator. ([arXiv][2])

Output should look like:

```json
{
  "object_regions": ["r1", "r2", "r3", "r4"],
  "demo_contact_region": "r2",
  "object_motion": "translate-right",
  "final_state": "object-at-target",
  "candidate_contact_regions": ["r1", "r2", "r3"],
  "candidate_constraints": ["keep-r2-free", "avoid-r4", "approach-from-left"]
}
```

Do not let the VLM free-form hallucinate intent. Force it to choose from
grounded candidates extracted from the third-person Franka demo.

---

## Step 2: Generate multiple intent hypotheses

The VLM proposes a distribution over structured intent factors:

[
B_0(z) = p(z \mid D, X)
]

Example:

```json
{
  "goal_state": {
    "move-object-to-target": 0.78,
    "align-object-with-target": 0.22
  },
  "contact_region": {
    "r2": 0.52,
    "r3": 0.31,
    "r1": 0.17
  },
  "approach_direction": {
    "left": 0.61,
    "top": 0.24,
    "right": 0.15
  },
  "constraint_region": {
    "keep-r2-free": 0.46,
    "none": 0.41,
    "avoid-r4": 0.13
  }
}
```

This is important: BABYSTEPS should not produce one intent. It should keep **uncertainty**. Without uncertainty, failure revision becomes arbitrary.

---

## Step 3: Plan from the current intent hypothesis

Select a feasible hypothesis:

[
z_t^* = \arg\max_z B_t(z) \cdot \text{Feasible}(z, X, E_r)
]

where (E_r) is the robot embodiment.

Then convert intent into robot-executable constraints:

* contact region → grasp/contact target
* approach direction → motion planner constraint
* goal state → task success predicate
* safety/task constraint → forbidden region or value-map penalty
* object motion → trajectory objective

VoxPoser is relevant here because it composes affordance and constraint value maps from LLM/VLM outputs for manipulation planning. BABYSTEPS can use that idea, but the novelty should be the **post-failure update of intent**, not the value-map planner itself. ([arXiv][3])

---

## Step 4: Execute and record a structured failure packet

Do not feed the VLM only a video and ask “what went wrong?” That is too loose.

Create a structured failure packet:

```json
{
  "attempt_id": 1,
  "chosen_intent": {
    "goal_state": "move-object-to-target",
    "contact_region": "r2",
    "approach_direction": "left",
    "constraint_region": "none"
  },
  "execution_trace": {
    "reached_contact": true,
    "contact_region_actual": "r2",
    "object_moved": false,
    "collision": false,
    "grasp_slip": false,
    "planner_failed": false,
    "task_region_blocked": true
  },
  "failure_predicate": "task-region-blocked",
  "visual_evidence": ["robot occupied r2", "target region unchanged"]
}
```

This is where BABYSTEPS becomes more rigorous than an LLM monologue.

---

## Step 5: Decide whether this is an intent failure

You need a gate:

[
p(\text{semantic intent error} \mid f_t, \tau_t, X) > \lambda
]

Failure categories:

| Failure type                    | Response                             |
| ------------------------------- | ------------------------------------ |
| Perception failure              | re-perceive, do not revise intent    |
| Motion planning failure         | replan path, do not revise intent    |
| Control/slippage failure        | local recovery, do not revise intent |
| Wrong contact/affordance        | revise intent                        |
| Wrong goal interpretation       | revise intent                        |
| Wrong constraint interpretation | revise intent                        |
| Embodiment mismatch             | revise embodiment mapping            |

This gate is critical. Without it, reviewers will say BABYSTEPS over-interprets ordinary execution errors as semantic errors.

RACER is a direct baseline threat because it uses a VLM supervisor and language-conditioned actor for failure recovery. Your distinction must be: RACER guides recovery actions; BABYSTEPS updates a latent intent belief. ([arXiv][4])

---

## Step 6: Attribute the failure to one factor

Infer:

[
c_t = \arg\max_i p(z_i \text{ caused failure} \mid D, X, z_t, \tau_t, f_t)
]

Example:

```json
{
  "semantic_failure": true,
  "implicated_factor": "contact_region",
  "confidence": 0.81,
  "reason": "The selected contact region was physically reachable, but occupying it prevented the task condition from being satisfied.",
  "factors_to_freeze": ["goal_state", "object_motion", "approach_direction"],
  "factor_to_update": "contact_region"
}
```

This factor attribution module should eventually be learned, not purely prompted.

Start with oracle labels in simulation, then train a small classifier:

[
h_\theta(E_t, z_t, \tau_t, f_t) \rightarrow p(c_t)
]

Inputs:

* DINO visual region embeddings
* VLM symbolic factors
* robot trajectory features
* failure predicates
* contact/collision/reachability logs

Outputs:

* semantic vs non-semantic failure
* implicated factor
* confidence
* suggested revision candidates

---

## Step 7: Locally revise only the implicated factor

This is the most important part.

Do not regenerate the whole plan.

If (c_t = z_c), revise only contact:

[
B_{t+1}(z_c) \propto B_t(z_c) \cdot p(f_t \mid z_c, X)
]

and freeze:

[
z_{\neg c}^{t+1} = z_{\neg c}^{t}
]

Example update:

```json
{
  "before": {
    "contact_region": {
      "r2": 0.52,
      "r3": 0.31,
      "r1": 0.17
    }
  },
  "failure_evidence": "r2 caused blocked task region",
  "after": {
    "contact_region": {
      "r2": 0.04,
      "r3": 0.71,
      "r1": 0.25
    }
  }
}
```

Then retry using the revised factor.

This is what makes BABYSTEPS different from full replanning.

---

# The full algorithm

```text
Input:
  Third-person Franka demo D
  Robot scene X (first-person view of executing Franka)
  Robot embodiment E_r (Franka / Panda)
  Max retries K

1. Extract grounded visual evidence:
     E_0 = Ground(D, X)

2. Generate initial structured intent belief:
     B_0(z) = IntentHypotheses(E_0, D, X)

3. For t = 0 ... K:
     z_t = SelectFeasibleIntent(B_t, X, E_r)

     P_t = Plan(z_t, X, E_r)

     Execute P_t and collect trace τ_t

     If Success(τ_t, z_t):
         Store success case
         Return success

     f_t = BuildFailurePacket(τ_t, X, E_t, z_t)

     q_t = ClassifyFailureType(f_t)

     If q_t is non-semantic:
         P_t = LocalRecovery(P_t, f_t)
         Continue

     c_t = AttributeFailureToIntentFactor(f_t, z_t, E_t)

     B_{t+1} = SelectiveBeliefUpdate(B_t, c_t, f_t)

     Freeze all non-implicated factors

4. Return failure with final diagnosis
```

---

# Make the loop two-level

You should have both an **online loop** and an **offline learning loop**.

## Online loop: trial-time belief revision

This is what happens during one task episode.

* no weight update
* no finetuning
* update only (B_t(z))
* retry with revised intent

This keeps the system stable.

## Offline loop: improve the diagnoser

After many episodes, train:

1. failure-type classifier
2. failure-to-factor attribution model
3. revision-ranking model
4. success verifier

This is where the paper can become more ICLR-like.

LMPC is relevant because it treats robot–teacher interaction as a POMDP
and uses experience to improve teachability across tasks/embodiments. But
BABYSTEPS should not require external corrections of any kind (human or
otherwise); it should use the executing Franka's own failures as the
correction signal. ([arXiv][5])

---

# Use the related papers as baselines

The papers in your graph suggest natural baselines:

| Baseline                                  | What it tests                                            |
| ----------------------------------------- | -------------------------------------------------------- |
| Inner Monologue-style feedback replanning | Does generic language feedback solve it?                 |
| VoxPoser-style recompose value maps       | Is revising value maps enough?                           |
| Full VLM replan after failure             | Is selective revision better than complete regeneration? |
| RACER-style action recovery               | Is this intent revision or just recovery?                |
| RT-2 / Octo / VLA policy                  | Does a generalist policy already handle retry?           |
| Random factor revision                    | Does attribution matter?                                 |
| Oracle factor revision                    | Upper bound for BABYSTEPS                                |

RT-2 shows the strength of vision-language-action models trained on robot data; Octo is a generalist robot policy trained on Open X-Embodiment-scale data. These are relevant as strong policy baselines or low-level executors, but they should not be the conceptual core of BABYSTEPS. ([arXiv][6])

---

# What should be updated?

Update **intent factors**, not raw language and not weights online.

## Good update targets

| Factor             | Example correction                                                        |
| ------------------ | ------------------------------------------------------------------------- |
| Goal state         | “The object should be upright, not merely moved.”                            |
| Contact region     | “Do not reuse the same contact region as the third-person Franka demo.”      |
| Approach direction | “The demonstrated approach is blocked at execution time; use top approach.”  |
| Constraint region  | “This region must remain free.”                                              |
| Object affordance  | “The contacted part is a handle/lever/support, not a generic grasp site.”    |
| Embodiment mapping | “The demo grasp-and-turn maps to a closed-gripper poke-turn at execution.”   |

## Bad update targets

| Bad target                | Why                                      |
| ------------------------- | ---------------------------------------- |
| Whole prompt              | too unstable                             |
| Whole trajectory          | loses useful information from the demo   |
| Whole policy              | too slow and data-hungry                 |
| Free-form VLM explanation | hard to evaluate                         |
| Direct action tokens      | collapses into ordinary failure recovery |

---

# The correction operator library

Define explicit revision operators.

## 1. Reweight

Use when one factor had multiple candidates.

```text
contact_region: r2 → r3
```

## 2. Constraint insertion

Use when failure reveals a forbidden region.

```text
constraint_region += keep r2 free
```

## 3. Role reassignment

Use when the demo-Franka contact site should not be copied verbatim by
the executing Franka (e.g. because the execution-time scene blocks it).

```text
demo_contact_region = task-relevant region observed in the demo
exec_contact_region = alternative reachable region the executing Franka uses
```

## 4. Approach substitution

Use when the intended interaction is correct but the route is infeasible.

```text
approach_direction: left → top
```

## 5. Goal refinement

Use when the final state was under-specified.

```text
goal: move object to target → align object orientation at target
```

## 6. Embodiment remapping

Use when the same semantic action observed in the demo requires a
different physical contact at execution time. Even Franka-to-Franka, the
execution-time scene (e.g. a faucet handle whose width exceeds the
gripper opening) can force a different skill primitive.

```text
demo grasp-and-turn → execution closed-gripper poke-turn
demo top-grasp → execution side approach
```

These operators make the system testable. They also prevent the VLM from rewriting everything.

---

# How to use VLMs safely inside the loop

Use the VLM in three places only:

1. **Intent proposal**

   * choose structured factors from grounded candidates

2. **Failure explanation**

   * summarize failure evidence in symbolic form

3. **Revision proposal**

   * rank candidate revisions

But the final update should be constrained:

```json
{
  "allowed_update": {
    "factor": "contact_region",
    "candidate_values": ["r1", "r3", "r4"]
  },
  "forbidden": [
    "changing_goal_state",
    "changing_object_identity",
    "changing_success_predicate"
  ]
}
```

This avoids the biggest weakness of Inner Monologue-style systems: uncontrolled language-level plan drift. Inner Monologue showed the usefulness of environment feedback for LLM planning, but BABYSTEPS should be stricter and more grounded. ([arXiv][1])

---

# What makes this publishable

The publishable claim is not:

> We retry after failure.

The publishable claim is:

> We introduce a factorized belief-revision loop where Franka execution failures selectively update latent intent inferred from a third-person Franka demonstration of the same task.

To prove that, you need these metrics:

| Metric                                     | Why it matters              |
| ------------------------------------------ | --------------------------- |
| retry success rate                         | basic outcome               |
| number of retries to success               | efficiency                  |
| failure-to-factor attribution accuracy     | proves diagnosis works      |
| harmful revision rate                      | shows safety of update      |
| unnecessary revision rate                  | tests semantic-failure gate |
| success under cross-view mismatch          | tests perception robustness |
| success under contact/affordance ambiguity | tests core novelty          |

The most important ablation:

```text
BABYSTEPS selective factor revision
vs.
full VLM replanning after failure
vs.
action-level recovery
vs.
random factor revision
vs.
oracle factor revision
```

If selective revision does not beat full replanning and action recovery, the paper becomes weak.

---

# My recommended final loop design

Use this as the actual BABYSTEPS pipeline:

```text
Third-person Franka demo (one Franka, external desk-front camera)
   ↓
DINO/SAM/flow grounding
   ↓
Candidate object regions, contact traces, object motion
   ↓
VLM proposes structured intent belief B_0(z)
   ↓
Feasibility-aware planner chooses executable intent
   ↓
Franka executes from its first-person view (wrist / robot-front camera)
   ↓
Failure monitor creates structured failure packet
   ↓
Semantic-failure gate:
      perception/control/planning failure → local recovery
      intent failure → continue
   ↓
Failure-to-factor attribution:
      goal? contact? approach? constraint? affordance? embodiment mapping?
   ↓
Selective belief update:
      revise only implicated factor
   ↓
Retry (same Franka, same first-person view)
   ↓
Store episode for offline training of attribution/revision model
```

This is the version I would pursue.

The clean paper sentence:

> BABYSTEPS differs from closed-loop LLM planning and failure recovery by
> treating Franka execution failures as observations in a structured
> latent-intent belief state. Stage 0 isolates the mechanism with a
> deliberately controlled cross-view setup: one Franka demonstrates the
> task on the desk and is recorded from a third-person external camera;
> the same Franka then executes the task and is observed from its own
> first-person view. Each retry updates only the factor implicated by
> the failure, rather than regenerating the full plan or directly
> recovering the action.

[1]: https://arxiv.org/abs/2207.05608?utm_source=chatgpt.com "Inner Monologue: Embodied Reasoning through Planning with Language Models"
[2]: https://arxiv.org/abs/2304.07193?utm_source=chatgpt.com "DINOv2: Learning Robust Visual Features without Supervision"
[3]: https://arxiv.org/abs/2307.05973?utm_source=chatgpt.com "VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models"
[4]: https://arxiv.org/abs/2409.14674?utm_source=chatgpt.com "RACER: Rich Language-Guided Failure Recovery Policies for Imitation Learning"
[5]: https://arxiv.org/abs/2402.11450?utm_source=chatgpt.com "Learning to Learn Faster from Human Feedback with Language Model Predictive Control"
[6]: https://arxiv.org/abs/2307.15818?utm_source=chatgpt.com "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control"
