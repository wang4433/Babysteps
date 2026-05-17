# BABYSTEPS Stage 0 Goal

## Purpose

Stage 0 is a controlled data-preparation and loop-validation phase for BABYSTEPS.

The goal is to prove the smallest honest version of the system:

```text
third-person demonstration proxy
-> structured intent factors
-> Franka first-person execution
-> structured failure packet
-> selective intent-factor revision
-> retry
```

This stage should validate the data contracts and correction loop before adding real human demonstrations, real-world Franka execution, or learned perception/revision models.

## Boundary Line

Stage 0 may use ManiSkill and Franka-generated demonstrations, but those demonstrations are not human demonstrations.

Use this wording:

> third-person demonstration proxy

Do not use this wording:

> human demonstration

Stage 0 tests:

- Cross-view intent transfer: external demo view to robot-centric execution view.
- Failure-guided intent revision: failure updates a structured intent factor.
- Franka execution loop: intent compiles into executable robot behavior.
- Data preparation: every episode stores enough evidence for later training and evaluation.

Stage 0 does not test:

- Full human-to-robot embodiment transfer.
- Real human video understanding.
- Real-world robot deployment.
- End-to-end VLA control.
- Generic LLM replanning after feedback.

The paper-facing claim for Stage 0 is:

> We evaluate BABYSTEPS in a controlled simulated setting where third-person demonstration proxies provide object-centric intent evidence, while a Franka executes from a robot-centric view. This isolates the failure-guided structured intent revision mechanism before moving to real human demonstrations.

## Core Research Invariant

BABYSTEPS updates structured intent factors, not raw language, policy weights, or whole action trajectories.

The key invariant is:

```text
failure -> identify implicated intent factor -> revise only that factor -> preserve the rest
```

If the implementation asks a VLM or LLM to freely regenerate the whole plan after failure, it is no longer testing the main BABYSTEPS claim.

## Stage 0 Intent Factors

Use a compact object-centric intent representation for data preparation:

```json
{
  "goal_state": "cube_at_target",
  "object_motion": "translate_right",
  "contact_region": "left_face",
  "approach_direction": "from_left",
  "constraint_region": "none",
  "embodiment_mapping": "proxy_contact_to_franka_push"
}
```

These factors correspond to the broader BABYSTEPS latent intent:

- `goal_state`: desired final object relation or pose.
- `object_motion`: observed or intended object movement.
- `contact_region`: demonstrated or inferred contact site.
- `approach_direction`: route or side used to reach contact.
- `constraint_region`: scene region or object state that must be preserved.
- `embodiment_mapping`: how a proxy/human-like contact maps to Franka action.

Avoid task-specific fields such as `drawer_axis_correct`, `push_side_correct`, or `peg_depth_correct`.

## Stage 0 Data Pipeline

### 1. Generate Third-Person Demonstration Proxy

Use ManiSkill to generate a successful manipulation from an external camera.

Acceptable demonstrators:

- Oracle planner.
- Scripted policy.
- Franka, if the arm is treated only as a proxy demonstrator.
- Simple hand/contact proxy such as a capsule, sphere fingertip, or kinematic contact tool.
- Object-only trajectory with contact side labels.

Record:

- External RGB or RGB-D video.
- Object masks if available.
- Object poses.
- Contact region label.
- Start state.
- End state.
- Object trajectory.

Do not expose robot joint states or privileged action trajectories to the BABYSTEPS intent inference path. Those can be stored as simulator metadata for debugging, but they must not be treated as demo input.

### 2. Convert Demo Evidence Into Intent Factors

For Stage 0, this conversion can be scripted or label-driven.

The intent extraction target is structured data, not free-form text:

```json
{
  "goal_state": "cube_at_target",
  "object_motion": "translate_right",
  "contact_region": "left_face",
  "approach_direction": "from_left",
  "constraint_region": "none",
  "embodiment_mapping": "proxy_contact_to_franka_push"
}
```

Later stages can replace scripted labels with DINO/DINOv2 grounding plus VLM symbolization.

### 3. Execute From Robot-Centric View

Reset the scene and execute with Franka using robot-centric observations:

- Wrist camera.
- Robot front camera.
- Robot-centric state view during early debugging.

The important experimental condition is:

```text
demo view != execution view
```

Stage 0 may use privileged simulator state internally for deterministic labels and success checks, but the conceptual data contract should distinguish demo evidence from robot execution evidence.

### 4. Create Controlled Semantic Failures

Failures should be intentional and labeled. Do not depend on random simulator instability.

Initial controlled failure sources:

- Block the demonstrated approach side.
- Make the demonstrated contact region unreachable.
- Add an obstacle near the target.
- Require a final orientation that a coarse goal omits.
- Change object affordance so the demonstrated contact is not robot-feasible.

Each failure should have an oracle wrong factor label for evaluation.

### 5. Build Failure Packet

Every failed attempt should produce a structured packet:

```json
{
  "chosen_intent": {
    "goal_state": "cube_at_target",
    "object_motion": "translate_right",
    "contact_region": "left_face",
    "approach_direction": "from_left",
    "constraint_region": "none",
    "embodiment_mapping": "proxy_contact_to_franka_push"
  },
  "execution_trace": {
    "reached_contact": false,
    "object_moved": false,
    "collision": true,
    "planner_failed": true,
    "grasp_slip": false
  },
  "failure_predicate": "approach_blocked"
}
```

Then the BABYSTEPS attribution target is:

```json
{
  "semantic_failure": true,
  "wrong_factor": "approach_direction",
  "freeze": ["goal_state", "object_motion"],
  "revise": ["approach_direction", "contact_region"]
}
```

### 6. Revise and Retry

The revision must preserve non-implicated factors.

Example:

```json
{
  "operator": "approach_substitution",
  "old_value": "from_left",
  "new_value": "from_right",
  "frozen_factors": ["goal_state", "object_motion", "constraint_region"]
}
```

The retry should be logged as part of the same episode.

## Minimal Task Set

Start with three Stage 0 tasks.

### Task 1: PushCube With Blocked Approach

Scenario:

```text
third-person proxy demo: push cube from left to target
robot execution: left side is blocked
first attempt: push from left fails
revision: approach/contact side changes
retry: goal stays fixed, execution succeeds
```

Factors tested:

- `object_motion`
- `contact_region`
- `approach_direction`

This is the first priority.

### Task 2: PickCube With Wrong Contact Region

Scenario:

```text
third-person proxy demo: pick object from one side
robot execution: that side is inaccessible or unstable
first attempt: selected grasp/contact fails
revision: contact region changes
retry: object identity and goal stay fixed
```

Factors tested:

- `contact_region`
- `embodiment_mapping`

### Task 3: Place or Align With Hidden Goal Refinement

Scenario:

```text
third-person proxy demo: object is placed at target with orientation
initial robot plan: only moves object to target
failure: position correct, orientation wrong
revision: goal_state is refined
retry: contact and object identity stay fixed
```

Factors tested:

- `goal_state`
- `object_motion`

## Episode Data Format

Each Stage 0 episode should be serializable as JSON:

```json
{
  "episode_id": "pushcube_blocked_approach_seed_0001",
  "stage": "stage_0",
  "task": "PushCube-v1",
  "claim_boundary": "third_person_demo_proxy_not_human_demo",
  "demo": {
    "camera": "third_person",
    "rgbd_video": "data/demos/pushcube_blocked_approach_seed_0001/demo_rgbd.mp4",
    "object_trajectory": "data/demos/pushcube_blocked_approach_seed_0001/object_trajectory.json",
    "contact_region_label": "left_face",
    "final_state": "cube_at_target",
    "demonstrator_type": "proxy_oracle"
  },
  "execution": {
    "camera": "robot_first_person",
    "robot": "Franka",
    "initial_intent": {
      "goal_state": "cube_at_target",
      "object_motion": "translate_right",
      "contact_region": "left_face",
      "approach_direction": "from_left",
      "constraint_region": "none",
      "embodiment_mapping": "proxy_contact_to_franka_push"
    },
    "success": false
  },
  "failure_packet": {
    "failure_predicate": "approach_blocked",
    "wrong_factor": "approach_direction",
    "oracle_wrong_factor": "approach_direction"
  },
  "revision": {
    "operator": "approach_substitution",
    "old_value": "from_left",
    "new_value": "from_right",
    "frozen_factors": ["goal_state", "object_motion", "constraint_region", "embodiment_mapping"]
  },
  "retry": {
    "success": true,
    "num_retries": 1
  }
}
```

## Required Metrics

Track these metrics from the beginning:

- `final_success_rate`
- `retry_success_rate`
- `num_attempts_to_success`
- `failure_type_accuracy`
- `intent_factor_attribution_accuracy`
- `unnecessary_factor_change_rate`
- `frozen_factor_preservation_rate`
- `revision_success_rate`

The most important Stage 0 comparison is:

```text
BABYSTEPS selective factor revision
vs.
full replanning
vs.
failure-agnostic retry
vs.
oracle factor revision
```

## Stage 0 Non-Goals

Do not add these until the Stage 0 data format and PushCube loop are stable:

- Real human videos.
- Real Franka execution.
- Diffusion counterfactual scoring.
- Learned failure attribution.
- VLM free-form replanning.
- Multi-task benchmark claims.
- Cross-embodiment paper claims.

## Success Criteria

Stage 0 is complete when:

- A PushCube third-person demonstration proxy is generated and stored.
- A structured initial intent is derived from the demo evidence.
- Franka executes from a robot-centric view or robot-centric state abstraction.
- A controlled semantic failure is produced and logged.
- A structured failure packet identifies the intended wrong factor.
- BABYSTEPS revises only the implicated factor.
- The retry succeeds in at least one deterministic seed.
- The episode JSON includes demo, execution, failure, revision, and retry sections.
- The writeup consistently describes the input as a third-person demonstration proxy, not a human demonstration.

## Later Stages

Stage 1:

```text
Replace proxy demos with real human or human-hand-object videos.
```

Possible sources include self-recorded demonstrations or datasets such as HOI4D/OakInk, if they fit the manipulation tasks and licensing constraints.

Stage 2:

```text
Move selected tasks to a real Franka setup.
```

Stage 3:

```text
Add learned attribution, revision ranking, and optional diffusion counterfactual scoring.
```

## Guidance for Future Agents

When implementing from this file:

1. Preserve the Stage 0 boundary line: proxy demo, not human demo.
2. Start with PushCube blocked approach before adding PickCube or orientation refinement.
3. Keep simulator privilege out of the demo-to-intent input path.
4. Use simulator privilege only for labels, success checks, and evaluation.
5. Log every attempt as structured JSON.
6. Make revision operators explicit and factor-local.
7. Reject implementations that freely regenerate all intent fields after failure.
