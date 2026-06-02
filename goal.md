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

This stage should validate the data contracts and the correction loop
before adding richer cross-view stress, real-world Franka execution, or
learned perception/revision models.

## Boundary Line

Stage 0 is **Franka-to-Franka**. The same Franka / Panda performs both
the demonstration and the execution:

1. **Demonstration phase.** A Franka executes the correct task on the
   desk under an oracle / scripted policy. We record this from a fixed
   **third-person** external camera (desk-front view). The demo Franka's
   joint trace is *not* exposed to the BABYSTEPS intent path — only the
   third-person video and object-centric labels are.
2. **Execution phase.** The scene is reset and the (same) Franka attempts
   the task. We observe it from a **first-person** view (wrist /
   robot-front camera). Failures, attribution, and revisions all happen
   in this first-person view.

There is no human demonstrator anywhere in Stage 0. The arm is always a
Franka; the cross-view stressor is *demo camera ≠ execution camera*, not
*demo embodiment ≠ execution embodiment*.

Use this wording:

> third-person Franka demonstration proxy
> (and "first-person Franka execution" for the retry phase)

Do not use this wording:

> human demonstration / human pinch / hand-object contact

Stage 0 tests:

- Cross-view intent transfer: third-person demo view → first-person Franka execution view.
- Failure-guided intent revision: failure updates a structured intent factor.
- Franka execution loop: intent compiles into executable Franka behavior.
- Data preparation: every episode stores enough evidence for later training and evaluation.

Stage 0 does not test:

- Full human-to-robot embodiment transfer.
- Real human video understanding.
- Real-world robot deployment.
- End-to-end VLA control.
- Generic LLM replanning after feedback.

The paper-facing claim for Stage 0 is:

> We evaluate BABYSTEPS in a controlled simulated setting where one Franka demonstrates the task from a third-person external camera and the same Franka then executes the task from its own first-person view. This isolates the failure-guided structured intent revision mechanism under a clean cross-view setup before moving to richer demonstrators in later stages.

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
  "object_motion": "translate_+x",
  "contact_region": "minus_x_face",
  "approach_direction": "from_minus_x",
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
- `embodiment_mapping`: how the demo-time Franka contact maps to an execution-time Franka skill primitive (e.g. demo grasp-and-turn → execution closed-gripper poke-turn when the handle exceeds the gripper opening).

Avoid task-specific fields such as `drawer_axis_correct`, `push_side_correct`, or `peg_depth_correct`.

## Stage 0 Data Pipeline

### 1. Generate Third-Person Franka Demonstration Proxy

Use ManiSkill to generate a successful Franka manipulation, recorded
from a fixed external (third-person) camera positioned in front of the
desk.

The demonstrator is always the Franka arm itself, driven by:

- An oracle planner, or
- A scripted policy.

No capsule / sphere fingertip / object-only / hand-proxy demonstrators
are used in Stage 0. Keeping the demonstrator embodiment fixed (Franka)
while only varying the camera view (third-person → first-person)
isolates the cross-view stressor without confounding it with
cross-embodiment transfer.

Record:

- Third-person RGB or RGB-D video of the demo Franka on the desk.
- Object masks if available.
- Object poses.
- Contact region label (where the demo Franka gripper contacted the object).
- Start state.
- End state.
- Object trajectory.

Do not expose the demo Franka's joint states or privileged action
trajectories to the BABYSTEPS intent inference path. Those can be stored
as simulator metadata for debugging, but they must not be treated as
demo input — the only demo input is the third-person video and the
object-centric labels above.

### 2. Convert Demo Evidence Into Intent Factors

For Stage 0, this conversion can be scripted or label-driven.

The intent extraction target is structured data, not free-form text:

```json
{
  "goal_state": "cube_at_target",
  "object_motion": "translate_+x",
  "contact_region": "minus_x_face",
  "approach_direction": "from_minus_x",
  "constraint_region": "none",
  "embodiment_mapping": "proxy_contact_to_franka_push"
}
```

Later stages can replace scripted labels with DINO/DINOv2 grounding plus VLM symbolization.

### 3. Execute From First-Person Franka View

Reset the scene and execute with the same Franka, but now record and
condition on its **first-person** observations:

- Wrist camera, or
- Robot-front camera.
- Robot-centric privileged state view is permitted during early
  debugging, but the conceptual data contract is that the executing
  Franka sees the world in first person.

The important experimental condition is:

```text
demo view  = third-person external desk-front camera
exec view  = first-person wrist / robot-front camera
demo view != exec view
```

Failure detection, attribution, and revision all happen in the
first-person execution view; the third-person demo is only used to
ground the *initial* intent.

Stage 0 may use privileged simulator state internally for deterministic
labels and success checks, but the conceptual data contract is
demo-evidence (third-person) vs execution-evidence (first-person), not
just "different cameras".

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
    "object_motion": "translate_+x",
    "contact_region": "minus_x_face",
    "approach_direction": "from_minus_x",
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
  "old_value": "from_minus_x",
  "new_value": "from_plus_x",
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
    "contact_region_label": "minus_x_face",
    "final_state": "cube_at_target",
    "demonstrator_type": "proxy_oracle"
  },
  "execution": {
    "camera": "robot_first_person",
    "robot": "Franka",
    "initial_intent": {
      "goal_state": "cube_at_target",
      "object_motion": "translate_+x",
      "contact_region": "minus_x_face",
      "approach_direction": "from_minus_x",
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
    "old_value": "from_minus_x",
    "new_value": "from_plus_x",
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

- Real human videos / human demonstrators of any kind.
- Cross-embodiment demonstrators (no capsules, no proxy hands, no
  non-Franka arms — Stage 0 is Franka-to-Franka only).
- Real Franka execution (sim-to-real).
- Diffusion counterfactual scoring.
- Learned failure attribution.
- VLM free-form replanning.
- Multi-task benchmark claims.
- Cross-embodiment paper claims.

## Success Criteria

Stage 0 is complete when:

- A PushCube third-person Franka demonstration proxy is generated and stored.
- A structured initial intent is derived from the demo evidence.
- The same Franka executes the task from its first-person view (or, during
  early debugging, a robot-centric state abstraction acceptable as a stand-in).
- A controlled semantic failure is produced and logged.
- A structured failure packet identifies the intended wrong factor.
- BABYSTEPS revises only the implicated factor.
- The retry succeeds in at least one deterministic seed.
- The episode JSON includes demo, execution, failure, revision, and retry sections.
- The writeup consistently describes the input as a third-person demonstration proxy, not a human demonstration.

## Later Stages

The project stays Franka-to-Franka throughout. Later stages stress the
*cross-view* and *deployment* axes, never the demonstrator-embodiment
axis.

Stage 1:

```text
Richer cross-view stress: more third-person camera placements, occlusions,
lighting / background variation, and a literal first-person sensor stream
(wrist / robot-front RGB-D) for the executing Franka — replacing the
single default render camera that Stage 0 uses for both phases.
```

Stage 2:

```text
Move selected tasks to a real Franka setup (sim-to-real). Demonstrator
and executor remain the same physical / kinematic Franka.
```

Stage 3:

```text
Add learned attribution, revision ranking, and optional diffusion
counterfactual scoring on top of the same Franka-to-Franka loop.
The learned attribution and revision-ranking components produced here
are also the supervision signal that the Stage 4 latent slot-intent
track consumes.
```

## Stage 4: Object-Centric Latent Slot-Intent Bottleneck

Stage 4 is the **learned-latent** track. It does not replace Stage 0; it sits
on top of it. The Stage-0 discrete schema is preserved, but its role changes:
it becomes the *supervision signal and certification scaffold* for the latent
representation, not the runtime intent.

### Purpose

Replace the discrete intent JSON with a per-object **latent slot-intent**
representation, while preserving the BABYSTEPS core invariant:

```text
failure -> identify implicated slot -> revise only that slot's intent latent -> preserve the rest
```

### Boundary Line

* Stage 4 stays Franka-to-Franka and sim-only. The Stage-0 cross-view
  condition (third-person demo, first-person execution) carries over unchanged.
* The latent representation is **object-centric and slot-local**. There is no
  monolithic intent vector and no whole-plan rewrite.
* $\tilde{g}_t^i$ (the corrective intent latent for slot $i$) is produced by a
  *learned* revision policy trained offline on Stage-0 (failure_packet,
  revision) pairs. Stage 4 does **not** call a VLM to freely regenerate the
  intent after failure; doing so would be a latent-space violation of the
  core selectivity principle.

### Architecture

$$o_t \xrightarrow{\text{SlotEncoder}} Z_t = \{z_t^1, \dots, z_t^K\} \quad \text{(object latents, masked to max object count } K\text{)}$$

$$\text{lang} + Z_t \xrightarrow{\text{IntentHead}} G_t = \{g_t^1, \dots, g_t^K\} \quad \text{(slot intents)}$$

$$(g_t^i, \text{failure\_packet\_vector}) \xrightarrow{\text{ReviseHead}} \tilde{g}_t^i \quad \text{(slot-local edit)}$$

$$\text{edited } G_t, Z_t \xrightarrow{\pi_\theta} a_t \quad \text{(action decoder)}$$

Architectural invariants (enforced by function signature, not by hope):

1. `ReviseHead` consumes **exactly one** slot intent and the vectorized
   failure packet, and emits **exactly one** revised slot intent.
   Whole-$G_t$ rewrites are forbidden by the interface.
2. The action decoder $\pi_\theta$ consumes the edited $G_t$, but cross-slot
   influence is bounded empirically by the selectivity certification
   below — not assumed.

### Certification Interface (the Stage-0 scaffold's job)

Every claim about the latent representation is grounded in the discrete schema:

1. **Probe recoverability.** A frozen linear probe must recover each Stage-0
   discrete factor (`goal_state`, `object_motion`, `contact_region`,
   `approach_direction`, `constraint_region`, `embodiment_mapping`) from
   $G_t$ with held-out accuracy $\ge$ 90%. An MLP probe is maintained
   purely as a capacity diagnostic.
2. **Frozen-slot preservation.** After applying $g_t^i \leftarrow \tilde{g}_t^i$,
   the $\ell_2$ drift of all other $g_t^j$ must be $\le \epsilon$. This is
   the latent analog of Stage-0's `frozen_factor_preservation_rate`.
3. **Selectivity certification.** Counterfactual rollout pairs (same $Z_t$,
   one edited slot vs. unedited) must show predicted future-slot drift on
   slots $j \neq i$ statistically indistinguishable from the natural
   simulator noise floor $\epsilon_{\text{sim}}$ evaluated across identical
   seeds (paired $t$-test, $p > \alpha$). This makes the "physically
   decoupled" claim falsifiable.

If probe recoverability fails, Stage 4 is not yet a faithful refinement of
Stage 0 and may not be reported as a latent-revision result.

### Data Dependencies

Stage 4 reuses Stage-0 episode JSONs as supervision:

* `chosen_intent` + `oracle_wrong_factor` + revision JSON $\rightarrow$
  training pairs for `ReviseHead`.
* `frozen_factors` from each revision JSON $\rightarrow$ attention masks
  enforcing the slot-local constraint during training.
* `intent_factor_attribution_accuracy` from Stage 0 is the analytic upper
  bound the learned attribution must approach.

The hand-coded `failure.py` rules become the **teacher** of the learned
Stage-4 components, not the thing they replace.

### Success Criteria

Stage 4 is complete when, on the Stage-0 task set:

* Probe recoverability of all six discrete factors from $G_t \ge$ 90%.
* Frozen-slot $\ell_2$ drift after a single-slot edit $\le \epsilon$
  (calibrated against the natural per-step drift of unedited rollouts).
* $\Delta pp$ of latent revision vs. learned-failure-agnostic retry $\ge$ 10.
* $\Delta pp$ of latent revision vs. oracle discrete revision within 5 pp
  (ensuring the latent loop does not collapse relative to the Stage-0
  baseline).
* Selectivity certification: counterfactual cross-slot drift is
  indistinguishable from the noise floor (paired test, $p > \alpha$).

All numeric thresholds (90% recoverability, $\Delta pp \ge 10$, 5 pp window,
$\epsilon$, $\alpha$) are calibrated against Stage-0 oracle variance on the
existing task set, not arbitrary absolute targets.

### Stage 4 Non-Goals

* Not testing cross-embodiment demonstrators (still Franka-to-Franka).
* Not testing real-Franka deployment (Stage 2's job).
* Not adopting a pretrained end-to-end VLA as the policy backbone.
* Not allowing `ReviseHead` to read $G_t$ as a whole. Whole-bottleneck
  rewrites are off-limits; the slot-local interface is the contribution.
* Not relying on a free-form VLM call to produce $\tilde{g}_t^i$ at inference.

## Stage 5: Vision-Grounded Latent Intent (ICLR target)

> **Status:** planned 2026-05-24. This is the paper-submission track.
> Stage 5 replaces Stage 4's handcrafted-feature bottleneck with
> vision-grounded slot intents, VLM-based attribution, and world-model
> counterfactual verification. Stage 0's discrete schema remains as the
> supervision signal and certification scaffold — it is never discarded.

### Motivation

Stage 4 proved the slot-local revision interface works end-to-end, but
its IntentHead consumes a 20-dim handcrafted feature vector (trajectory
stats + one-hot labels), and the output quantizes back to discrete
tokens via nearest-centroid lookup. The continuous bottleneck adds
nothing that the discrete schema alone does not provide. An ICLR
"latent intent" claim requires that the representation is grounded in
raw observations, not hand-engineered inputs.

### Framing: latent intent vs. the swappable diagnoser

The load-bearing claim is **latent**: the vision-grounded slot intents
$G$ *are* the latent intent, and the learned **ReviseHead editing one
slot in that latent space** is the contribution. The attribution step
("which slot failed?") is a **separate, swappable module** — rule-table →
learned attribution head → VLM — that returns *a factor name, not a
value*; it selects which slot the ReviseHead edits and never produces
$\tilde{g}^i$. The VLM is therefore a plug-in diagnoser, not part of the
latent edit, and cannot dilute the latent claim.

**Honesty boundary:** the latent path is demonstrated **end-to-end only
on PushCube** today (P1 G1 ≥ 90% there); the 5-task P2 table runs on the
discrete Stage-0 schema + failure frames, not on latent $G$. Claim
"latent" for the *representation and slot-local revision interface*
(end-to-end on PushCube), with the symbolic schema as the
supervision/certification scaffold — do not over-claim a 5-task latent
result.

### Architecture (target)

```text
Demo video frames ──→ [Frozen Vision Encoder] ──→ z_demo
                                                     │
                                                     ▼
                                           [IntentHead (learned)]
                                                     │
                                                     ▼
                                         G = {g¹, g², ..., g⁶}     (slot intents)
                                                     │
                             ┌───────────────────────┤
                             │                       │
                        [Skill Decoder]         [on failure]
                             │                       │
                             ▼                       ▼
                         action a_t       [VLM Attribution] → "which slot?"
                                                     │
                                                     ▼
                                           [ReviseHead (learned)]
                                             g^i, fp → g̃^i
                                                     │
                                                     ▼
                                           [World Model Verifier]
                                         "does this revision help?"
                                                     │
                                                     ▼
                                             revised G → retry
```

### Priority 1 — Vision Encoder Swap (critical first step)

Replace the 20-dim handcrafted `Z` with a **frozen pretrained vision
encoder** (DINOv2, R3M, or SPA) applied to third-person demo frames
rendered by ManiSkill. The IntentHead architecture stays the same; only
`z_dim` changes (e.g. 20 → 768).

```python
demo_frames: Tensor   # (T, 3, 224, 224) from env.render()
z_demo = dinov2(demo_frames).mean(dim=0)   # (768,) pooled over time
G = intent_head(z_demo)                    # (6, d_slot)
```

**Gate:** G1 probe recoverability of all six discrete factors from
the vision-grounded $G_t$ at ≥ 90% held-out accuracy. If this fails,
the vision features do not carry enough task-intent signal and the
encoder or pooling strategy must be revised before proceeding.

**What this proves:** The slot intents encode visual task structure,
not memorized one-hot labels passed through a bottleneck.

### Priority 2 — VLM Attribution (diagnosis, not replanning)

> **Status (2026-06-01):** P2 **done on all five tasks** (PushCube,
> PickCube, StackCube, TurnFaucet, CrossViewPush) with **InternVL3.5-8B**
> (BF16, A100-40GB), 50 held-out failure episodes per task (seeds 100-149).
> C1 (constrained-diagnosis + slot-local revision) **passes all three gates
> on all five tasks** (attribution ≥ rule-table · preservation ≥ C2 ·
> success within 5pp of C2). C1 attribution: PushCube 1.00, PickCube 1.00,
> StackCube 0.86 (= rule-table), TurnFaucet 1.00 (vs rule 0.50),
> CrossViewPush 1.00 (vs rule 0.00). C1 ≥ C2 on success and preservation
> everywhere — strictly better on PickCube success (**+92pp**; C2 returns
> the unchanged intent verbatim) and StackCube preservation (**+50pp**).
> Task-aware prompts lifted StackCube attribution from 0/50 → 43/50; the 7
> residual misses all land on `object_motion` (visually-ambiguous off-goal
> cubes) — the remaining open problem for VLM goal_state attribution.
> (TurnFaucet success stays low — an unreliable poke-turn skill primitive,
> an execution issue, not an attribution one.) See
> `reports/stage5/p2_vlm_attribution/` and `slurm/CLAUDE.md` §"Stage-5 P2".

Use a VLM (we use **InternVL3.5-8B**; the step is VLM-agnostic — GPT-4o /
Gemini are drop-in alternatives) for the failure attribution step.
The VLM outputs **one factor name** from the fixed set — it never
freely regenerates the entire intent.

```text
Prompt template:
  You are diagnosing a robot manipulation failure.
  The robot attempted: {initial_intent as JSON}
  Failure observation: {failure trace / first-person frames}

  Which ONE intent factor was wrong? Choose exactly one:
  [goal_state, object_motion, contact_region,
   approach_direction, constraint_region, embodiment_mapping]

  Output ONLY the factor name.
```

**Comparison:** VLM-constrained-attribution + slot-local revision
vs. VLM free-form replanning (the VLM regenerates the entire intent
JSON). The paper's punchline: *VLMs are good at diagnosing failures
but wasteful at fixing them; give the VLM the diagnosis job, give a
learned slot-local editor the repair job.*

**Core invariant preserved:** The VLM is used for *diagnosis only*.
It selects which slot to edit; it does NOT produce the revised intent.
The ReviseHead (learned, slot-local) produces $\tilde{g}_t^i$. This
is not "ask a VLM/LLM to replan the whole thing after failure."

### Priority 3 — World Model Counterfactual Verification

Train a latent dynamics model on ManiSkill rollouts:

$$(z_t, a_t) \xrightarrow{f_\phi} z_{t+1}, \quad z_t \xrightarrow{r_\phi} \hat{r}_t$$

This serves three purposes:

1. **G3 selectivity certification.** Counterfactual rollout pairs
   (same $Z_t$, edited slot $i$ vs. unedited) must show predicted
   future-slot drift on $j \neq i$ indistinguishable from the
   simulator noise floor. The world model provides the forward
   predictions; G3 becomes a real test, not a mechanical bit-identity.
2. **Revision ranking.** When multiple slots are plausible edit
   targets, simulate both revisions in imagination and pick the one
   with higher predicted success.
3. **Paper narrative.** "We use a world model to verify that
   single-factor revision is physically sufficient — the
   counterfactual rollout confirms that editing one slot does not
   require compensating edits to other slots."

### Priority 4 — Learned Action Decoder (optional)

Replace the fixed skill compiler with a small policy conditioned on G.
This closes the loop (everything is learned) but is the riskiest
change. Deferrable; the paper is strong with Priorities 1–3 and the
existing skill compilers.

### Stage 5 Success Criteria

* Priority 1 gate: G1 probe ≥ 90% on vision-grounded $G_t$ (not
  handcrafted features).
* Priority 2 gate: VLM attribution accuracy ≥ rule-table accuracy;
  VLM-diagnosis + slot-local revision ≥ VLM free-form replan on
  frozen-factor preservation with comparable recovery rate.
* Priority 3 gate: G3 counterfactual selectivity passes with
  world-model forward predictions (not mechanical bit-identity).
* $\Delta pp$ of vision-grounded latent revision vs. same_intent_retry
  ≥ 10; vs. oracle discrete revision within 5 pp.

### Stage 5 Non-Goals

* Not replacing the Stage-0 discrete schema (it remains as supervision
  + certification scaffold).
* Not training the vision encoder end-to-end (frozen pretrained only
  in v1; fine-tuning is a follow-on if frozen features pass G1).
* Not adding cross-embodiment demonstrators (still Franka-to-Franka).
* Not allowing the VLM to produce the revised intent value (only the
  factor name — the ReviseHead produces the edit).

### Immediate Next Step — P1 Implementation Sequence

> What to build right now. This is the P1 critical path. Do these in
> order; each step unlocks the next.

**Step 1. `babysteps/stage4/vision_features.py` — DINOv2 extraction.**

Create a new module that takes demo RGB frames and returns a feature
vector. Must obey the Stage-4 firewall (no label-side fields):

```python
def extract_vision_features(
    demo_frames: list[np.ndarray],   # T × (H, W, 3) uint8 from env.render()
    encoder: str = "dinov2_vitb14",
    pool: str = "cls_mean",
    device: str = "cuda",
) -> np.ndarray:
    """Return (d_encoder,) float32. DINOv2 ViT-B/14 → CLS → mean-pool over T."""
```

Input: the RGB frames already captured by `render_frame(env)` in
`babysteps/render/common.py`. Output: `(768,)` for DINOv2 ViT-B/14.

Requires GPU for DINOv2 inference but is fast (~2 min for 50 episodes).
Write a sim-free unit test with a random tensor to verify shapes.

**Step 2. Extend the collection pipeline to save demo frames.**

The existing `run_episode()` in `babysteps/episode.py` calls
`generate_proxy_demo()` which internally calls `env_runner.run()`.
The env_runner already steps the env, but does NOT capture RGB frames
(it only reads `obs["extra"]` for poses).

Two options:
- **Option A (simpler):** Add a separate `collect_demo_frames()` that
  re-runs the demo execution with `env.render()` per step. Called
  after `run_episode()` for the same seed, saves frames as `.npz`.
- **Option B (cleaner):** Extend `EnvRunner.run()` to optionally
  capture `env.render()` per step and return frames alongside the
  `AttemptResult`. Requires touching the runner interface.

Prefer Option A for speed — it avoids changing the tested runner
interface. The demo is deterministic (same seed, same oracle), so
re-running it yields identical frames.

Output path: `datasets/stage5/varied_intent/<task>/frames/seed_NNNN.npz`.

**Step 3. Extract and cache DINOv2 features for each episode.**

Run Step 1's extractor on each seed's saved frames. Cache the output
as `seed_NNNN_dinov2.npy` alongside the `.npz`. This is a one-off
GPU job (~2 min per task).

**Step 4. Train IntentHead on vision features + run G1 probe.**

Modify the existing `scripts/stage4_m2a_train_pack.py` (or write a
new `scripts/stage5_p1_vision_probe.py`) to:
1. Load cached DINOv2 features instead of `features.py` output.
2. Instantiate `IntentHead(z_dim=768, d_slot=32, hidden=256)`.
3. Train with the same per-slot CE supervision as M2a.
4. Run the same nested-CV G1 probe as M2a A1.
5. Write the G1 report to `reports/stage5/p1_vision_g1/`.

**Pass:** all non-trivially-constant (task, factor) cells ≥ 90%.

**Step 5. Retrain ReviseHead on vision-grounded G + run G4/G5.**

Same as M2a A2/A3 but on vision-grounded slots. ReviseHead L2 loss
to centroids in the new slot space. Then sim rollout eval on held-out
seeds (GPU job). Gate: G4 Δpp ≥ 10, G5 within 5 pp of oracle.

**Data prerequisite:** The varied-intent collection currently has
PushCube (20 episodes) and StackCube (40 episodes). For the vision
probe to be meaningful, we need:
- At least 50 seeds per task with saved demo frames.
- PickCube varied cut (not yet collected).
- Intent diversity across seeds (the M1.5 varied cut partially
  addresses this for PushCube and StackCube; PickCube needs a
  parallel varied-cut design).

Collect new data with frame saving as part of Step 2.

## Guidance for Future Agents

When implementing from this file:

1. Preserve the Stage 0 boundary: third-person Franka demo proxy, not human demo.
2. Preserve the Stage 0 invariant: demo and execution are the same Franka.
   No capsules, sphere fingertips, hand proxies, or non-Franka arms.
3. Preserve the cross-view condition: demo camera (third-person desk-front)
   ≠ execution camera (first-person wrist / robot-front).
4. Start with PushCube blocked approach before adding PickCube or orientation refinement.
5. Keep simulator privilege out of the demo-to-intent input path.
6. Use simulator privilege only for labels, success checks, and evaluation.
7. Log every attempt as structured JSON.
8. Make revision operators explicit and factor-local.
9. Reject implementations that freely regenerate all intent fields after failure.
