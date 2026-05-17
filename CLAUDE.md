# Stage-0 status (added 2026-05-15)

> **Read `goal.md` first.** Stage-0 reframes the intent schema to be
> object-centric (`goal_state / object_motion / contact_region /
> approach_direction / constraint_region / embodiment_mapping`) instead of
> the older `target / goal / interaction.{mode,contact_site} /
> reference_frame / constraints` shape described below. If the two
> disagree, **`goal.md` wins for Stage 0**.

The Stage-0 PushCube blocked-approach data-prep loop is implemented:

For GPU tasks (Vulkan + NVIDIA ICD), connect a compute node and render
both tasks' three-phase MP4s. Same script, different --task value:

```bash
# PushCube (Sub-project A — approach_blocked)
srun --account=rpaleja --partition=a100-40gb --gres=gpu:1 --mem=115G --time=00:20:00 bash -lc '
  cd /scratch/gilbreth/wang4433/babysteps &&
  source /apps/external/conda/2025.09/etc/profile.d/conda.sh &&
  conda activate handover &&
  OUT_DIR=/scratch/gilbreth/wang4433/render_pushcube &&
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
  python scripts/render_stage0_maniskill.py \
    --task PushCube-v1 \
    --out_dir "$OUT_DIR" \
    --n_episodes 2 \
    --seed_start 0 &&
  ls -lh "$OUT_DIR/videos_maniskill"
'

# PickCube (Sub-project B — grasp_slip; closes B's acceptance gate item 4)
srun --account=rpaleja --partition=a100-40gb --gres=gpu:1 --mem=115G --time=00:20:00 bash -lc '
  cd /scratch/gilbreth/wang4433/babysteps &&
  source /apps/external/conda/2025.09/etc/profile.d/conda.sh &&
  conda activate handover &&
  OUT_DIR=/scratch/gilbreth/wang4433/render_pickcube &&
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
  python scripts/render_stage0_maniskill.py \
    --task PickCube-v1 \
    --out_dir "$OUT_DIR" \
    --n_episodes 2 \
    --seed_start 0 &&
  ls -lh "$OUT_DIR/videos_maniskill"
'

# StackCube (Sub-project C — goal under-specification; closes C's acceptance gate item 4)
srun --account=rpaleja --partition=a100-40gb --gres=gpu:1 --mem=115G --time=00:20:00 bash -lc '
  cd /scratch/gilbreth/wang4433/babysteps &&
  source /apps/external/conda/2025.09/etc/profile.d/conda.sh &&
  conda activate handover &&
  OUT_DIR=/scratch/gilbreth/wang4433/render_stackcube &&
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
  python scripts/render_stage0_maniskill.py \
    --task StackCube-v1 \
    --out_dir "$OUT_DIR" \
    --n_episodes 2 \
    --seed_start 0 &&
  ls -lh "$OUT_DIR/videos_maniskill"
'
```

Expected output per task: 2 episodes × 3 MP4s = 6 files in
`videos_maniskill/`, named
`<task_prefix>_seed_NNNN__{1_demo,2_attempt_blocked,3_retry}.mp4`.

- Design: `docs/superpowers/specs/2026-05-15-stage0-pushcube-blocked-design.md`
- Plan:   `docs/superpowers/plans/2026-05-15-stage0-pushcube-blocked-plan.md`
- Code:   `babysteps/` (pure modules) + `babysteps/envs/{pushcube,pickcube,stackcube}_runner.py` (sim adapters),
          `babysteps/envs/task_registry.py` (--task dispatch),
          `babysteps/render/{pushcube,pickcube,stackcube}.py` (per-task MP4 flows)
- Scripts: `scripts/{stage0_collect,render_stage0_maniskill}.py` accept `--task {PickCube-v1,PushCube-v1,StackCube-v1}`.
          `scripts/stage0_summarize.py` derives the task from the input JSONL (no flag).
          `scripts/smoke_pushcube.py` remains a PushCube-only loadability check.
- Tests:  221 sim-free unit tests in `tests/` (PushCube + PickCube + StackCube, snapshot-stable across all three)

Quickstart: `README.md`. The acceptance gate is `delta_pp >= 10` between
revised-retry success rate and initial-attempt success rate, identical to
Pick4Pass M-BABY-1.

The older sections below describe a fuller (post-Stage-0) system. They are
kept as reference — they motivated the modular boundaries — but their
schema and module names do not match the current Stage-0 implementation.

---

Below is an opinionated cold-start handover document. It assumes the current direction: **Franka-only manipulation, ManiSkill simulation, VLM-generated structured intent, failure-guided intent revision, optional diffusion counterfactual diagnosis**. ManiSkill is a reasonable default because it is a SAPIEN-based manipulation framework with GPU-parallelized simulation/rendering and built-in tasks; Qwen2.5-VL-7B is a reasonable first VLM because it supports visual understanding, object localization, and long-video comprehension; Diffusion Policy is related but should be treated as an action-generation baseline rather than the core BABYSTEPS contribution. ([ManiSkill][1])

---

# BABYSTEPS Cold-Start Handover

## 1. Project Goal

The goal of this project is to build a **failure-guided intent revision framework** for Franka robot manipulation.

The core idea is:

```text
ambiguous demo/instruction
→ structured intent hypothesis
→ action execution
→ structured failure signal
→ revise only the wrong intent factor
→ re-execute
```

The key claim is that robot failure should not be treated only as an action-level recovery signal. A failure can reveal that the robot misunderstood part of the task intent.

For example:

```text
Instruction/demo:
"move the red block left of the bowl"

Initial interpretation:
target = red_block
goal = left_of(red_block, bowl)
interaction = push
reference_frame = camera_frame

Failure:
block moved in the wrong direction

Revision:
reference_frame = object_centric

Everything else stays fixed.
```

The novelty is **factor-level intent revision**, not generic retry.

This follows the current BABYSTEPS abstract: the method factorizes task intent, attributes failures to likely intent factors, and selectively revises those factors before retrying. 

---

## 2. What This Project Is Not

This project is **not**:

```text
- cross-embodiment imitation
- direct language-to-torque control
- end-to-end VLA training
- reinforcement learning
- generic retry after failure
- asking a VLM/LLM to simply “try again”
```

The robot is always **Franka/Panda**.

The project should not depend on learning a full robot policy from scratch. The low-level controller can be a fixed skill primitive, motion planner, or Cartesian controller.

---

## 3. Core Research Question

Main question:

> Can execution failure diagnose which part of a structured task intent was wrong?

More concretely:

```text
Given:
- current intent hypothesis z
- executed action/skill a
- observed failure f

Infer:
- which factor of z caused the failure
- how to revise only that factor
```

The important target is not only final success rate.

We must measure:

```text
- Did the system revise the correct intent factor?
- Did it preserve the correct factors?
- Did the revised intent improve re-execution?
```

---

## 4. Main Difference From Related Work

### Inner Monologue

Inner Monologue uses environment feedback such as success detection, object recognition, scene description, and human interaction as language context for an LLM planner. It improves closed-loop planning by feeding feedback back into the prompt. ([Inner Monologue][2])

BABYSTEPS is different:

```text
Inner Monologue:
failure → update LLM prompt/context → replan

BABYSTEPS:
failure → structured failure signal → identify wrong intent factor
→ revise that factor only → re-execute
```

Reviewer-facing distinction:

> Inner Monologue updates the planner’s context. BABYSTEPS updates the robot’s belief about what the task meant.

### Diffusion Policy

Diffusion Policy uses diffusion to generate robot behavior/action trajectories directly. ([Diffusion Policy][3])

BABYSTEPS should not use diffusion mainly as an action generator.

Preferred use:

```text
candidate revised intent
→ imagined/counterfactual rollout
→ score whether this edit explains/fixes the failure
```

Reviewer-facing distinction:

> Diffusion Policy generates actions. BABYSTEPS optionally uses diffusion-generated counterfactuals to diagnose which structured intent factor should be revised.

---

## 5. System Boundary

The system has five modules:

```text
1. Intent proposer
2. Intent compiler
3. Skill executor
4. Failure detector
5. Intent reviser
```

Full loop:

```text
demo/instruction + scene
→ VLM proposes top-k intent JSONs
→ compiler maps intent JSON to Franka skill
→ skill executes in ManiSkill
→ detector emits failure JSON
→ reviser updates one intent factor
→ retry
```

---

## 6. Structured Intent Representation

The intent is a latent task hypothesis.

In code, we store it as JSON.

The JSON is not the research contribution. It is just the serialization of a typed latent variable.

Minimal schema:

```json
{
  "target": "red_block",
  "goal": {
    "predicate": "left_of",
    "args": ["red_block", "bowl"]
  },
  "interaction": {
    "mode": "push",
    "contact_site": "right_side"
  },
  "reference_frame": "object_centric",
  "constraints": [
    "do_not_move(bowl)"
  ]
}
```

Recommended intent factors:

| Factor                     | Meaning                      | Example                                         |
| -------------------------- | ---------------------------- | ----------------------------------------------- |
| `target`                   | object to manipulate         | `red_block`                                     |
| `goal`                     | desired final predicate      | `inside(red_block, bowl)`                       |
| `interaction.mode`         | physical interaction type    | `push`, `grasp`, `pull`                         |
| `interaction.contact_site` | where contact happens        | `right_side`                                    |
| `reference_frame`          | spatial interpretation frame | `robot_frame`, `object_centric`, `camera_frame` |
| `constraints`              | what should be preserved     | `do_not_move(bowl)`                             |

Do not create task-specific fields like:

```text
drawer_axis_correct
push_side_correct
peg_depth_correct
```

Those make the method look hand-designed per task.

---

## 7. Structured Failure Representation

Failure should also be structured.

Example:

```json
{
  "failure_type": "direction_error",
  "goal_satisfied": false,
  "contact_established": true,
  "object_moved": true,
  "wrong_object_moved": false,
  "constraint_violated": false
}
```

Recommended failure types:

| Failure type           | Likely wrong intent factor                   |
| ---------------------- | -------------------------------------------- |
| `direction_error`      | `reference_frame` or `goal`                  |
| `contact_failure`      | `interaction.mode` or `contact_site`         |
| `wrong_object_moved`   | `target`                                     |
| `goal_not_satisfied`   | `goal`                                       |
| `constraint_violation` | `constraints`                                |
| `planning_failure`     | `interaction.mode`, `goal`, or `constraints` |

Initial implementation can use rule-based attribution. Later, diffusion counterfactuals can score candidate edits.

---

## 8. Intent-to-Action Mapping

The intent JSON does not directly become torque.

It maps through a skill compiler:

```text
intent JSON
→ parameterized skill
→ controller command
→ Franka motion
```

Example:

```json
{
  "target": "red_block",
  "goal": {
    "predicate": "left_of",
    "args": ["red_block", "bowl"]
  },
  "interaction": {
    "mode": "push",
    "contact_site": "right_side"
  },
  "reference_frame": "object_centric"
}
```

Compiles to:

```python
direction = resolve_direction(
    predicate="left_of",
    frame="object_centric",
    anchor="bowl"
)

contact_pose = compute_push_pose(
    target="red_block",
    contact_site="right_side",
    direction=direction
)

move_to(contact_pose)

push_until(
    target="red_block",
    direction=direction,
    stop_condition=left_of("red_block", "bowl")
)
```

The compiler is a critical interface.

It should be explicit and deterministic at first.

---

## 9. VLM Choice

Default VLM:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

Use it frozen, few-shot.

Input:

```text
demo keyframes
+ scene graph
+ allowed object list
+ JSON schema
+ examples
```

Output:

```text
top-k candidate intent JSONs
```

Do not fine-tune first.

Fine-tuning becomes relevant only if intent extraction is clearly the bottleneck.

Backup models:

```text
OpenGVLab/InternVL3-8B
OpenGVLab/InternVL2_5-8B
LLaVA-OneVision-Qwen2-7B
```

Qwen2.5-VL is a strong default because its technical report emphasizes visual recognition, object localization, document parsing, and long-video comprehension. ([arXiv][4])

---

## 10. Simulator Choice

Use:

```text
ManiSkill 3
```

Why:

```text
- Franka/Panda manipulation support
- built-in tabletop manipulation tasks
- object states and success predicates
- RGB-D and segmentation rendering
- fast rollout generation
- suitable for failure labeling
```

ManiSkill provides built-in rigid-body tasks and a standard Gym/Gymnasium interface; it also supports GPU/CPU simulation and fast parallelized rendering. ([ManiSkill][5])

Initial tasks:

```text
PushCube-v1
PullCube-v1
PickCube-v1
StackCube-v1
PegInsertionSide-v1
OpenCabinetDrawer-v1
OpenCabinetDoor-v1
```

First minimal set:

```text
PushCube-v1
PullCube-v1
PickCube-v1
OpenCabinetDrawer-v1
```

---

## 11. Baselines

Required baselines:

```text
1. One-shot execution
   Execute the first inferred intent once.

2. Failure-agnostic retry
   Retry with alternative low-level parameters without diagnosing intent.

3. Full replanning
   Re-query VLM and allow all intent fields to change.

4. Inner-Monologue-style feedback replanning
   Convert failure to text and ask VLM/LLM for a new plan.

5. BABYSTEPS without diffusion
   Rule-based failure-to-factor revision.

6. BABYSTEPS with diffusion counterfactuals
   Score candidate intent edits using imagined rollouts.

7. Oracle intent
   Ground-truth intent factor.
```

The most important comparison:

```text
BABYSTEPS vs full replanning
```

Reason:

```text
Full replanning can accidentally change correct fields.
BABYSTEPS should revise only the implicated field.
```

---

## 12. Metrics

Primary metrics:

```text
final_success_rate
retry_success_rate
num_attempts_to_success
correct_factor_revision_accuracy
unnecessary_factor_change_rate
```

Failure diagnosis metrics:

```text
failure_type_accuracy
intent_factor_attribution_accuracy
candidate_edit_ranking_accuracy
```

Ablation metrics:

```text
with/without structured intent
with/without selective revision
with/without diffusion scoring
with/without VLM re-query
```

---

## 13. Proposed File Structure

```text
babysteps/
├── README.md
├── PROJECT_HANDOVER.md
├── configs/
│   ├── default.yaml
│   ├── maniskill_pushcube.yaml
│   ├── maniskill_pickcube.yaml
│   └── maniskill_drawer.yaml
│
├── babysteps/
│   ├── __init__.py
│   │
│   ├── schemas/
│   │   ├── intent_schema.py
│   │   ├── failure_schema.py
│   │   └── scene_schema.py
│   │
│   ├── vlm/
│   │   ├── intent_proposer.py
│   │   ├── prompts.py
│   │   ├── qwen_vl_client.py
│   │   └── json_parser.py
│   │
│   ├── envs/
│   │   ├── maniskill_wrapper.py
│   │   ├── task_registry.py
│   │   └── scene_graph.py
│   │
│   ├── skills/
│   │   ├── base_skill.py
│   │   ├── push.py
│   │   ├── pull.py
│   │   ├── pick_place.py
│   │   ├── drawer.py
│   │   └── skill_compiler.py
│   │
│   ├── execution/
│   │   ├── controller.py
│   │   ├── rollout.py
│   │   └── success_checker.py
│   │
│   ├── failure/
│   │   ├── failure_detector.py
│   │   ├── failure_attributor.py
│   │   └── failure_rules.py
│   │
│   ├── revision/
│   │   ├── candidate_edits.py
│   │   ├── revise_intent.py
│   │   ├── edit_scorer.py
│   │   └── diffusion_counterfactual.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── baselines.py
│   │   └── logger.py
│   │
│   └── utils/
│       ├── io.py
│       ├── geometry.py
│       ├── transforms.py
│       └── visualization.py
│
├── scripts/
│   ├── run_episode.py
│   ├── collect_vlm_prompts.py
│   ├── evaluate_baselines.py
│   ├── evaluate_babysteps.py
│   └── render_rollouts.py
│
├── data/
│   ├── demos/
│   ├── vlm_prompts/
│   ├── rollouts/
│   ├── failures/
│   └── results/
│
├── notebooks/
│   ├── inspect_intents.ipynb
│   ├── inspect_failures.ipynb
│   └── plot_results.ipynb
│
└── paper/
    ├── main.tex
    ├── figures/
    └── tables/
```

---

## 14. Key Interfaces

### Intent proposer

```python
def propose_intents(demo_frames, scene_graph, schema, k=5):
    """
    Returns top-k structured intent hypotheses.
    """
    return list_of_intent_jsons
```

### Skill compiler

```python
def compile_intent_to_skill(intent, scene_graph):
    """
    Converts structured intent into executable skill.
    """
    return skill
```

### Skill executor

```python
def execute_skill(skill, env):
    """
    Executes skill in ManiSkill and returns rollout data.
    """
    return rollout
```

### Failure detector

```python
def detect_failure(rollout, intent, scene_graph):
    """
    Converts rollout into structured failure signature.
    """
    return failure_json
```

### Intent reviser

```python
def revise_intent(intent, failure, candidate_pool):
    """
    Revises only the implicated factor.
    """
    return revised_intent
```

### Diffusion counterfactual scorer

```python
def score_candidate_edits(intent, failure, candidate_edits, scene_graph):
    """
    Imagines outcomes under candidate edits and ranks them.
    """
    return ranked_edits
```

---

## 15. First Milestone

Goal:

```text
Make BABYSTEPS work without diffusion on PushCube-v1.
```

Minimum demo:

```text
1. VLM proposes 3 candidate intents.
2. System executes one push intent.
3. Failure detector detects direction error.
4. Failure attribution says reference_frame is wrong.
5. Reviser changes reference_frame only.
6. Re-execution succeeds.
```

Do not start with drawer opening or diffusion.

Start with controlled pushing.

---

## 16. Second Milestone

Add multiple failure types:

```text
PushCube-v1:
- reference_frame error
- contact_site error

PickCube-v1:
- interaction.mode error
- goal_relation error

OpenCabinetDrawer-v1:
- contact_site error
- interaction.mode error
- constraint/articulation error
```

---

## 17. Third Milestone

Add diffusion counterfactual scoring.

Initial role:

```text
candidate edits → imagined rollout → edit ranking
```

Do not use diffusion as the controller.

The diffusion module should answer:

```text
Which intent edit is most likely to turn failure into success?
```

Not:

```text
What action should the robot do next?
```

---

## 18. Paper Claim

Core claim:

> Robot execution failure can be used as evidence to revise structured task intent.

Precise novelty:

> BABYSTEPS represents task interpretation as structured intent factors and uses failure evidence, optionally supported by diffusion-generated counterfactual rollouts, to revise only the factor likely responsible for failure.

Reviewer-facing one-liner:

> Inner Monologue replans after feedback; BABYSTEPS diagnoses and edits the latent task intent that made the previous plan fail.

---

## 19. Risks

### Risk 1: Looks like prompt engineering

Mitigation:

```text
VLM is only an intent proposer.
Main result is failure-guided factor revision.
Report factor-revision accuracy.
```

### Risk 2: Looks like generic retry

Mitigation:

```text
Compare against failure-agnostic retry and full replanning.
Show BABYSTEPS changes fewer correct factors.
```

### Risk 3: Looks task-specific

Mitigation:

```text
Use same intent schema across PushCube, PickCube, PullCube, Drawer.
No task-specific diagnostic fields.
```

### Risk 4: Diffusion feels unnecessary

Mitigation:

```text
Make diffusion an optional scorer.
Show whether it improves edit ranking or factor diagnosis.
```

---

## 20. Immediate To-Do List

```text
[ ] Set up ManiSkill 3.
[ ] Run PushCube-v1 with Panda/Franka.
[ ] Implement Intent JSON schema.
[ ] Implement Failure JSON schema.
[ ] Implement Push skill compiler.
[ ] Implement rule-based failure attribution.
[ ] Implement VLM prompt for top-k intents.
[ ] Run first BABYSTEPS loop on PushCube-v1.
[ ] Add baselines: one-shot, random retry, full re-query.
[ ] Log factor-level revision metrics.
```

---

## 21. Working Definition

BABYSTEPS is:

```text
A test-time structured intent revision framework for robot manipulation.
```

It is not:

```text
a VLA model
a diffusion policy
an RL algorithm
a generic LLM retry loop
```

The implementation uses JSON.

The paper should describe it as:

```text
a structured latent intent hypothesis
```

[1]: https://maniskill.readthedocs.io/?utm_source=chatgpt.com "ManiSkill — ManiSkill 3.0.0b22 documentation"
[2]: https://innermonologue.github.io/?utm_source=chatgpt.com "Inner Monologue: Embodied Reasoning through Planning ..."
[3]: https://diffusion-policy.cs.columbia.edu/?utm_source=chatgpt.com "Diffusion Policy"
[4]: https://arxiv.org/abs/2502.13923?utm_source=chatgpt.com "[2502.13923] Qwen2.5-VL Technical Report"
[5]: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html?utm_source=chatgpt.com "Quickstart — ManiSkill 3.0.0b22 documentation - Read the Docs"
