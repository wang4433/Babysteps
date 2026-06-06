# Related Work and Baseline Policy

> **Decision date:** 2026-06-06.
>
> This document defines the paper-facing comparison boundary for BABYSTEPS.
> `goal.md` remains authoritative for the method and data contract.

## Research Question

BABYSTEPS does not primarily ask which controller produces the best action
trajectory from demonstrations. It asks:

> After a manipulation attempt fails because task intent was misgrounded, can
> the system identify and repair the implicated intent factor while preserving
> the factors that were already correct?

The main comparison must therefore hold the execution layer fixed and compare
**failure-recovery strategies**, not unrelated low-level policy architectures.

## Main Experimental Baselines

All main-table methods receive the same initial intent, failure evidence,
skill library/controller, held-out seeds, and retry budget.

| Method | Role |
| --- | --- |
| `one_shot` | No recovery. |
| `same_intent_retry` | Failure-agnostic retry control. |
| `random_factor_revision` | Tests whether diagnosis matters. |
| `vlm_free_replan` | Live VLM regenerates the full intent; direct broad-replanning baseline. |
| `vlm_diagnosis_slot_edit` | **BABYSTEPS:** VLM selects one factor; the slot-local editor supplies the value. |
| `oracle_factor_revision` / `oracle_single_slot` | Upper bound and certification reference, never the primary competitor. |

The older `text_feedback_replan` and `full_replan_analogue` rows remain useful
as deterministic Stage-0 controls. They must be labeled **procedural
analogues**, not presented as measured implementations of a published system.

### Headline Metrics

Success alone does not test the contribution. Report:

- final and retry success;
- factor-attribution accuracy;
- frozen-factor preservation;
- unnecessary-factor-change and harmful-change rates;
- edit cardinality;
- attempts to success;
- recovery-time latency and VLM calls/token cost, compared only among methods
  that use the same execution interface;
- gap to the oracle upper bound.

The intended result is:

> BABYSTEPS matches or approaches broad replanning on recovery while changing
> fewer already-correct factors and producing an auditable diagnosis/edit.

## Direct Neighbors

### Feedback-conditioned language replanning

**Inner Monologue** uses environment feedback to update language-model planning
in closed loop. It is the clearest conceptual contrast: feedback changes the
planner context and may regenerate the plan, whereas BABYSTEPS constrains the
diagnoser to one typed factor and delegates the repair value to a slot-local
editor.

**REFLECT** summarizes robot experience for failure explanation and then guides
a language planner to correct the failure. It is the closest failure
explanation-and-correction neighbor. BABYSTEPS differs by enforcing a
single-factor intervention and measuring preservation of non-implicated task
intent.

These papers motivate the `vlm_free_replan` baseline. A procedural
`text_feedback_replan` row is not a substitute for a faithful implementation;
the live VLM full-intent condition is the defensible main comparison.

### Failure detection and recovery models

**AHA** trains a VLM to detect and reason about robotic failures. It is related
primarily to BABYSTEPS' attribution module. BABYSTEPS' distinct claim begins
after detection: map the failure to a typed latent slot and repair only that
slot.

**SAFE** studies multitask failure detection for VLA policies. It predicts
whether failure is occurring, while BABYSTEPS assumes a failed execution packet
and asks which task-intent factor should change. SAFE is therefore related
detection work, not a replacement for the slot-revision baseline.

**FailSafe** generates failure/recovery-action data and uses failure reasoning
to improve VLA recovery on ManiSkill. It is an important modern neighbor
because it also evaluates recovery in simulation. Its intervention is at the
action/VLA layer; BABYSTEPS intervenes in a structured visual-intent
representation and explicitly audits what remains unchanged.

If an external diagnosis model is added, AHA is the most natural swap for the
current InternVL diagnoser. If an action-level recovery system is added,
FailSafe is the most relevant comparison, but it requires a separate,
carefully matched training-data protocol.

## Why Diffusion Policy and ACT Are Not Main Baselines

**Diffusion Policy** and **ACT** learn observation-to-action policies from
action-labeled demonstrations. BABYSTEPS currently uses a fixed skill compiler
and studies post-failure intent revision. A raw success-rate or inference-speed
comparison would confound:

- action supervision versus RGB intent supervision;
- learned low-level control versus fixed motion primitives;
- training-data scale and hardware;
- one-shot execution quality versus failure-recovery selectivity.

Therefore:

- do not put Diffusion Policy or ACT in the main recovery table;
- do not claim BABYSTEPS is faster or more successful than them;
- cite them as adjacent action-generation/imitation-learning approaches;
- add an action-policy table only if all methods receive matched action
  demonstrations, observations, train/test splits, and retry interfaces.

The same boundary applies to generalist VLAs: they are relevant context, but
they are not automatically fair baselines for a structured recovery module.

## Fairness Rules

1. Compare recovery methods after the same first attempt.
2. Give all methods the same failure frames/packet and task context.
3. Keep the controller, skill library, reset state, seeds, and retry count fixed.
4. Allow the VLM free-replan baseline enough output tokens and validate its
   schema; parse failure must not be manufactured by an undersized budget.
5. Report both outcome and selectivity. A method that succeeds by rewriting
   unrelated correct factors has not matched BABYSTEPS' claim.
6. Keep oracle rows visually separated and labeled `upper bound`.
7. Do not call a procedural analogue a published-method implementation.

## References

- Huang et al., [Inner Monologue: Embodied Reasoning through Planning with
  Language Models](https://arxiv.org/abs/2207.05608), 2022.
- Liu et al., [REFLECT: Summarizing Robot Experiences for Failure Explanation
  and Correction](https://arxiv.org/abs/2306.15724), CoRL 2023.
- Duan et al., [AHA: A Vision-Language-Model for Detecting and Reasoning Over
  Failures in Robotic Manipulation](https://arxiv.org/abs/2410.00371), ICLR
  2025.
- Gu et al., [SAFE: Multitask Failure Detection for Vision-Language-Action
  Models](https://arxiv.org/abs/2506.09937), NeurIPS 2025.
- Lin et al., [FailSafe: Reasoning and Recovery from Failures in
  Vision-Language-Action Models](https://arxiv.org/abs/2510.01642), 2025.
- Chi et al., [Diffusion Policy: Visuomotor Policy Learning via Action
  Diffusion](https://roboticsproceedings.org/rss19/p026.html), RSS 2023.
- Zhao et al., [Learning Fine-Grained Bimanual Manipulation with Low-Cost
  Hardware](https://arxiv.org/abs/2304.13705), RSS 2023 (ACT).
