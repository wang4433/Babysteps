# TurnFaucet oracle-controller exploration (Stage-5 Phase-3 follow-up)

Goal (user request): build a real ground-truth controller that turns the
faucet to the env's 90%-of-range target, to (a) prove the task is solvable by
a Franka and (b) lift execution success above the 4% noise floor.

Probe seeds 100-104 (1 tilted, 2 vertical-axis, 2 horizontal-axis handles).
Feasibility budget 300-400 steps (production caps at 200).

## What the env forces (mani_skill turn_faucet.py)

- success = `current_angle > qmin + 0.9*(qmax-qmin)` — turn through **90% of the
  full joint range** (needed_delta ≈ 1.2–2.7 rad).
- dense reward **commented out**; **no MP solution; not in demo manifest** →
  there is no shipped ground-truth demo for this task.
- joint: friction 0.1, **stiffness 0** (no spring-back), damping 2.0.
- `target_link_pos` = switch-link **cmass**, which sits ON the joint axis.

## Controllers tried

| controller | seed 100 (horiz) | 101 (horiz) | 102 (vert) | 103 (tilt) | 104 (vert) |
| --- | --- | --- | --- | --- | --- |
| arc small-lead (`_diag_tf_arc` v2) | 0% | 0% | 0% | 29% | 0% |
| arc strong-push (v3) | 0% | 4% | 0% | 13% | 0% |
| **grasp-centre + wrist-spin** (`_diag_tf_grasp`) | 0% | 19%* | **50%** | 15%* | **27%** |

(* = noisy/non-monotonic; the vertical-axis 102/104 traces are monotonic.)

0/5 reached the 90% success target in any controller.

## Findings

1. **The cmass sits on the joint axis (lever arm ≈ 0).** A single-point poke at
   the reported handle position applies ~zero torque. Using the handle OBB
   far-face (max lever arm) gives non-zero torque but contact is fragile.
2. **Push-based turning loses contact.** A strong tangential push (v3)
   overshoots and parks the TCP past the stalled handle (trace freezes); a
   gentle lead (v2) keeps contact but is too weak and stalls (~29% on the one
   large handle). Pushing cannot maintain contact through a large arc.
3. **Grasp + wrist-spin genuinely turns vertical-axis knobs** — monotonic
   accumulation (102: 0→0.81 rad; 104: 0→0.37, climbing every 50 steps). This
   PROVES the turn is feasible for that handle subclass. It caps below target
   because the Panda wrist (joint 7, ±2.9 rad) runs out of continuous rotation,
   and it is slow (grip transfers only a fraction of the commanded spin).
4. **Spinning ee-z does nothing for horizontal-axis handles** (100: 0%). Those
   need rotation about a horizontal axis = a large EE *arc* motion (full 6-DOF
   pose control about an arbitrary world axis), not a wrist spin.
5. **The handles ARE graspable** (OBB thin dims 1–4 cm ≤ the 8 cm gripper).

## Implication for Sub-project D's premise

Sub-project D's story is *"demo grasp-turn is infeasible (jaws can't close on
the thick handle) → revise embodiment_mapping to poke-turn → poke succeeds."*
The data complicates this:
- Jaws CAN close (handles are thin enough) — so the stated infeasibility reason
  does not hold physically.
- grasp+spin is the strategy that actually makes progress; **poke (the intended
  recovery) is the unreliable one.**
- Neither completes the 90%-range turn reliably within budget.

So the "grasp infeasible → poke recovery" embodiment_substitution does not hold
up at the physics level on ManiSkill3 TurnFaucet-v1.

## Options (decision required)

- **A. General 6-DOF grasp + arc-rotate** about the arbitrary joint axis
  (position + orientation follow R(axis,φ)); handles all axis orientations;
  re-grasp when the wrist saturates. ~2–4 days, uncertain near the arm's
  reach limits. Would be a real oracle but **requires redesigning the D
  narrative** (grasp is the feasible strategy, not the failed one).
- **B. Vertical-axis subset.** Refine grasp+spin (higher rate, re-grasp on
  wrist limit, longer budget) and restrict TurnFaucet to vertical-axis faucet
  models. Smaller, cleaner, but a curated subset of the asset population.
- **C. Re-found Sub-project D on a physically-true failure mode**, e.g. demo =
  continuous-wrist grasp-turn; Franka grasp-turn stalls at the wrist joint
  limit → revise embodiment_mapping to a re-grasp/regrip turn. Honest and
  embodiment-grounded; needs adapter + narrative work.
- **D. Attribution-only / appendix.** Keep TurnFaucet for its strong VLM
  attribution result (1.000 vs rule 0.500) with an execution caveat, or move
  it to the appendix (4-task main table). No further skill investment.

Scripts: `scripts/_diag_tf_arc.py`, `scripts/_diag_tf_grasp.py`
(GPU-only scratch). Jobs: 10892296/10892362/10892500/10892519 (arc),
10892576 (grasp+spin).

---

## Re-grasp ratchet oracle (the working controller)

Per user direction (2026-05-29): center-grasp + spin about the TRUE joint axis
(`action[3:6]=sign*axis_world*SPIN`, root frame) + RE-GRASP RATCHET — spin to
the wrist limit, release, rewind the wrist, regrip, spin again. stiffness=0 so
each partial turn persists. `scripts/_diag_tf_regrasp.py`.

**Result on seeds 100-149 (50, held-out):**

| version | success | mean progress | ≥50% progress | vert | tilt | horiz |
| --- | --- | --- | --- | --- | --- | --- |
| baseline poke | **4%** (2/50) | 0.18 | — | — | — | — |
| ratchet v1 (600 steps, 6 cyc) | **28%** (14/50) | 0.537 | 26/50 | 9/28 | 2/2 | 3/20 |
| ratchet v2 (1200 steps, 12 cyc, lifted rewind) | **26%** (13/50) | 0.623 | 28/50 | 9/28 | 2/2 | 2/20 |

(jobs 10895840 probe, 10895890 v1, 10896138 v2)

**This is a 7× lift over baseline and PROVES the task is solvable by a real
Franka controller** — handles of every axis orientation reached the 90%-range
target on at least some seeds; mean progress 0.62; ~56% reach ≥50%.

**But it plateaus at ~27% overall, structurally:**
- Per-axis success is IDENTICAL across v1/v2 (vert 9/28=**32%**, horiz 2-3/20=
  10-15%, tilt 2/2). More budget/cycles did not move it — every non-success
  seed already uses the full budget; per-cycle gains diminish near target.
- **Horizontal-axis faucets (20/50 = 40% of the population)** are the hard
  ceiling: rotating about a horizontal axis from a top-down grasp wedges the
  arm. They cap overall success.
- **Grip-loss (~18/50):** the handle reaches a peak then slips back during a
  re-grasp transition before `current>target` latches (e.g. seed 136 hit the
  target 1.41 exactly then fell to 1.22). The lifted-rewind fix did not remove
  it; it is a stochastic ~15-20% leak.

**Vertical-axis subset = 9/28 = 32%, which clears the original ≥30% gate.**

## Where this leaves Sub-project D

- The chosen narrative (continuous grasp-turn → wrist-limit → re-grasp
  embodiment adaptation) is **validated as the real mechanism**: re-grasping
  is exactly what lets the Franka exceed its wrist range, and it works.
- A ground-truth oracle now exists (proves solvability; can source demos).
- Execution success is moderate (~27% all / 32% vertical), not high — limited
  by horizontal-axis geometry + re-grasp grip-loss.

## Options (decision)

- **Vertical-axis subset** (the pre-specified fallback): restrict TurnFaucet to
  vertical-axis faucet models → clean 32% (> gate), strongest where the
  re-grasp story is cleanest. Curated asset subset (documented).
- **Keep full population at ~27%** with the honest execution caveat; productionize
  the ratchet as the D recovery skill and rewrite the narrative.
- **Attack grip-loss + horizontal axes** (more engineering, uncertain) to lift
  overall past 30%.
- **Attribution-only / appendix** (original fallback) if execution at this level
  isn't worth the productionization cost.

Script: `scripts/_diag_tf_regrasp.py`; data:
`reports/stage5/turnfaucet_diagnostic/regrasp{,_v2}/`.
