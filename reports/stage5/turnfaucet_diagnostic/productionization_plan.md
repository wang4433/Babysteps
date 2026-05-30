# Sub-project D productionization plan — re-grasp embodiment adaptation

Decision (2026-05-29/30): re-found Sub-project D on the TRUE failure mode and
ship the re-grasp ratchet oracle, restricted to **vertical-axis faucet models**
(32% > the 30% gate; the cleanest re-grasp story). Evidence:
`oracle_exploration.md`, `reports/stage5/turnfaucet_diagnostic/regrasp{,_v2}/`.

## New narrative (replaces "grasp infeasible → poke")

| step | old | NEW |
| --- | --- | --- |
| demo / initial intent | grasp_turn ("jaws can't close") | **grasp_turn = continuous grasp-and-rotate** (what the source embodiment does) |
| initial Franka attempt | grasp fails to close → no motion | **grasp succeeds, continuous wrist-spin STALLS at Panda joint-7 ±2.9 rad limit** → partial turn, fail |
| failure predicate | `grasp_infeasible` | **`continuous_rotation_infeasible`** (grasp OK; rotation range exceeds wrist) |
| attribution | embodiment_mapping | embodiment_mapping (unchanged) |
| revision (single-factor) | grasp_turn → poke_turn | **grasp_turn → `regrasp_turn`** |
| retry skill | poke lateral sweep | **re-grasp ratchet** (spin to wrist limit → release → rewind → regrip → spin) |

The single-factor invariant holds (only embodiment_mapping changes). The
embodiment mismatch is now *kinematic* (wrist range), not gripper-width — which
is what the physics actually shows.

## Work breakdown (TDD, sim-free first)

1. **Schema (additive).** Add `proxy_contact_to_franka_regrasp_turn` to
   `EMBODIMENT_MAPPINGS`; add `continuous_rotation_infeasible` predicate. Keep
   `poke_turn` + `grasp_infeasible` in the whitelist (back-compat; removal is a
   later cleanup pass). Update `tests/snapshots/` deliberately. [sim-free]
2. **failure.py.** Add the `continuous_rotation_infeasible` predicate row
   (→ embodiment_mapping) and fire it for grasp_turn on TurnFaucet. [sim-free + test]
3. **revision.py.** `embodiment_substitution` grasp_turn → regrasp_turn (keep the
   poke_turn path for back-compat / existing P2-M3 data). [sim-free + test]
4. **adapter.** `scripted_demo_to_intent` → grasp_turn (continuous);
   `oracle_correct_intent` → regrasp_turn; `oracle_wrong_factor` unchanged. [sim-free + test]
5. **skills/turn.py.** Compile `regrasp_turn` → ratchet PARAMS (anchor, axis,
   grip point, spin, spin_steps, n_rewind, max_cycles); compile `grasp_turn` →
   grasp+spin params (single grasp, no re-grasp → stalls). Pure/deterministic. [sim-free + test]
6. **envs/turnfaucet_runner.py.** Add `_execute_grasp_spin` (initial attempt) and
   `_execute_regrasp_ratchet` (retry), ported from `scripts/_diag_tf_regrasp.py`;
   dispatch in `run()` on embodiment_mapping. Reads qpos for control (consistent
   with existing poke auto-sign). [GPU]
7. **Vertical-axis subset.** Curate vertical-axis faucet model IDs
   (`_diag_tf_axis_census` GPU job → `reports/.../vertical_axis_models.json`);
   enforce in the runner/eval (sample only those models). [GPU prereq — running now]
8. **Render.** Update `render/turnfaucet.py` 3-phase MP4: demo = continuous
   grasp-turn; attempt = grasp+spin stalls at wrist limit; retry = ratchet
   succeeds. Captions describe object motion only. [GPU]
9. **Eval + report.** Re-run the held-out cut (vertical-axis subset) through the
   production runner; record the gate result. Update milestones/spec/CLAUDE.md.

## Open design choices (defaults chosen; flag if you disagree)

- **Token name** `proxy_contact_to_franka_regrasp_turn` (parallel to existing).
- **Keep poke_turn** in the schema + as a still-valid revision target (the P2
  5-task table + M3 baselines already use it); regrasp_turn is the new oracle
  target. No data is invalidated.
- **Vertical-axis enforcement** at the runner level (subclass/filter the env's
  model sampling), not a seed filter — principled + reproducible.
- Production step budget: the ratchet needs >200 steps; raise TurnFaucet's
  control-step cap for this skill (honest: re-grasping is inherently multi-step).

## ⚠ BLOCKER discovered during step 1 — Stage-4 feature-dim coupling

`babysteps/stage4/attribution_features.py` (FEATURE_DIM=47) and
`revise_head.py` build their one-hot vocab LIVE from `tuple(sorted(
EMBODIMENT_MAPPINGS))` / `tuple(sorted(FAILURE_PREDICATES))`. Adding the
`regrasp_turn` token + `continuous_rotation_infeasible` predicate shifts
FEATURE_DIM 47→49 and the block offsets, **breaking the trained Stage-4
attribution/latent model packs** and 4 tests (`test_feature_dim_is_47`,
`test_block_offsets`, `test_intent_one_hot_layout`,
`test_latent_pack_save_load_round_trip`). The Intent dataclass validates
`embodiment_mapping` against the frozenset, so the token MUST live there —
the coupling is unavoidable. Schema additions reverted to keep the suite green
(472 pass) pending a decision:

- **(a) Pin the Stage-4 feature vocab** to explicit ordered lists (snapshot),
  append new tokens, bump FEATURE_DIM→49, update the 4 tests deliberately, and
  **regenerate/retrain the Stage-4 packs** at 49-dim. Correct + future-proof
  (schema can then grow safely); costs a Stage-4 retrain.
- **(b) Decouple regrasp from learned attribution:** add only what's needed for
  Intent validation, and keep the Stage-4 feature space frozen by NOT feeding
  the new tokens to it (out-of-vocab → zero one-hot). Cheaper, but the learned
  attribution model can't see the new predicate/token (the rule-table + VLM
  attribution still can).
- Note: Stage-4 learned attribution (M2a) is a completed milestone; Stage-5
  P2 uses VLM attribution, not the Stage-4 packs — so (b)/regen may be low-risk
  on the Stage-5 critical path. Confirm before choosing.

## Status

- [x] Oracle validated (32% vertical, 7× baseline) — `_diag_tf_regrasp.py`.
- [~] Step 7 curation job — running (job 10896981).
- [!] Step 1 schema — blocked on the Stage-4 feature-dim decision above.
- [ ] Steps 2–6, 8–9 — pending the decision + plan review.
