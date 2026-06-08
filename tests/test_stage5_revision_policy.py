"""Sim-free tests for the Stage-5 typed revision-decision interface.

Covers (no env / GPU / Vulkan — login node):
  * PushCube LEARNED slot edit: a trained tiny residual ReviseHead exercised
    end-to-end through ResidualSlotEditor (vectorize residual → head →
    decode_slot → token).
  * StackCube typed-operator edit (goal_refinement) and its consistency with
    babysteps.revision.revise_intent.
  * Attribution ⟂ revision separation, candidate resolution for ANY diagnosed
    factor (incl. a mis-diagnosis), and the single-slot compiler invariant.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from babysteps.failure import Attribution
from babysteps.revision import revise_intent
from babysteps.schemas import INTENT_FIELDS, Intent, SceneState
from babysteps.stage4.revise_head import (
    FP_VECTOR_DIM_RESIDUAL,
    ReviseHead,
    train_revise_head_l2,
    vectorize_failure_packet_residual,
)
from babysteps.stage5.residual_reviser import ResidualSlotEditor
from babysteps.stage5.revision_policy import (
    FailureEvidence,
    OracleAttributor,
    OracleValuePolicy,
    PerTaskEditorAdapter,
    RandomAttributor,
    RandomCandidatePolicy,
    RevisionRequest,
    TYPED_OPERATORS,
    VLMAttributor,
    candidates_for,
    compile_single_slot_edit,
)
from babysteps.stage5.vlm_attribute import MockVLMClient

FACES = ("minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face")

GT = Intent(
    goal_state="cube_at_target",
    object_motion="translate_+x",
    contact_region="plus_x_face",
    approach_direction="from_plus_x",
    constraint_region="none",
    embodiment_mapping="proxy_contact_to_franka_push",
)


def _trained_pushcube_editor() -> ResidualSlotEditor:
    """A ResidualSlotEditor whose tiny head is trained to map (wrong face,
    residual toward the correct face) → the correct face's centroid."""
    centroids = {i: np.eye(4, dtype=np.float32)[i] for i in range(4)}
    head = ReviseHead(d_slot=4, fp_dim=FP_VECTOR_DIM_RESIDUAL, hidden=16, seed=0)
    # One memorized example: current=minus_x_face, residual=+x → target plus_x.
    rec = {"revision": {"factor": "contact_region"},
           "failure_packet": {"failure_predicate": "direction_error"}}
    g_pre = centroids[0][None, :]                       # minus_x_face
    fp = vectorize_failure_packet_residual(rec, (1.0, 0.0))[None, :]
    g_tgt = centroids[1][None, :]                       # plus_x_face
    train_revise_head_l2(head, g_pre, fp, g_tgt, n_epochs=400, lr=1e-2)
    return ResidualSlotEditor(factor="contact_region", centroids=centroids,
                              tokens=FACES, head=head)


def test_pushcube_learned_residual_editor_recovers():
    editor = _trained_pushcube_editor()
    assert editor("minus_x_face", (1.0, 0.0), "direction_error") == "plus_x_face"
    # A value outside the learned vocab → no decision (None).
    assert editor("faucet_base", (1.0, 0.0), "direction_error") is None


def test_pushcube_per_task_adapter_native_and_fallback():
    editor = _trained_pushcube_editor()
    adapter = PerTaskEditorAdapter({
        "contact_region": lambda req: editor(
            req.current_value, req.e_fail.residual_xy, req.e_fail.predicate)})
    # Native factor → learned edit.
    req = RevisionRequest(
        factor="contact_region", current_value="minus_x_face",
        candidates=candidates_for("PushCube-v1", "contact_region"),
        e_fail=FailureEvidence(predicate="direction_error",
                               residual_xy=(1.0, 0.0)))
    dec = adapter.decide(req)
    assert dec.factor == "contact_region" and dec.new_value == "plus_x_face"
    # Mis-diagnosed (non-native) factor → deterministic candidate fallback,
    # NEVER re-choosing the factor (corrections #3/#4).
    req2 = RevisionRequest(
        factor="approach_direction", current_value="from_minus_x",
        candidates=candidates_for("PushCube-v1", "approach_direction"))
    dec2 = adapter.decide(req2)
    assert dec2.factor == "approach_direction"
    assert dec2.new_value is not None and dec2.new_value != "from_minus_x"


def test_stackcube_typed_operator_and_compiler():
    adapter = PerTaskEditorAdapter({
        "goal_state": lambda req: TYPED_OPERATORS["goal_refinement"].get(
            req.current_value)})
    initial = replace(GT, goal_state="cube_at_target")
    req = RevisionRequest(
        factor="goal_state", current_value="cube_at_target",
        candidates=candidates_for("StackCube-v1", "goal_state"))
    dec = adapter.decide(req)
    assert dec.new_value == "cubeA_on_cubeB"
    revised = compile_single_slot_edit(initial, dec, INTENT_FIELDS)
    # Exactly one slot changed.
    changed = [f for f in INTENT_FIELDS
               if getattr(initial, f) != getattr(revised, f)]
    assert changed == ["goal_state"]


def test_typed_operator_matches_revision_module():
    """TYPED_OPERATORS['goal_refinement'] must mirror revise_intent so the
    interim StackCube editor and the Stage-0 compiler never drift."""
    initial = replace(GT, goal_state="cube_at_target")
    scene = SceneState(cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.1, 0.0),
                       tcp_start_pose=(0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0),
                       blocked_sides=())
    attr = Attribution(
        semantic_failure=True, wrong_factor="goal_state",
        freeze=tuple(f for f in INTENT_FIELDS if f != "goal_state"),
        revise=("goal_state",))
    revised, _rec = revise_intent(initial, attr, scene)
    assert revised.goal_state == TYPED_OPERATORS["goal_refinement"]["cube_at_target"]


def test_compiler_noop_when_value_unchanged_or_none():
    initial = replace(GT, contact_region="plus_x_face")
    from babysteps.stage5.revision_policy import RevisionDecision
    # No-op: new_value == current.
    noop = RevisionDecision("contact_region", new_value="plus_x_face")
    assert compile_single_slot_edit(initial, noop, INTENT_FIELDS) is initial
    # None decision → unchanged.
    assert compile_single_slot_edit(initial, None, INTENT_FIELDS) is initial


def test_candidates_for_resolves_any_factor():
    assert set(candidates_for("PushCube-v1", "contact_region")) == set(FACES)
    assert candidates_for("StackCube-v1", "goal_state") == (
        "cube_at_target", "cubeA_on_cubeB")
    # Any non-task factor still resolves to the schema vocab (corrections #4).
    appr = candidates_for("PushCube-v1", "approach_direction")
    assert "from_plus_x" in appr and appr == tuple(sorted(appr))


def test_random_and_oracle_attributors():
    obs_menu = INTENT_FIELDS
    from babysteps.stage5.revision_policy import AttributionObs
    obs = AttributionObs(task="PushCube-v1", factor_menu=obs_menu,
                         failure_predicate="direction_error",
                         initial_intent=GT, key=7)
    rnd = RandomAttributor(seed=0).attribute(obs)
    assert rnd.factor in obs_menu
    assert RandomAttributor(seed=0).attribute(obs).factor == rnd.factor  # det.
    ora = OracleAttributor("contact_region").attribute(obs)
    assert ora.factor == "contact_region"


def test_vlm_attributor_reads_cost_meter():
    vlm = MockVLMClient(constrained_response="contact_region",
                        synthetic_latency_s=0.5, synthetic_gen_tokens=6)
    from babysteps.stage5.revision_policy import AttributionObs
    obs = AttributionObs(task="PushCube-v1", factor_menu=INTENT_FIELDS,
                         failure_predicate="direction_error",
                         initial_intent=GT, frame_path=None, key=1)
    res = VLMAttributor(vlm).attribute(obs)
    assert res.factor == "contact_region"
    assert res.latency_s == 0.5
    assert res.cost["n_calls"] == 1 and res.cost["gen_tokens"] == 6


def test_oracle_value_policy_uses_injected_gt_not_request():
    # The oracle value is injected at construction (evaluator-side), never via
    # the model-visible RevisionRequest (corrections #2).
    policy = OracleValuePolicy(oracle_value="plus_x_face")
    req = RevisionRequest(factor="contact_region", current_value="minus_x_face",
                          candidates=candidates_for("PushCube-v1",
                                                    "contact_region"))
    assert policy.decide(req).new_value == "plus_x_face"


def test_random_candidate_policy_never_returns_current():
    pol = RandomCandidatePolicy(seed=0)
    req = RevisionRequest(factor="contact_region", current_value="plus_x_face",
                          candidates=candidates_for("PushCube-v1",
                                                    "contact_region"))
    assert pol.decide(req).new_value != "plus_x_face"
