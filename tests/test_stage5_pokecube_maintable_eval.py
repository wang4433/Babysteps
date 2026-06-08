"""Sim-free tests for scripts/stage5_pokecube_maintable_eval.py.

Guards the 6-condition fair-recovery HARNESS (reach filter, composite-episode
sourcing, on-initial-fail CI subset, 3-face candidate override, single-slot
invariant) with a deterministic _FakePoke (goal direction == injected motion;
a poke moves the cube 0.10 m along the contacted face's push dir) + MockVLM + a
tiny trained shared scorer. NO GPU, NO gitignored models/.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
for _p in (str(_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from babysteps.envs.pokecube_adapter import PokeCubeAdapter
from babysteps.envs.scene import face_to_push_unit, motion_to_unit
from babysteps.schemas import AttemptResult, SceneState
from babysteps.stage5.vlm_attribute import MockVLMClient

import stage5_pokecube_maintable_eval as M
from stage5_pokecube_loto_eval import LOTO_FACES


class _FakePoke:
    def __init__(self, cube=(0.13, 0.0)):
        self._m = None
        self._cube = cube

    def set_injection(self, m):
        self._m = m

    def reset(self, seed):
        u = np.array([1.0, 0.0]) if self._m is None else motion_to_unit(self._m)
        goal = (self._cube[0] + 0.10 * float(u[0]), self._cube[1] + 0.10 * float(u[1]))
        return SceneState(cube_xy=self._cube, cube_z=0.02, goal_xy=goal,
                          tcp_start_pose=(0, 0, 0, 0, 0, 0, 1),
                          blocked_sides=(), extra={})

    def run(self, intent, scene, **kw):
        # Faithful to the real PokeCube LOTO data: a CORRECT-face poke drives the
        # cube to the goal; a WRONG-face poke pokes the wrong side / misses and
        # leaves the cube ~stationary (so the observed residual = goal - cube is
        # clean and axis-aligned, exactly as in results_*.json, residual [0.1,0]).
        u = face_to_push_unit(intent.contact_region)
        g = np.asarray(scene.goal_xy) - np.asarray(scene.cube_xy)
        gn = g / (float(np.linalg.norm(g)) + 1e-9)
        aligned = float(u @ gn) > 0.9
        final = tuple(scene.goal_xy) if aligned else tuple(scene.cube_xy)
        succ = float(np.hypot(final[0] - scene.goal_xy[0],
                              final[1] - scene.goal_xy[1])) < 0.05
        return AttemptResult(initial_obj_xy=scene.cube_xy, final_obj_xy=final,
                             goal_xy=scene.goal_xy, reached_contact=True,
                             object_moved=bool(aligned), planner_failed=False,
                             collision=False, grasp_slip=False,
                             rollout_log_path=None, success=bool(succ),
                             trajectory_xy=())


def _trained_scorer(tmp_path):
    import torch
    from babysteps.stage5.shared_revision_policy import (
        EFAIL_DIM, FACTOR_ORDER, GI_DIM_DEFAULT, SharedScorer, SharedScorerPolicy,
        _efail_vec, build_value_vocab, save_shared_scorer,
    )
    vocab = build_value_vocab()
    scorer = SharedScorer(vocab_size=len(vocab) + 1, n_factors=len(FACTOR_ORDER),
                          d_gi=GI_DIM_DEFAULT, d_efail=EFAIL_DIM, seed=0)
    faces = {"minus_x_face": (1.0, 0.0), "plus_x_face": (-1.0, 0.0),
             "minus_y_face": (0.0, 1.0), "plus_y_face": (0.0, -1.0)}
    cand = list(faces)
    fidx = FACTOR_ORDER.index("contact_region")
    opt = torch.optim.Adam(scorer.parameters(), lr=5e-3)
    rng = np.random.default_rng(0)
    for _ in range(400):
        correct = cand[rng.integers(4)]
        resid = np.array(faces[correct]) * 0.1
        ef = torch.tensor(_efail_vec("direction_error", tuple(resid))).unsqueeze(0)
        cur = cand[rng.integers(4)]
        b = len(cand)
        logits = scorer(
            torch.zeros(b, GI_DIM_DEFAULT), ef.expand(b, -1),
            torch.full((b,), fidx),
            torch.full((b,), vocab[("contact_region", cur)]),
            torch.tensor([vocab[("contact_region", c)] for c in cand]))
        loss = torch.nn.functional.cross_entropy(
            logits.unsqueeze(0), torch.tensor([cand.index(correct)]))
        opt.zero_grad(); loss.backward(); opt.step()
    save_shared_scorer(scorer, vocab, tmp_path / "trained.pt")
    return SharedScorerPolicy.from_pack(tmp_path / "trained.pt")


def _spec(runner):
    return M._build_poke_spec(SimpleNamespace(fake=True), runner, PokeCubeAdapter())


def test_candidates_override_is_three_reachable_faces():
    spec = _spec(_FakePoke())
    assert spec.candidates_override["contact_region"] == tuple(LOTO_FACES)
    assert "plus_x_face" not in spec.candidates_override["contact_region"]


def test_reachable_seeds_filter():
    runner = _FakePoke(cube=(0.13, 0.0))
    dirs = ["+x", "+y", "-y"]
    # no filter -> all up to target_n
    assert M._reachable_seeds(runner, list(range(10)), dirs, None, 4) == [0, 1, 2, 3]
    # impossible reach budget -> none
    assert M._reachable_seeds(runner, list(range(10)), dirs, 0.1, 4) == []
    # generous budget -> kept
    assert len(M._reachable_seeds(runner, list(range(10)), dirs, 5.0, 6)) == 6


def test_ci_table_uses_initial_fail_subset():
    # 3 fail-initial clusters (recover via 'shared') + 1 success-initial cluster.
    flat = []
    for s in range(3):
        flat.append({"seed": s, "initial_success": False,
                     "shared_revision_policy_success": True,
                     "oracle_single_slot_success": True})
    flat.append({"seed": 99, "initial_success": True,
                 "shared_revision_policy_success": True,
                 "oracle_single_slot_success": True})
    ci = M._ci_table(flat, ["shared_revision_policy", "oracle_single_slot"],
                     n_boot=200, seed=0)
    assert ci["n_initial_fail"] == 3
    assert ci["recovery_ci"]["shared_revision_policy"]["n_clusters"] == 3
    assert ci["recovery_ci"]["shared_revision_policy"]["mean"] == 1.0


def test_run_maintable_end_to_end_fake(tmp_path):
    runner = _FakePoke()
    adapter = PokeCubeAdapter()
    vlm = MockVLMClient(constrained_response="contact_region")
    spec = _spec(runner)
    policy = _trained_scorer(tmp_path)
    matrix = M._CONDITION_MATRIX
    labels = [lab for (lab, _b, _a) in matrix]
    rows_by_label, flat_rows, n_reach = M.run_maintable(
        runner, adapter, vlm, spec, seeds=list(range(5)),
        directions=["+x", "+y", "-y"], condition_matrix=matrix,
        shared_policy=policy, max_approach_dist=None, target_n=5,
        capture_frames=False, frames_dir=None)
    # 5 seeds x 3 directions x 2 wrong faces
    assert n_reach == 5 and len(flat_rows) == 30
    for lab in labels:
        assert len(rows_by_label[lab]) == 30
        assert all(f"{lab}_success" in r for r in flat_rows)
    from babysteps.stage5.maintable import aggregate
    summ = aggregate(rows_by_label)
    assert summ["oracle_single_slot"]["recovery_on_initial_fail"] == 1.0
    assert summ["same_intent_retry"]["recovery_on_initial_fail"] == 0.0
    # Mock VLM returns contact_region -> attribution correct -> shared recovers
    # under BOTH attributors; @oracle_attr isolates the value policy.
    assert summ["shared_revision_policy"]["recovery_on_initial_fail"] >= 0.9
    assert summ["shared_revision_policy@oracle_attr"]["recovery_on_initial_fail"] >= 0.9
    # Single-slot invariant for the LOCAL-EDIT labels (<= 1 changed factor);
    # vlm_free_replan is the multi-slot baseline (its selectivity weakness).
    for lab in ("same_intent_retry", "random_factor_local_edit",
                "vlm_diagnosis_local_edit", "vlm_diagnosis_local_edit@oracle_attr",
                "shared_revision_policy", "shared_revision_policy@oracle_attr",
                "oracle_single_slot"):
        assert summ[lab]["edit_cardinality_mean"] <= 1.0
    assert (summ["vlm_free_replan"]["edit_cardinality_mean"]
            > summ["shared_revision_policy"]["edit_cardinality_mean"])
    # CI on the fail subset agrees; value-transfer (oracle attr) == oracle.
    ci = M._ci_table(flat_rows, labels, n_boot=200, seed=0)
    assert ci["recovery_ci"]["shared_revision_policy@oracle_attr"]["mean"] >= 0.9
    assert ci["recovery_ci"]["oracle_single_slot"]["mean"] == 1.0
    # value-transfer paired diff present (shared@oracle vs oracle)
    assert ("shared_revision_policy@oracle_attr__minus__oracle_single_slot"
            in ci["paired_diffs"])


def _distilled(default_mask="multimodal", seed=0):
    from babysteps.stage5.attribution_dataset import make_dataset
    from babysteps.stage5.attribution_head import (
        DistilledAttributor, train_attribution_head)
    head = train_attribution_head(
        make_dataset(n_per_case=32, noise=0.01, seed=seed),
        modality_dropout=0.5, epochs=400, seed=seed)
    return DistilledAttributor(head, default_mask=default_mask)


def test_distilled_attributor_recovery_wiring(tmp_path):
    """Recovery gate plumbing: with --distilled-head, the @distilled_attr rows
    run; the distilled head attributes contact_region (VLM-free) so the shared
    policy recovers like the oracle-attribution path."""
    runner = _FakePoke()
    adapter = PokeCubeAdapter()
    vlm = MockVLMClient(constrained_response="contact_region")
    spec = _spec(runner)
    policy = _trained_scorer(tmp_path)
    distilled = _distilled()
    matrix = M._CONDITION_MATRIX + [
        ("vlm_diagnosis_local_edit@distilled_attr",
         "vlm_diagnosis_local_edit", "distilled"),
        ("shared_revision_policy@distilled_attr",
         "shared_revision_policy", "distilled"),
    ]
    rows_by_label, flat_rows, _ = M.run_maintable(
        runner, adapter, vlm, spec, seeds=list(range(5)),
        directions=["+x", "+y", "-y"], condition_matrix=matrix,
        shared_policy=policy, max_approach_dist=None, target_n=5,
        capture_frames=False, frames_dir=None, distilled_attributor=distilled)
    from babysteps.stage5.maintable import aggregate
    summ = aggregate(rows_by_label)
    # distilled diagnosis == contact_region -> recovers like the oracle path
    assert summ["shared_revision_policy@distilled_attr"]["attribution_accuracy"] >= 0.9
    assert summ["shared_revision_policy@distilled_attr"]["recovery_on_initial_fail"] >= 0.9
    assert summ["shared_revision_policy@oracle_attr"]["recovery_on_initial_fail"] >= 0.9
    # single-slot invariant preserved for the distilled local-edit rows
    for lab in ("vlm_diagnosis_local_edit@distilled_attr",
                "shared_revision_policy@distilled_attr"):
        assert summ[lab]["edit_cardinality_mean"] <= 1.0
    # the recovery-gate paired diff is emitted
    ci = M._ci_table(flat_rows, [l for (l, _b, _a) in matrix], n_boot=100, seed=0)
    assert ("shared_revision_policy@distilled_attr__minus__"
            "shared_revision_policy@oracle_attr") in ci["paired_diffs"]


def test_distilled_override_requires_head():
    """attributor_override='distilled' with no loaded head -> clear error."""
    import pytest
    from stage5_unified_maintable_eval import _paired_actors
    runner = _FakePoke()
    adapter = PokeCubeAdapter()
    vlm = MockVLMClient(constrained_response="contact_region")
    spec = _spec(runner)
    ep = M._source_episode(runner, adapter, 0, "+x", "minus_y_face",
                           capture_frames=False, frames_dir=None)
    with pytest.raises(ValueError):
        _paired_actors("vlm_diagnosis_local_edit", spec, ep, vlm, 0,
                       attributor_override="distilled", distilled_attributor=None)
