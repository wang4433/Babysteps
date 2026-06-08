"""Sim-free tests for scripts/stage5_pokecube_loto_eval.py.

Validates the LOTO eval HARNESS (per-episode loop + aggregation) with a fake
PokeCube runner and a perfect-rule policy stub — NO GPU, NO gitignored models/.
The frozen scorer's actual transfer quality is measured by the GPU run; these
tests only guard the harness logic (residual computation, condition wiring,
reachability gating, aggregation).
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
for p in (str(_ROOT), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from babysteps.schemas import AttemptResult, SceneState
from babysteps.envs.pokecube_adapter import PokeCubeAdapter
from babysteps.envs.scene import face_to_push_unit, motion_to_unit
from babysteps.stage5.revision_policy import RevisionDecision

import stage5_pokecube_loto_eval as L


class _FakePoke:
    """Cube near base; a poke moves the cube 0.10 m along the contacted face's
    push direction (so the correct face reaches goal, a wrong face does not).
    Models exactly the abstraction the residual->face rule relies on."""

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

    def run(self, intent, scene):
        u = face_to_push_unit(intent.contact_region)
        final = (scene.cube_xy[0] + 0.10 * float(u[0]),
                 scene.cube_xy[1] + 0.10 * float(u[1]))
        succ = float(np.hypot(final[0] - scene.goal_xy[0],
                              final[1] - scene.goal_xy[1])) < 0.05
        return AttemptResult(initial_obj_xy=scene.cube_xy, final_obj_xy=final,
                             goal_xy=scene.goal_xy, reached_contact=True,
                             object_moved=True, planner_failed=False,
                             collision=False, grasp_slip=False,
                             rollout_log_path=None, success=bool(succ),
                             trajectory_xy=())


class _RulePolicy:
    """Picks the candidate face whose push direction best aligns with the
    residual (the ground-truth residual->face rule)."""

    def decide(self, req):
        r = np.asarray(req.e_fail.residual_xy, dtype=float)
        best = max(req.candidates, key=lambda f: float(face_to_push_unit(f) @ r))
        nv = None if best == req.current_value else best
        return RevisionDecision(req.factor, nv, confidence=1.0)


def test_residual_is_goal_minus_final():
    r = L._residual((0.30, 0.0), (0.10, 0.0))
    assert abs(r[0] - 0.20) < 1e-9 and abs(r[1]) < 1e-9
    assert L._residual((0.1, 0.2), (0.1, 0.2)) == (0.0, 0.0)


def test_run_loto_perfect_rule_recovers_open_loop_fails():
    rows, n_reach = L.run_loto(
        _RulePolicy(), PokeCubeAdapter(), _FakePoke(),
        seeds=list(range(5)), directions=["+x", "+y", "-y"],
        candidates=L.LOTO_FACES, max_approach_dist=None, target_n=5)
    # 5 seeds x 3 directions x 2 wrong faces
    assert n_reach == 5 and len(rows) == 30
    agg = L.aggregate_loto(rows)
    assert agg["open_loop_success"] == 0.0            # wrong face never succeeds
    assert agg["oracle_success"] == 1.0               # correct face always
    assert agg["shared_scorer_face_acc"] == 1.0       # rule picks correct face
    assert agg["shared_scorer_recovery"] == 1.0       # corrected -> success
    # random over 3 faces: correct ~1/3 (loose bound, deterministic seed)
    assert 0.0 <= agg["random_face_success"] <= 0.7


def test_target_n_caps_reachable_seeds():
    rows, n_reach = L.run_loto(
        _RulePolicy(), PokeCubeAdapter(), _FakePoke(),
        seeds=list(range(50)), directions=["+x"], candidates=L.LOTO_FACES,
        max_approach_dist=None, target_n=4)
    assert n_reach == 4
    # 1 direction (+x), 2 wrong faces among the 3 candidates
    assert len(rows) == 4 * 2


def test_aggregate_loto_empty():
    agg = L.aggregate_loto([])
    assert agg["n_episodes"] == 0 and agg["shared_scorer_recovery"] == 0.0
