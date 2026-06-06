"""Sim-free test for the StackCube goal_state loop driver core.

Uses a fake runner whose success keys on intent.goal_state (cubeA_on_cubeB ->
success; cube_at_target -> collide/scatter = goal_not_satisfied), a real
StackCubeAdapter + a real goal_state pack/VisionIntentExtractor, and demo feature
files crafted so some seeds decode correctly (initial success) and some misground
(initial fail -> operator/oracle recover, same_intent does not). No GPU/Vulkan.
"""
from __future__ import annotations

import json

import numpy as np

from babysteps.envs.stackcube_adapter import StackCubeAdapter
from babysteps.schemas import AttemptResult, INTENT_FIELDS, SceneState
from babysteps.stage4.vision_intent import VisionIntentExtractor
from scripts.stage5_goalstate_loop_eval import run_goalstate_episode, summarize
from scripts.stage5_train_goalstate_pack import main as train_pack_main

_D = 8
_NEAR = "cube_at_target"
_STACK = "cubeA_on_cubeB"


def _spike(cls_i, rng, noise=0.05):
    v = np.zeros(_D, dtype=np.float32)
    v[cls_i] = 5.0
    return v + rng.normal(0, noise, _D).astype(np.float32)


def _train_goalstate_pack(tmp_path):
    feats = tmp_path / "trainfeats"
    feats.mkdir()
    rng = np.random.default_rng(0)
    labels = {}
    for cls_i, (tok, tag) in enumerate(((_NEAR, "near"), (_STACK, "stack"))):
        for k in range(30):
            stem = f"seed_{k:04d}_{tag}"
            np.save(feats / f"{stem}_dinov2_fl.npy", _spike(cls_i, rng, 0.3))
            labels[stem] = tok
    (feats / "labels.json").write_text(json.dumps({
        "task": "StackCube-v1", "factor": "goal_state", "pool": "first_last",
        "feature_suffix": "dinov2_fl", "labels": labels}))
    pack = tmp_path / "pack"
    rc = train_pack_main(["--features-dir", str(feats), "--feature-suffix",
                          "dinov2_fl", "--out-dir", str(pack)])
    assert rc == 0
    return pack


class _FakeStackRunner:
    """reset->SceneState; run succeeds iff intent.goal_state == cubeA_on_cubeB.
    The cube_at_target failure mimics the low-z collision -> goal_not_satisfied."""

    def reset(self, seed):
        self._scene = SceneState(
            cube_xy=(0.0, 0.0), cube_z=0.02, goal_xy=(0.05, 0.0),
            tcp_start_pose=(0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0),
            blocked_sides=(),
            extra={"cubeB_xy": (0.05, 0.0), "cubeB_z": 0.02, "cubeB_top_z": 0.06})
        return self._scene

    def run(self, intent, scene):
        ok = intent.goal_state == _STACK
        return AttemptResult(
            initial_obj_xy=scene.cube_xy, final_obj_xy=scene.goal_xy,
            goal_xy=scene.goal_xy, reached_contact=True, object_moved=True,
            planner_failed=False, collision=False, grasp_slip=False,
            rollout_log_path=None, success=ok)


def test_goalstate_loop_recovers_misgroundings(tmp_path):
    pack = _train_goalstate_pack(tmp_path)
    adapter = StackCubeAdapter()
    runner = _FakeStackRunner()
    template = adapter.oracle_correct_intent(runner.reset(0))
    ex = VisionIntentExtractor.from_pack(pack, template)

    feats = tmp_path / "evalfeats"
    feats.mkdir()
    rng = np.random.default_rng(7)
    # seeds 200,202 decode STACK (correct -> initial success);
    # seeds 201,203 decode NEAR (misground -> initial fail -> revise).
    plan = {200: 1, 201: 0, 202: 1, 203: 0}
    for s, cls_i in plan.items():
        np.save(feats / f"seed_{s:04d}_stack_dinov2_fl.npy", _spike(cls_i, rng))

    revisers = ["same_intent", "operator", "oracle_value"]
    rows = []
    for s in plan:
        row = run_goalstate_episode(
            adapter, runner, ex, seed=s, demo_features_dir=feats,
            suffix="dinov2_fl", revisers=revisers)
        rows.append(row)

    by_seed = {r["seed"]: r for r in rows}
    # Correct decode -> stack -> initial success.
    assert by_seed[200]["decoded_goal_state"] == _STACK
    assert by_seed[200]["initial_success"] is True
    assert by_seed[200]["vision_decode_correct"] is True
    # Misground -> near -> initial fail; goal_not_satisfied; operator/oracle recover.
    assert by_seed[201]["decoded_goal_state"] == _NEAR
    assert by_seed[201]["initial_success"] is False
    assert by_seed[201]["failure_predicate"] == "goal_not_satisfied"
    assert by_seed[201]["revisers"]["operator"]["final_success"] is True
    assert by_seed[201]["revisers"]["operator"]["new_goal_state"] == _STACK
    assert by_seed[201]["revisers"]["oracle_value"]["final_success"] is True
    # same_intent re-runs the mis-grounded intent -> still fails.
    assert by_seed[201]["revisers"]["same_intent"]["final_success"] is False

    summ = summarize(rows, revisers)
    assert summ["n"] == 4 and summ["n_initial_fail"] == 2
    assert summ["vision_decode_acc"] == 0.5      # 2/4 decode correctly
    assert summ["initial_success_rate"] == 0.5
    # operator/oracle recover ALL initial failures; same_intent recovers none.
    assert summ["final_success_rate_on_initial_fail"]["operator"] == 1.0
    assert summ["final_success_rate_on_initial_fail"]["oracle_value"] == 1.0
    assert summ["final_success_rate_on_initial_fail"]["same_intent"] == 0.0
    # all-episode final: operator lifts 0.5 -> 1.0.
    assert summ["final_success_rate"]["operator"] == 1.0
    assert summ["final_success_rate"]["same_intent"] == 0.5
