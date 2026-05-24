"""Stage-4 M2a counterfactual-synthesis tests.

The substitution primitive lives outside features.py to keep the
firewall test on features.py clean (no mentions of `revision` etc).
"""
from __future__ import annotations

import numpy as np
import pytest


def _fake_pushcube_record() -> dict:
    return {
        "demo": {
            "camera": "third_person",
            "demonstrator_type": "proxy_oracle",
            "object_trajectory": [[0.0, 0.0], [0.05, 0.0], [0.10, 0.01]],
            "contact_region_label": "minus_x_face",
            "final_state": "cube_at_target",
        },
        "execution": {"initial_intent": {"goal_state": "cube_at_target"}},
        "failure_packet": {"failure_predicate": "approach_blocked"},
        "revision": None,
        "retry": None,
    }


def test_substitute_goal_state_flips_only_goal_oh():
    from babysteps.stage4.counterfactual import substitute_label_identity_feature
    from babysteps.stage4.features import (
        CONTACT_OH_START, GOAL_OH_START, GOAL_ORDER,
        extract_episode_features,
    )
    base = extract_episode_features(_fake_pushcube_record())
    out = substitute_label_identity_feature(base, "goal_state", "cubeA_on_cubeB")
    assert out.shape == base.shape
    pre = list(range(0, GOAL_OH_START))
    post = list(range(GOAL_OH_START + len(GOAL_ORDER), len(base)))
    np.testing.assert_array_equal(out[pre], base[pre])
    np.testing.assert_array_equal(out[post], base[post])
    goal_oh = out[GOAL_OH_START:GOAL_OH_START + len(GOAL_ORDER)]
    assert goal_oh.sum() == pytest.approx(1.0)
    assert goal_oh[GOAL_ORDER.index("cubeA_on_cubeB")] == 1.0


def test_substitute_contact_region_flips_only_contact_oh():
    from babysteps.stage4.counterfactual import substitute_label_identity_feature
    from babysteps.stage4.features import (
        CONTACT_OH_START, CONTACT_ORDER, GOAL_OH_START,
        extract_episode_features,
    )
    base = extract_episode_features(_fake_pushcube_record())
    out = substitute_label_identity_feature(base, "contact_region", "plus_x_face")
    pre = list(range(0, CONTACT_OH_START))
    post = list(range(CONTACT_OH_START + len(CONTACT_ORDER), len(base)))
    np.testing.assert_array_equal(out[pre], base[pre])
    np.testing.assert_array_equal(out[post], base[post])
    contact_oh = out[CONTACT_OH_START:CONTACT_OH_START + len(CONTACT_ORDER)]
    assert contact_oh.sum() == pytest.approx(1.0)
    assert contact_oh[CONTACT_ORDER.index("plus_x_face")] == 1.0


def test_substitute_rejects_other_factors():
    from babysteps.stage4.counterfactual import substitute_label_identity_feature
    from babysteps.stage4.features import extract_episode_features
    base = extract_episode_features(_fake_pushcube_record())
    for factor in ("object_motion", "approach_direction",
                   "constraint_region", "embodiment_mapping"):
        with pytest.raises(ValueError):
            substitute_label_identity_feature(base, factor, "anything")


def test_substitute_rejects_unknown_value():
    from babysteps.stage4.counterfactual import substitute_label_identity_feature
    from babysteps.stage4.features import extract_episode_features
    base = extract_episode_features(_fake_pushcube_record())
    with pytest.raises(ValueError):
        substitute_label_identity_feature(base, "goal_state", "not_a_real_token")


def test_substitute_does_not_mutate_input():
    from babysteps.stage4.counterfactual import substitute_label_identity_feature
    from babysteps.stage4.features import extract_episode_features
    base = extract_episode_features(_fake_pushcube_record())
    snapshot = base.copy()
    _ = substitute_label_identity_feature(base, "goal_state", "cubeA_on_cubeB")
    np.testing.assert_array_equal(base, snapshot)
