"""Stage-4 feature-extraction tests — firewall-strict by design."""

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


def test_features_shape_is_19():
    from babysteps.stage4.features import extract_episode_features
    feats = extract_episode_features(_fake_pushcube_record())
    assert feats.shape == (19,)
    assert feats.dtype == np.float64


def test_features_are_deterministic_in_order():
    from babysteps.stage4.features import extract_episode_features
    a = extract_episode_features(_fake_pushcube_record())
    b = extract_episode_features(_fake_pushcube_record())
    np.testing.assert_array_equal(a, b)


def test_one_hot_contact_region_matches_whitelist():
    from babysteps.stage4.features import extract_episode_features
    from babysteps.schemas import CONTACT_REGIONS
    feats = extract_episode_features(_fake_pushcube_record())
    one_hot_slice = feats[9:9 + len(CONTACT_REGIONS)]
    assert one_hot_slice.sum() == pytest.approx(1.0)


def test_displacement_norm_is_positive_for_moving_cube():
    from babysteps.stage4.features import extract_episode_features
    feats = extract_episode_features(_fake_pushcube_record())
    assert feats[6] > 0.0  # displacement norm index


def test_firewall_rejects_missing_demo():
    from babysteps.stage4.features import extract_episode_features
    rec = _fake_pushcube_record()
    rec.pop("demo")
    with pytest.raises(KeyError):
        extract_episode_features(rec)


def test_firewall_extractor_does_not_reference_intent_fields():
    """Static check: extractor source must not mention leakage field names."""
    import inspect
    from babysteps.stage4 import features
    src = inspect.getsource(features)
    for forbidden in (
        "initial_intent",
        "failure_packet",
        "revision",
        "retry",
        "oracle_wrong_factor",
        "wrong_factor",
    ):
        assert forbidden not in src, f"firewall violation: {forbidden!r} in features.py"
