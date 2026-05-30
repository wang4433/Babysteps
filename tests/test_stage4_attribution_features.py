"""Sim-free tests for the Stage-4 M2.5 attribution-feature extractor.

These tests pin the layout, enforce the no-privileged-import firewall,
and cover the None-direction edge case. They MUST stay independent of
numpy/torch GPU code and run on the login node.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from babysteps.schemas import (
    APPROACH_DIRECTIONS,
    CONSTRAINT_REGIONS,
    CONTACT_REGIONS,
    EMBODIMENT_MAPPINGS,
    FAILURE_PREDICATES,
    GOAL_STATES,
    INTENT_FIELDS,
    OBJECT_MOTIONS,
)
from babysteps.stage4 import attribution_features as af


def _fake_intent(**overrides):
    """Minimal dict-form intent suitable for the feature extractor."""
    base = {
        "goal_state": "cube_at_target",
        "object_motion": "translate_+y",
        "contact_region": "minus_x_face",
        "approach_direction": "from_minus_x",
        "constraint_region": "none",
        "embodiment_mapping": "proxy_contact_to_franka_push",
    }
    base.update(overrides)
    return base


def _fake_fp(**overrides):
    base = {
        "failure_predicate": "approach_blocked",
        "execution_trace": {
            "reached_contact": False,
            "object_moved": False,
            "collision": False,
            "planner_failed": True,
            "grasp_slip": False,
        },
        "object_displacement": 0.0,
        "direction_alignment": None,
    }
    base.update(overrides)
    return base


# --------------------------- Layout pinning ----------------------------- #


def test_feature_dim_is_47():
    """If this changes the model in/out dims and the saved packs change too."""
    assert af.FEATURE_DIM == 47


def test_block_offsets():
    """Pin block layout so downstream callers can rely on it."""
    assert af.PRED_OH_START == 0
    assert af.TRACE_START == len(af.PREDICATE_ORDER) == 9
    assert af.DISP_START == af.TRACE_START + 5
    assert af.ALIGN_START == af.DISP_START + 1
    assert af.INTENT_OH_START == af.ALIGN_START + 2


def test_predicate_one_hot_layout():
    """Predicates are in sorted order (pinned via FEATURE_FROZEN_EXCLUDE so the
    feature vocab stays frozen as the schema grows); 'none' is included."""
    assert af.PREDICATE_ORDER == tuple(
        sorted(FAILURE_PREDICATES - af.FEATURE_FROZEN_EXCLUDE))
    assert "none" in af.PREDICATE_ORDER


def test_intent_one_hot_layout():
    """Intent factors appear in INTENT_FIELDS order."""
    sizes = [len(af.FACTOR_TOKEN_ORDER[f]) for f in INTENT_FIELDS]
    assert sum(sizes) == 30
    # Per-factor sizes match the schema whitelists, minus any tokens pinned out
    # of the frozen feature vocab (FEATURE_FROZEN_EXCLUDE — keeps FEATURE_DIM
    # stable as the schema grows).
    excl = af.FEATURE_FROZEN_EXCLUDE
    expected = {
        "goal_state": len(GOAL_STATES - excl),
        "object_motion": len(OBJECT_MOTIONS - excl),
        "contact_region": len(CONTACT_REGIONS - excl),
        "approach_direction": len(APPROACH_DIRECTIONS - excl),
        "constraint_region": len(CONSTRAINT_REGIONS - excl),
        "embodiment_mapping": len(EMBODIMENT_MAPPINGS - excl),
    }
    for f in INTENT_FIELDS:
        assert len(af.FACTOR_TOKEN_ORDER[f]) == expected[f]


# --------------------------- Firewall ---------------------------------- #


def test_firewall_no_privileged_imports():
    """AST-level check: the module must not IMPORT any simulator / GPU symbol.
    Docstring mentions are fine (we describe what we are NOT allowed to do)."""
    import ast
    src = inspect.getsource(af)
    tree = ast.parse(src)
    forbidden_modules = (
        "mani_skill", "sapien", "babysteps.envs", "torch.cuda",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for bad in forbidden_modules:
                    assert not alias.name.startswith(bad), (
                        f"firewall violation: import {alias.name!r}")
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for bad in forbidden_modules:
                assert not node.module.startswith(bad), (
                    f"firewall violation: from {node.module!r} import …")


def test_firewall_no_oracle_label():
    """The extractor must not read `oracle_wrong_factor` (label leakage)."""
    src = inspect.getsource(af)
    assert "oracle_wrong_factor" not in src
    # Likewise must not read fields that only exist on the disk record's
    # serialized attribution (those would also leak).
    for forbidden in ("freeze", "wrong_factor"):
        # The string `freeze` is fine as a method name but should not be
        # referenced as a dict key on the input.
        assert f'"{forbidden}"' not in src, f"firewall violation: {forbidden!r}"
        assert f"'{forbidden}'" not in src, f"firewall violation: {forbidden!r}"


# --------------------------- Numeric behavior --------------------------- #


def test_vectorize_returns_fixed_shape_float64():
    v = af.vectorize_attribution_input(_fake_fp(), _fake_intent())
    assert v.shape == (af.FEATURE_DIM,)
    assert v.dtype == np.float64


def test_direction_alignment_none_handled():
    v = af.vectorize_attribution_input(_fake_fp(direction_alignment=None),
                                       _fake_intent())
    assert v[af.ALIGN_START] == 0.0
    assert v[af.ALIGN_START + 1] == 0.0  # not-present flag


def test_direction_alignment_present_flag_set():
    v = af.vectorize_attribution_input(_fake_fp(direction_alignment=0.42),
                                       _fake_intent())
    assert v[af.ALIGN_START] == pytest.approx(0.42)
    assert v[af.ALIGN_START + 1] == 1.0


def test_predicate_one_hot_is_unique():
    v = af.vectorize_attribution_input(_fake_fp(failure_predicate="no_motion"),
                                       _fake_intent())
    block = v[af.PRED_OH_START:af.PRED_OH_START + 9]
    assert block.sum() == 1.0
    assert block[af.PREDICATE_ORDER.index("no_motion")] == 1.0


def test_intent_factor_one_hots_sum_to_six():
    v = af.vectorize_attribution_input(_fake_fp(), _fake_intent())
    intent_block = v[af.INTENT_OH_START:]
    assert intent_block.sum() == 6.0  # one one-hot per factor


def test_object_displacement_clamped():
    v = af.vectorize_attribution_input(_fake_fp(object_displacement=10.0),
                                       _fake_intent())
    assert v[af.DISP_START] == 1.0  # clamped to upper bound
    v = af.vectorize_attribution_input(_fake_fp(object_displacement=-1.0),
                                       _fake_intent())
    assert v[af.DISP_START] == 0.0  # clamped to lower bound


def test_rejects_unknown_predicate():
    with pytest.raises(ValueError):
        af.vectorize_attribution_input(
            _fake_fp(failure_predicate="not_a_real_predicate"), _fake_intent())


def test_rejects_unknown_intent_token():
    with pytest.raises(ValueError):
        af.vectorize_attribution_input(_fake_fp(),
                                       _fake_intent(goal_state="bogus"))


def test_works_on_dataclass_inputs():
    """The extractor accepts the live FailurePacket + Intent dataclasses
    in addition to dicts (same string field names)."""
    from babysteps.schemas import FailurePacket, Intent
    intent = Intent.from_dict(_fake_intent())
    fp = FailurePacket(
        chosen_intent=intent,
        execution_trace=_fake_fp()["execution_trace"],
        failure_predicate="approach_blocked",
        object_displacement=0.0,
        direction_alignment=None,
    )
    v = af.vectorize_attribution_input(fp, intent)
    assert v.shape == (af.FEATURE_DIM,)
