"""Cross-view grounding (Sub-project E) unit + end-to-end tests."""
from __future__ import annotations

import numpy as np
import pytest

from babysteps import schemas


def test_direction_groundings_whitelist():
    assert schemas.DIRECTION_GROUNDINGS == frozenset(
        {"actor_frame", "observer_frame", "object_frame", "world_frame"}
    )


def test_grounding_substitution_operator_registered():
    assert "grounding_substitution" in schemas.REVISION_OPERATORS
    # Existing operators preserved.
    assert "approach_substitution" in schemas.REVISION_OPERATORS


def test_intent_direction_grounding_defaults_and_omits():
    base = dict(
        goal_state="cube_at_target", object_motion="translate_+x",
        contact_region="minus_x_face", approach_direction="from_minus_x",
        constraint_region="none", embodiment_mapping="proxy_contact_to_franka_push",
    )
    # Default value is world_frame and is OMITTED from to_dict (snapshot-safe).
    i_default = schemas.Intent(**base)
    assert i_default.direction_grounding == "world_frame"
    assert "direction_grounding" not in i_default.to_dict()

    # Non-default value IS serialized and round-trips.
    i_actor = schemas.Intent(**base, direction_grounding="actor_frame")
    d = i_actor.to_dict()
    assert d["direction_grounding"] == "actor_frame"
    assert schemas.Intent.from_dict(d) == i_actor

    # A dict without the key reads back as the default.
    assert schemas.Intent.from_dict(i_default.to_dict()).direction_grounding == "world_frame"


def test_intent_direction_grounding_validated():
    with pytest.raises(ValueError):
        schemas.Intent(
            goal_state="cube_at_target", object_motion="translate_+x",
            contact_region="minus_x_face", approach_direction="from_minus_x",
            constraint_region="none",
            embodiment_mapping="proxy_contact_to_franka_push",
            direction_grounding="banana",
        )
