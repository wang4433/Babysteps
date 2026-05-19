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
