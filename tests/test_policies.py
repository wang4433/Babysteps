import random
from babysteps.policies import resample_factor
from babysteps.schemas import Intent

_BASE = Intent(
    goal_state="cube_at_target",
    object_motion="translate_+x",
    contact_region="minus_x_face",
    approach_direction="from_minus_x",
    constraint_region="none",
    embodiment_mapping="proxy_contact_to_franka_push",
)
_TOKS = ("minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face")


def test_resample_excludes_current_value():
    rng = random.Random(0)
    for _ in range(50):
        new = resample_factor(_BASE, "contact_region", _TOKS, rng)
        assert new != _BASE.contact_region
        assert new in _TOKS


def test_resample_single_alternative_is_deterministic():
    rng = random.Random(1)
    new = resample_factor(_BASE, "goal_state", ("cube_at_target", "cubeA_on_cubeB"), rng)
    assert new == "cubeA_on_cubeB"


def test_resample_no_alternative_raises():
    import pytest
    rng = random.Random(2)
    with pytest.raises(ValueError):
        resample_factor(_BASE, "contact_region", ("minus_x_face",), rng)
