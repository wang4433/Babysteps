"""Stage-4 M2a — counterfactual synthesis for label-identity factors.

This module lives outside `babysteps/stage4/features.py` so the static
firewall guard on features.py (no mentions of `revision` / `intent` /
`failure_packet`) stays clean. The synthesis primitive
`substitute_label_identity_feature` only consumes a feature vector
`Z` and a target token — it does not read records, intents, or
failure packets.

Use case: a revision whose target class isn't in any initial intent
(e.g. StackCube `goal_state` always starts at `cube_at_target` but
is revised to `cubeA_on_cubeB`) has no encoder centroid to land on
during M2a A2 (see `reports/stage4/m2a_a2/notes.md` §"Data
limitation"). For LABEL-IDENTITY factors (`goal_state`,
`contact_region`) whose value lives in a one-hot column of Z, we can
synthesize `(Z', new_class)` training pairs by flipping the relevant
one-hot. The encoder then has a region for the previously-unseen
class without any new GPU rollouts.

This trick does NOT extend to factors whose value is not in Z
(`object_motion` is geometric not one-hot; `approach_direction` is
not in Z directly — it is derived from `contact_region` in PushCube
but not in StackCube; `constraint_region` and `embodiment_mapping`
are task-constant in the current cut).
"""
from __future__ import annotations

import numpy as np

from babysteps.stage4.features import (
    CONTACT_OH_START, CONTACT_ORDER, GOAL_OH_START, GOAL_ORDER,
)


def substitute_label_identity_feature(
    Z: np.ndarray,
    factor: str,
    new_value: str,
) -> np.ndarray:
    """Return a copy of Z with the label-identity one-hot for `factor`
    swapped to `new_value`'s position.

    Supported `factor` values: `goal_state` (lives in the goal one-hot)
    and `contact_region` (lives in the contact one-hot). Other factors
    raise ValueError.

    `new_value` must be in the corresponding schema whitelist
    (`babysteps.schemas.GOAL_STATES` or `.CONTACT_REGIONS`). The input
    `Z` is never mutated.
    """
    out = Z.copy()
    if factor == "goal_state":
        if new_value not in GOAL_ORDER:
            raise ValueError(
                f"new_value {new_value!r} not in GOAL_STATES whitelist; "
                f"valid: {GOAL_ORDER}"
            )
        out[GOAL_OH_START:GOAL_OH_START + len(GOAL_ORDER)] = 0.0
        out[GOAL_OH_START + GOAL_ORDER.index(new_value)] = 1.0
    elif factor == "contact_region":
        if new_value not in CONTACT_ORDER:
            raise ValueError(
                f"new_value {new_value!r} not in CONTACT_REGIONS whitelist; "
                f"valid: {CONTACT_ORDER}"
            )
        out[CONTACT_OH_START:CONTACT_OH_START + len(CONTACT_ORDER)] = 0.0
        out[CONTACT_OH_START + CONTACT_ORDER.index(new_value)] = 1.0
    else:
        raise ValueError(
            f"substitute_label_identity_feature only supports "
            f"goal_state and contact_region; got {factor!r}"
        )
    return out
