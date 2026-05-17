"""Factor-local intent revision.

`revise_intent` produces a new Intent that differs from the input in EXACTLY
ONE field — the field the attribution rule named. Every other factor is
copied byte-identical from the input. This invariant is what makes Stage 0
"factor-local" rather than "generic retry"; the summarizer's
`non_regression_score` audits it on every revised episode.

Stage 0 implements only `approach_substitution`. Other operators raise
`NotImplementedError` — honest about what is and isn't validated.
"""
from __future__ import annotations

from dataclasses import replace

from babysteps.envs.scene import OPPOSITE_APPROACH
from babysteps.failure import Attribution
from babysteps.schemas import APPROACH_DIRECTIONS, INTENT_FIELDS, Intent, Revision, SceneState

# Order in which the reviser searches for an unblocked approach. The opposite
# of the current approach is tried first (the canonical "approach_substitution"
# move), then the remaining cardinals, then "from_above" as a last resort.
_CARDINAL_FALLBACK_ORDER: tuple[str, ...] = (
    "from_minus_x", "from_plus_x", "from_minus_y", "from_plus_y",
)


def _pick_unblocked_approach(current: str, blocked: tuple[str, ...]) -> str:
    blocked_set = set(blocked)
    # First preference: the geometric opposite of the current approach.
    if current in OPPOSITE_APPROACH:
        cand = OPPOSITE_APPROACH[current]
        if cand not in blocked_set and cand != current:
            return cand
    # Next: any other cardinal not currently used or blocked.
    for cand in _CARDINAL_FALLBACK_ORDER:
        if cand != current and cand not in blocked_set:
            return cand
    # Last resort: top approach (whitelisted in APPROACH_DIRECTIONS).
    assert "from_above" in APPROACH_DIRECTIONS
    return "from_above"


def revise_intent(
    intent: Intent, attribution: Attribution, scene: SceneState,
) -> tuple[Intent, Revision]:
    """Return (revised_intent, Revision record). Stage 0 supports
    `wrong_factor == "approach_direction"` only.
    """
    if attribution.wrong_factor is None:
        raise ValueError(
            "revise_intent called with attribution.wrong_factor=None; "
            "the failure was not semantic — nothing to revise."
        )
    if attribution.wrong_factor != "approach_direction":
        raise NotImplementedError(
            f"Stage-0 reviser only handles 'approach_direction'; "
            f"got {attribution.wrong_factor!r}. (Other factors are reserved "
            f"for later stages — see docs/.../2026-05-15-stage0-pushcube-blocked-design.md "
            f"§14)"
        )

    old = intent.approach_direction
    new = _pick_unblocked_approach(old, scene.blocked_sides)
    revised = replace(intent, approach_direction=new)
    frozen = tuple(f for f in INTENT_FIELDS if f != "approach_direction")
    rev_record = Revision(
        operator="approach_substitution",
        factor="approach_direction",
        old_value=old,
        new_value=new,
        frozen_factors=frozen,
    )
    return revised, rev_record
