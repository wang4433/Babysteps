"""Stage-5 P2 — slot-local selectivity metrics (PURE, sim-free).

The headline experiment compares C1 (VLM-constrained diagnosis + slot-local
edit) against C2 (VLM free-form replan). Both are measured against the SAME
``implicated_factor`` (the oracle wrong factor) and the SAME ground-truth
correct intent, so the two conditions are directly comparable.

This module has no env / GPU / simulator import — it operates purely on
:class:`babysteps.schemas.Intent` values and a factor menu. It is imported by
``scripts/stage5_p2_vlm_eval.py`` and unit-tested in
``tests/test_stage5_p2_selectivity.py``.

Definitions (all measured against ``implicated_factor = oracle_wrong_factor``):

* ``preservation`` — mean over the NON-implicated factors of whether the
  revised value equals the initial value (a slot-local C1 edit preserves all
  of them → 1.0). If ``revised is None`` (parse-fail / no-op), preservation is
  0.0 for aggregation; the caller keeps ``parse_failure`` separately so the two
  remain separable.
* ``unnecessary_changes`` — factors OTHER than the implicated one that the
  revision changed (count + rate over ``len(menu) - 1``).
* ``harmful_changes`` — factors that were CORRECT in the initial intent
  (``initial == gt``) but the revision moved AWAY from the ground truth
  (``revised != gt``). Count + rate over ``len(menu)``. This is the metric that
  exposes a free-replan baseline flipping a factor it had right.
"""
from __future__ import annotations

from typing import Optional

from babysteps.schemas import INTENT_FIELDS, Intent


def _changed_factors(
    initial: Intent, revised: Intent, factor_menu: tuple[str, ...],
) -> tuple[str, ...]:
    """Factors in ``factor_menu`` whose value differs between the two intents.

    Same semantics as ``stage5_p2_vlm_eval._factors_changed`` — kept here so
    the pure module has no dependency on the eval script.
    """
    return tuple(
        f for f in factor_menu if getattr(initial, f) != getattr(revised, f)
    )


def selectivity_metrics(
    initial: Intent,
    revised: Optional[Intent],
    gt: Intent,
    implicated_factor: str,
    factor_menu: tuple[str, ...] = INTENT_FIELDS,
) -> dict:
    """Compute slot-local selectivity metrics for ONE episode.

    Measured against ``implicated_factor`` (= oracle wrong factor) for BOTH C1
    and C2, so they are directly comparable.

    Returns a dict with:
      * ``preservation`` (float in [0, 1])
      * ``unnecessary_changes_count`` / ``unnecessary_changes_rate``
      * ``harmful_changes_count`` / ``harmful_changes_rate``
    """
    non_implicated = tuple(f for f in factor_menu if f != implicated_factor)

    if revised is None:
        # No revision was produced (C2 parse-fail or C1 revise exception): the
        # intent is UNCHANGED, so every frozen factor is preserved and nothing
        # is damaged. Whether the repair *succeeded* is captured separately by
        # success / parse_failure, so selectivity must not double-penalize a
        # non-repair as if it had rewritten everything.
        return {
            "preservation": 1.0,
            "unnecessary_changes_count": 0,
            "unnecessary_changes_rate": 0.0,
            "harmful_changes_count": 0,
            "harmful_changes_rate": 0.0,
        }

    # preservation = mean over non-implicated factors of (initial == revised).
    if non_implicated:
        preserved = sum(
            1 for f in non_implicated
            if getattr(initial, f) == getattr(revised, f)
        )
        preservation = preserved / len(non_implicated)
    else:
        preservation = 1.0

    changed = _changed_factors(initial, revised, factor_menu)

    unnecessary = [f for f in changed if f != implicated_factor]
    unnecessary_count = len(unnecessary)
    unnecessary_rate = (
        unnecessary_count / (len(factor_menu) - 1)
        if len(factor_menu) > 1 else 0.0
    )

    # Harmful = collateral damage on a FROZEN (non-implicated) factor that was
    # correct and got moved away from ground truth. The implicated factor is
    # EXCLUDED: changing it is the intended repair, and for blocked-side / frame
    # failures the operationally-correct revised value legitimately differs from
    # the naive oracle_correct_intent token (e.g. approach_direction rerouted
    # around an obstacle), which must not be scored as harm.
    harmful = [
        f for f in factor_menu
        if f != implicated_factor
        and getattr(initial, f) == getattr(gt, f)
        and getattr(revised, f) != getattr(gt, f)
    ]
    harmful_count = len(harmful)
    harmful_rate = harmful_count / len(factor_menu) if factor_menu else 0.0

    return {
        "preservation": preservation,
        "unnecessary_changes_count": unnecessary_count,
        "unnecessary_changes_rate": unnecessary_rate,
        "harmful_changes_count": harmful_count,
        "harmful_changes_rate": harmful_rate,
    }
