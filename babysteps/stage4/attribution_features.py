"""Stage-4 M2.5 — feature extractor for the learned attribution head.

Vectorize the existing FailurePacket fields + the chosen Intent into a
fixed-dim row consumed by `AttributionHead` (see
`babysteps/stage4/attribution_head.py`). The output of the head replaces
the rule-based `babysteps.failure.attribute_failure` for the
`latent_revision` policy only — the Stage-0 baseline still uses the
rule.

Non-privileged by construction: this module reads only

- FailurePacket fields (the rule already consumes these), and
- the agent's chosen Intent (not simulator state).

It does NOT import any simulator (mani_skill / sapien / envs) nor any
GPU/Vulkan code. The static firewall test in
`tests/test_stage4_attribution_features.py` enforces this.

Symmetry: same function works on disk records (dict failure_packet,
dict intent) and on the live runtime path (dataclass FailurePacket,
Intent dataclass) because both expose the same string field names.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

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

# ---- Layout ---------------------------------------------------------- #

# Schema tokens added AFTER the Stage-4 feature layout was frozen (2026-05-30,
# Sub-project D re-grasp). They are valid Intent/predicate values but are kept
# OUT of the learned-attribution feature vocab so FEATURE_DIM, block offsets,
# and the trained model packs stay stable as the schema grows. The encoder maps
# such tokens to an all-zero one-hot (see vectorize_attribution_input); the
# rule-table and VLM attribution paths handle them instead.
FEATURE_FROZEN_EXCLUDE: frozenset[str] = frozenset({
    "proxy_contact_to_franka_regrasp_turn",
    "continuous_rotation_infeasible",
})

# Predicate one-hot, sorted; "none" included so the encoder is
# well-defined on success records too (the runtime caller guards
# against running on success records upstream). Pinned via FEATURE_FROZEN_EXCLUDE.
_PREDICATE_ORDER: tuple[str, ...] = tuple(sorted(FAILURE_PREDICATES - FEATURE_FROZEN_EXCLUDE))

# Per-factor token orders. Tuple-of-sorted gives a stable layout across
# Python versions (frozenset iteration order is not guaranteed); the
# FEATURE_FROZEN_EXCLUDE subtraction keeps the layout pinned across schema growth.
_FACTOR_TOKEN_ORDER: dict[str, tuple[str, ...]] = {
    "goal_state":         tuple(sorted(GOAL_STATES)),
    "object_motion":      tuple(sorted(OBJECT_MOTIONS)),
    "contact_region":     tuple(sorted(CONTACT_REGIONS)),
    "approach_direction": tuple(sorted(APPROACH_DIRECTIONS)),
    "constraint_region":  tuple(sorted(CONSTRAINT_REGIONS)),
    "embodiment_mapping": tuple(sorted(EMBODIMENT_MAPPINGS - FEATURE_FROZEN_EXCLUDE)),
}

# Execution-trace bool keys; same order the rule-based detector emits them.
_TRACE_KEYS: tuple[str, ...] = (
    "reached_contact",
    "object_moved",
    "collision",
    "planner_failed",
    "grasp_slip",
)

_PRED_DIM = len(_PREDICATE_ORDER)
_TRACE_DIM = len(_TRACE_KEYS)
_DISP_DIM = 1
_ALIGN_DIM = 2  # value + present-flag
_INTENT_DIM = sum(len(toks) for toks in _FACTOR_TOKEN_ORDER.values())

FEATURE_DIM: int = _PRED_DIM + _TRACE_DIM + _DISP_DIM + _ALIGN_DIM + _INTENT_DIM

# Public layout constants for downstream callers that need block offsets.
PRED_OH_START: int = 0
TRACE_START: int = _PRED_DIM
DISP_START: int = TRACE_START + _TRACE_DIM
ALIGN_START: int = DISP_START + _DISP_DIM
INTENT_OH_START: int = ALIGN_START + _ALIGN_DIM

PREDICATE_ORDER: tuple[str, ...] = _PREDICATE_ORDER
FACTOR_TOKEN_ORDER: dict[str, tuple[str, ...]] = dict(_FACTOR_TOKEN_ORDER)
TRACE_KEYS: tuple[str, ...] = _TRACE_KEYS


# ---- Public API ------------------------------------------------------ #


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from `obj` regardless of dataclass or Mapping.

    Returns `default` when missing. Older records may omit
    `object_displacement` / `direction_alignment`.
    """
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def vectorize_attribution_input(
    fp_fields: Any,
    intent: Any,
) -> np.ndarray:
    """Return a (FEATURE_DIM,) float64 vector for the attribution head.

    Parameters
    ----------
    fp_fields
        Anything exposing ``failure_predicate`` (str), ``execution_trace``
        (Mapping[str, bool]), ``object_displacement`` (float | None),
        ``direction_alignment`` (float | None). Both the runtime
        ``babysteps.schemas.FailurePacket`` dataclass and the on-disk dict
        form work.
    intent
        Anything exposing the six INTENT_FIELDS as string-valued
        attributes / keys. ``babysteps.schemas.Intent`` dataclass and
        the on-disk ``execution.initial_intent`` dict both work.
    """
    out = np.zeros(FEATURE_DIM, dtype=np.float64)

    pred = _get(fp_fields, "failure_predicate")
    if pred in _PREDICATE_ORDER:
        out[PRED_OH_START + _PREDICATE_ORDER.index(pred)] = 1.0
    elif pred in FAILURE_PREDICATES:
        pass  # valid predicate added after the feature vocab was frozen → zero
    else:
        raise ValueError(f"unknown failure_predicate {pred!r}")

    trace = _get(fp_fields, "execution_trace")
    for i, key in enumerate(_TRACE_KEYS):
        # Older records may not carry every key (None-safe default).
        val = trace.get(key, False) if isinstance(trace, Mapping) else getattr(
            trace, key, False)
        out[TRACE_START + i] = 1.0 if bool(val) else 0.0

    disp = _get(fp_fields, "object_displacement")
    # Clamp into [0, 1]; disp is meters in the sim arena. Defensive
    # against malformed records — None → 0.
    out[DISP_START] = float(np.clip(disp or 0.0, 0.0, 1.0))

    align = _get(fp_fields, "direction_alignment")
    if align is None:
        out[ALIGN_START] = 0.0
        out[ALIGN_START + 1] = 0.0  # not present
    else:
        out[ALIGN_START] = float(align)
        out[ALIGN_START + 1] = 1.0  # present

    cursor = INTENT_OH_START
    for factor in INTENT_FIELDS:
        toks = _FACTOR_TOKEN_ORDER[factor]
        val = _get(intent, factor)
        if val in toks:
            out[cursor + toks.index(val)] = 1.0
        elif factor == "embodiment_mapping" and val in EMBODIMENT_MAPPINGS:
            pass  # token added after the feature vocab was frozen → zero one-hot
        else:
            raise ValueError(
                f"intent factor {factor!r} value {val!r} not in whitelist")
        cursor += len(toks)

    return out
