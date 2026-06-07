"""Stage-5 — typed revision-decision interface (PURE, sim-free).

This module factors the babystep loop's *revision step* into two independent
roles, so the unified main-table evaluator can compare conditions fairly and so
the eventual **shared, task-general ``RevisionPolicy``** (see
``redesign_failure_paradigm.md`` → "Shared policy interface") is a drop-in:

* **Attribution** (``Attributor``) selects WHICH factor to edit.
* **Revision**   (``RevisionPolicy``) selects the typed VALUE/edit for that
  factor — it never re-chooses the factor.

Hard boundaries (enforced by the dataclass shapes, not by convention):

* A ``RevisionPolicy`` sees only a :class:`RevisionRequest` — the diagnosed
  ``factor``, its ``current_value``/``g_i``, an observable failure-evidence
  embedding ``e_fail``, observable context ``z``, and the typed ``candidates``.
  It does **not** receive the task id, the full initial ``Intent``, the raw
  scene, the ground truth, or the oracle factor. That prevents task
  memorisation and oracle leakage when we later train one policy across tasks.
* A ``RevisionPolicy`` returns a :class:`RevisionDecision` (a factor + chosen
  value/operator args), **not** a rewritten ``Intent``. The evaluator-side
  :func:`compile_single_slot_edit` applies it and enforces exactly one changed
  slot — a policy cannot smuggle a multi-slot rewrite.

The privileged conditions (oracle upper bound; VLM baselines) are explicitly
allowed evaluator-side knowledge: :class:`OracleAttributor` /
:class:`OracleValuePolicy` receive the implicated factor / gt value injected by
the evaluator (never through ``RevisionRequest``), and :class:`VLMAttributor`
is a task-aware teacher/baseline, not the model whose generalisation we claim.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Optional, Protocol, runtime_checkable

from babysteps.schemas import (
    APPROACH_DIRECTIONS,
    CONSTRAINT_REGIONS,
    CONTACT_REGIONS,
    DIRECTION_GROUNDINGS,
    EMBODIMENT_MAPPINGS,
    GOAL_STATES,
    INTENT_FIELDS,
    OBJECT_MOTIONS,
    Intent,
)

ZERO_COST: dict = {"n_calls": 0, "latency_s": 0.0, "gen_tokens": 0,
                   "input_tokens": 0}

# Per-factor token vocabulary (schema whitelist) — the fallback candidate set
# for ANY diagnosed factor, so a VLM mis-diagnosis still resolves to a typed
# edit rather than crashing (corrections #4).
_FACTOR_VOCAB: dict[str, frozenset] = {
    "goal_state": GOAL_STATES,
    "object_motion": OBJECT_MOTIONS,
    "contact_region": CONTACT_REGIONS,
    "approach_direction": APPROACH_DIRECTIONS,
    "constraint_region": CONSTRAINT_REGIONS,
    "embodiment_mapping": EMBODIMENT_MAPPINGS,
    "direction_grounding": DIRECTION_GROUNDINGS,
}

# Task-scoped candidate VALUES for the factors that actually vary in a task's
# natural loop. Other factors fall back to the full schema vocab above.
_TASK_FACTOR_CANDIDATES: dict[str, dict[str, tuple[str, ...]]] = {
    "PushCube-v1": {
        "contact_region": (
            "minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face"),
    },
    "StackCube-v1": {
        "goal_state": ("cube_at_target", "cubeA_on_cubeB"),
    },
}

# Deterministic typed operators: a scene-free value→value transition table that
# mirrors the Stage-0 deterministic operators in ``babysteps/revision.py``
# (the COMPILER that executes a chosen typed edit; not the policy). Kept here so
# the StackCube interim editor produces a value without the scene/full-intent the
# policy boundary forbids. ``tests/test_stage5_revision_policy.py`` asserts this
# stays consistent with ``revision.revise_intent``.
TYPED_OPERATORS: dict[str, dict[str, str]] = {
    # goal_refinement (StackCube): the single valid strict-extension.
    "goal_refinement": {"cube_at_target": "cubeA_on_cubeB"},
}


def candidates_for(task: str, factor: str) -> tuple[str, ...]:
    """Typed candidate VALUES for ``factor`` in ``task``.

    Task-scoped where the factor varies naturally (e.g. PushCube
    ``contact_region`` = the 4 cube faces); otherwise the full schema vocab so
    any diagnosed factor (incl. a VLM mis-diagnosis) resolves to a typed set."""
    task_map = _TASK_FACTOR_CANDIDATES.get(task, {})
    if factor in task_map:
        return task_map[factor]
    if factor not in _FACTOR_VOCAB:
        raise KeyError(f"unknown factor {factor!r}")
    return tuple(sorted(_FACTOR_VOCAB[factor]))


def _stable_idx(key: object, salt: str, n: int) -> int:
    h = int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:8], 16)
    return h % max(1, n)


# --------------------------------------------------------------------------- #
# Data contracts
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FailureEvidence:
    """Observable failure evidence (``e_fail``). Computed evaluator-side from the
    failure packet + exec observation; carries NO privileged scene/gt state.

    For step 1 this is the failure predicate + the 2D goal-relative residual
    (PushCube). A learned failure encoder replaces it in step 4 without changing
    the policy interface."""
    predicate: Optional[str] = None
    residual_xy: Optional[tuple[float, float]] = None


@dataclass(frozen=True)
class RevisionRequest:
    """MODEL-VISIBLE inputs to a revision policy. Deliberately excludes task id,
    full Intent, scene, gt, and oracle factor."""
    factor: str
    current_value: object
    candidates: tuple[str, ...]
    e_fail: FailureEvidence = field(default_factory=FailureEvidence)
    g_i: Optional[tuple[float, ...]] = None
    z: Optional[Mapping] = None


@dataclass(frozen=True)
class RevisionDecision:
    """A typed, factor-local edit. ``new_value`` is the chosen discrete token;
    ``operator_args`` is reserved for parametric/continuous edits (unused in
    step 1). ``new_value is None`` means "no change"."""
    factor: str
    new_value: Optional[object] = None
    operator_args: Optional[Mapping] = None
    confidence: float = 0.0
    telemetry: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class AttributionResult:
    factor: Optional[str]
    confidence: float = 0.0
    latency_s: float = 0.0
    cost: Mapping = field(default_factory=lambda: dict(ZERO_COST))


@dataclass(frozen=True)
class AttributionObs:
    """Inputs to an attributor. The VLM baseline/teacher is task-aware (it reads
    ``task``/``initial_intent`` for its prompt); the honest random attributor
    uses only ``factor_menu`` + ``key``; the oracle ignores all of it."""
    task: str
    factor_menu: tuple[str, ...]
    failure_predicate: Optional[str]
    initial_intent: Intent
    frame_path: Optional[str] = None
    wrist_frame_path: Optional[str] = None
    key: object = 0


@runtime_checkable
class Attributor(Protocol):
    name: str
    def attribute(self, obs: AttributionObs) -> AttributionResult: ...


@runtime_checkable
class RevisionPolicy(Protocol):
    name: str
    def decide(self, req: RevisionRequest) -> RevisionDecision: ...


# --------------------------------------------------------------------------- #
# Attributors
# --------------------------------------------------------------------------- #

class RandomAttributor:
    """Pick a factor uniformly from the menu (deterministic per episode via
    ``obs.key``). Tests whether DIAGNOSIS matters."""
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def attribute(self, obs: AttributionObs) -> AttributionResult:
        menu = obs.factor_menu
        idx = _stable_idx(obs.key, f"attr:{self.seed}", len(menu))
        return AttributionResult(menu[idx], confidence=1.0 / max(1, len(menu)))


class VLMAttributor:
    """VLM-constrained diagnosis (baseline/teacher). Brackets the call with the
    client's cost meter so the evaluator can charge it to DIAGNOSIS latency."""
    name = "vlm"

    def __init__(self, vlm) -> None:
        self.vlm = vlm

    def attribute(self, obs: AttributionObs) -> AttributionResult:
        reset = getattr(self.vlm, "reset_cost", None)
        snap = getattr(self.vlm, "cost_snapshot", None)
        if reset:
            reset()
        factor = self.vlm.diagnose_constrained(
            task=obs.task, image_path=obs.frame_path,
            initial_intent=obs.initial_intent,
            failure_predicate=obs.failure_predicate,
            wrist_image_path=obs.wrist_frame_path,
        )
        cost = dict(snap()) if snap else dict(ZERO_COST)
        return AttributionResult(factor, confidence=1.0 if factor else 0.0,
                                 latency_s=float(cost.get("latency_s", 0.0)),
                                 cost=cost)


class OracleAttributor:
    """Returns the implicated factor the evaluator already knows. Upper-bound /
    reference only — explicitly privileged, never a primary competitor."""
    name = "oracle"

    def __init__(self, implicated_factor: str) -> None:
        self.implicated_factor = implicated_factor

    def attribute(self, obs: AttributionObs) -> AttributionResult:
        return AttributionResult(self.implicated_factor, confidence=1.0)


# --------------------------------------------------------------------------- #
# Revision policies (value producers)
# --------------------------------------------------------------------------- #

def _first_alternative(current_value: object,
                       candidates: tuple[str, ...]) -> Optional[str]:
    """Deterministic generic fallback: the first candidate != current_value."""
    for c in candidates:
        if c != current_value:
            return c
    return None


class RandomCandidatePolicy:
    """Pick a random candidate value for the diagnosed factor (deterministic per
    request). Never re-chooses the factor."""
    name = "random_candidate"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def decide(self, req: RevisionRequest) -> RevisionDecision:
        alts = tuple(c for c in req.candidates if c != req.current_value)
        pool = alts or req.candidates
        if not pool:
            return RevisionDecision(req.factor, None, confidence=0.0,
                                    telemetry={"policy": self.name})
        idx = _stable_idx((req.factor, req.current_value),
                          f"rev:{self.seed}", len(pool))
        return RevisionDecision(req.factor, pool[idx], confidence=0.0,
                                telemetry={"policy": self.name})


class OracleValuePolicy:
    """Set the diagnosed factor to the privileged oracle value (injected by the
    evaluator at construction; NOT carried in the model-visible request)."""
    name = "oracle_value"

    def __init__(self, oracle_value: object) -> None:
        self.oracle_value = oracle_value

    def decide(self, req: RevisionRequest) -> RevisionDecision:
        return RevisionDecision(req.factor, self.oracle_value, confidence=1.0,
                                telemetry={"policy": self.name})


class PerTaskEditorAdapter:
    """Interim per-task slot editor wrapped behind the policy interface.

    ``producers`` maps a *native* factor → a value-producer ``(req) -> token``
    (PushCube ``contact_region`` = residual head; StackCube ``goal_state`` =
    typed operator). For any other diagnosed factor (e.g. a VLM mis-diagnosis)
    it falls back to a deterministic candidate choice so every factor resolves
    (corrections #4). This is mechanism-validation only — step 2 replaces it
    with one shared policy."""
    name = "per_task_editor"

    def __init__(self, producers: Mapping[str, Callable[[RevisionRequest],
                                                        Optional[str]]]) -> None:
        self.producers = dict(producers)

    def decide(self, req: RevisionRequest) -> RevisionDecision:
        producer = self.producers.get(req.factor)
        if producer is not None:
            new_value = producer(req)
            path = "native"
        else:
            new_value = _first_alternative(req.current_value, req.candidates)
            path = "fallback"
        conf = 1.0 if (new_value is not None
                       and new_value != req.current_value) else 0.0
        return RevisionDecision(req.factor, new_value, confidence=conf,
                                telemetry={"policy": self.name, "path": path})


class SharedRevisionPolicy:
    """Placeholder for the proposed shared, task-general policy (build-order
    step 2). It is NOT registered as a runnable condition — the registry marks
    ``shared_revision_policy`` deferred and never instantiates this — so this
    raising body is reached only if someone wires it in prematurely."""
    name = "shared_revision_policy"

    def decide(self, req: RevisionRequest) -> RevisionDecision:
        raise NotImplementedError(
            "SharedRevisionPolicy is build-order step 2; the shared "
            "factor/value space + pooled checkpoint are not built yet.")


# --------------------------------------------------------------------------- #
# Evaluator-side compiler
# --------------------------------------------------------------------------- #

def compile_single_slot_edit(
    initial: Intent, decision: Optional[RevisionDecision],
    factor_menu: tuple[str, ...] = INTENT_FIELDS,
) -> Intent:
    """Apply a typed :class:`RevisionDecision` to ``initial`` and return the
    revised ``Intent``, enforcing the single-factor invariant: at most the
    decided factor may differ. Raises if a decision would change any other slot.

    ``decision is None`` or ``new_value is None`` → no change (returns
    ``initial`` unchanged)."""
    if decision is None or decision.new_value is None:
        return initial
    if decision.factor not in factor_menu:
        raise ValueError(
            f"decision factor {decision.factor!r} not in menu {factor_menu}")
    if decision.new_value == getattr(initial, decision.factor):
        return initial  # no-op edit → unchanged
    revised = replace(initial, **{decision.factor: decision.new_value})
    changed = tuple(f for f in factor_menu
                    if getattr(initial, f) != getattr(revised, f))
    if any(f != decision.factor for f in changed):
        raise ValueError(
            f"single-slot invariant violated: decision on "
            f"{decision.factor!r} changed {changed}")
    return revised
