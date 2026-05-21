"""Stage-0 procedural retry policies for the M3 baseline table.

Each policy is a pure function `(RetryContext) -> Optional[(Intent, Revision)]`.
Returning None means "no retry" (the one_shot baseline). All policies are
sim-free and import no simulator. See
docs/superpowers/specs/2026-05-20-stage0-baselines-design.md.

These are DETERMINISTIC PROCEDURAL ANALOGUES of replanning, not LLM/VLM
planners. full_replan_analogue / text_feedback_replan must always be reported
as procedural analogues.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Optional

from babysteps.failure import Attribution
from babysteps.schemas import INTENT_FIELDS, Intent, Revision, SceneState


@dataclass(frozen=True)
class RetryContext:
    """Everything a retry policy needs. Built once per episode by run_episode."""

    initial_intent: Intent
    attribution: Attribution
    scene: SceneState
    oracle_correct_intent: Intent
    oracle_wrong_factor: str
    task_valid_tokens: Mapping[str, tuple[str, ...]]
    rng: random.Random
    # adapter.revise_intent, bound — used by selective/oracle policies so this
    # module never imports the adapter (avoids an import cycle).
    revise_fn: Callable[[Intent, Attribution, SceneState], tuple[Intent, Revision]]


def resample_factor(
    intent: Intent,
    factor: str,
    valid_tokens: tuple[str, ...],
    rng: random.Random,
) -> str:
    """Return a task-valid token for `factor` other than its current value.

    Excludes the CURRENT value only (not the oracle value) — see spec §2:
    this lets random_factor_revision occasionally land on the correct value,
    while extra perturbations of already-correct factors are necessarily wrong.
    """
    current = getattr(intent, factor)
    alternatives = [t for t in valid_tokens if t != current]
    if not alternatives:
        raise ValueError(
            f"resample_factor: no task-valid alternative for {factor!r} "
            f"(current={current!r}, tokens={valid_tokens!r})"
        )
    return rng.choice(alternatives)


RetryPolicy = Callable[[RetryContext], Optional[tuple[Intent, Revision]]]


def one_shot(ctx: RetryContext) -> Optional[tuple[Intent, Revision]]:
    """No retry — the lower-bound baseline."""
    return None


def same_intent_retry(ctx: RetryContext) -> Optional[tuple[Intent, Revision]]:
    """Retry the identical intent (a fresh rollout may recover by luck)."""
    rev = Revision(
        operator="same_intent_retry",
        factor="none",
        old_value="",
        new_value="",
        frozen_factors=INTENT_FIELDS,
    )
    return ctx.initial_intent, rev
