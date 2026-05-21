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


def babysteps_selective(ctx: RetryContext) -> Optional[tuple[Intent, Revision]]:
    """Ours: revise only the attributed implicated factor."""
    return ctx.revise_fn(ctx.initial_intent, ctx.attribution, ctx.scene)


def oracle_factor_revision(ctx: RetryContext) -> Optional[tuple[Intent, Revision]]:
    """Upper bound: revise the ground-truth wrong factor."""
    factor = ctx.oracle_wrong_factor
    oracle_attr = Attribution(
        semantic_failure=True,
        wrong_factor=factor,
        freeze=tuple(f for f in INTENT_FIELDS if f != factor),
        revise=(factor,),
    )
    return ctx.revise_fn(ctx.initial_intent, oracle_attr, ctx.scene)


def _editable_factors(ctx: RetryContext) -> tuple[str, ...]:
    """Task-editable factors (those with a task-valid token set)."""
    return tuple(f for f in INTENT_FIELDS if f in ctx.task_valid_tokens)


def _perturb(
    intent: Intent, factors: tuple[str, ...], ctx: RetryContext,
) -> Intent:
    """Resample each factor in `factors` to a task-valid non-current token."""
    out = intent
    for f in factors:
        new = resample_factor(out, f, tuple(ctx.task_valid_tokens[f]), ctx.rng)
        out = replace(out, **{f: new})
    return out


def _frozen_against_ground_truth(ctx: RetryContext) -> tuple[str, ...]:
    """The factors that SHOULD be preserved = all but the true wrong factor."""
    return tuple(f for f in INTENT_FIELDS if f != ctx.oracle_wrong_factor)


def random_factor_revision(ctx: RetryContext) -> Optional[tuple[Intent, Revision]]:
    """Ignore attribution: resample one random editable factor."""
    editable = _editable_factors(ctx)
    factor = ctx.rng.choice(editable)
    old = getattr(ctx.initial_intent, factor)
    revised = _perturb(ctx.initial_intent, (factor,), ctx)
    rev = Revision(
        operator="random_factor_revision",
        factor=factor,
        old_value=old,
        new_value=getattr(revised, factor),
        frozen_factors=_frozen_against_ground_truth(ctx),
    )
    return revised, rev


def text_feedback_replan(ctx: RetryContext) -> Optional[tuple[Intent, Revision]]:
    """Fix implicated correctly, then perturb its sibling editable factors
    (attribution.revise minus the implicated factor)."""
    fixed, _ = ctx.revise_fn(ctx.initial_intent, ctx.attribution, ctx.scene)
    siblings = tuple(
        f for f in ctx.attribution.revise
        if f != ctx.attribution.wrong_factor and f in ctx.task_valid_tokens
    )
    revised = _perturb(fixed, siblings, ctx)
    rev = Revision(
        operator="text_feedback_replan",
        factor=ctx.attribution.wrong_factor or "none",
        old_value=getattr(ctx.initial_intent, ctx.attribution.wrong_factor),
        new_value=getattr(revised, ctx.attribution.wrong_factor),
        frozen_factors=_frozen_against_ground_truth(ctx),
    )
    return revised, rev


def full_replan_analogue(ctx: RetryContext) -> Optional[tuple[Intent, Revision]]:
    """Fix implicated correctly, then perturb ALL other editable factors."""
    fixed, _ = ctx.revise_fn(ctx.initial_intent, ctx.attribution, ctx.scene)
    others = tuple(
        f for f in _editable_factors(ctx) if f != ctx.attribution.wrong_factor
    )
    revised = _perturb(fixed, others, ctx)
    rev = Revision(
        operator="full_replan_analogue",
        factor=ctx.attribution.wrong_factor or "none",
        old_value=getattr(ctx.initial_intent, ctx.attribution.wrong_factor),
        new_value=getattr(revised, ctx.attribution.wrong_factor),
        frozen_factors=_frozen_against_ground_truth(ctx),
    )
    return revised, rev


POLICIES: dict[str, RetryPolicy] = {
    "one_shot": one_shot,
    "same_intent_retry": same_intent_retry,
    "random_factor_revision": random_factor_revision,
    "babysteps_selective": babysteps_selective,
    "text_feedback_replan": text_feedback_replan,
    "full_replan_analogue": full_replan_analogue,
    "oracle_factor_revision": oracle_factor_revision,
}
