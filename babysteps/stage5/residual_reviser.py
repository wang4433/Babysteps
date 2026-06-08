"""Stage-5 — residual-conditioned slot editor (value-only, sim-free).

This is the PushCube *interim* slot-local editor used by the unified main-table
evaluator's ``vlm_diagnosis_local_edit`` condition. It is the learned
counterpart of the ``feedback_residual`` hand rule: a residual-conditioned
:class:`~babysteps.stage4.revise_head.ReviseHead` edits ONE slot in the
vision-grounded latent space and nearest-centroid-decodes the corrected token.

Two design points matter for the elevated research target (a *shared* revision
policy, see ``redesign_failure_paradigm.md``):

* **Value-only.** :class:`ResidualSlotEditor` maps ``(current_value, residual,
  predicate)`` → a corrected token. It never sees the full ``Intent``, the raw
  scene, the task id, or the ground truth — the evaluator computes the
  observable residual (``e_fail``) and passes only that. This keeps the editor
  behind the same no-leakage boundary the shared policy will use.
* **One slot.** The underlying ReviseHead's forward shapes already forbid
  reading/writing more than one slot; the editor returns a single token and the
  evaluator-side compiler enforces exactly-one changed factor.

``_observed_residual`` lives here (relocated from
``scripts/stage5_natural_loop_eval.py``) so the natural-loop script, the unified
evaluator, and the G3 certification consume the identical non-privileged signal.

Sim-free, CPU-only torch. The real run loads a trained pack + residual head via
:meth:`ResidualSlotEditor.from_pack`; tests construct a tiny synthetic editor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from babysteps.stage4.revise_head import (
    ReviseHead,
    vectorize_failure_packet_residual,
)
from babysteps.stage4.slot_decode import decode_slot


def _observed_residual(fp, scene) -> np.ndarray:
    """The goal-relative residual = goal - final_cube, both observable in the
    robot's exec view (final = cube0 + observed displacement).

    Non-privileged: the executing robot observes its own goal and the cube's
    final position at exec time; this is execution feedback, not demo-path
    privilege (CLAUDE.md invariant #4). Shared by the ``feedback_residual`` hand
    rule, the learned :class:`ResidualSlotEditor`, and the G3 violation scoring
    so they read the identical signal."""
    vec = fp.object_displacement_vec or (0.0, 0.0)
    final = np.asarray(scene.cube_xy, dtype=np.float64) + np.asarray(
        vec, dtype=np.float64)
    return np.asarray(scene.goal_xy, dtype=np.float64) - final


class ResidualSlotEditor:
    """Value-only residual-conditioned slot editor for ONE factor.

    ``__call__(current_value, residual_xy, predicate)`` returns the corrected
    token, or ``None`` if ``current_value`` is outside the learned latent
    vocabulary (the caller then leaves the slot unchanged).

    Parameters
    ----------
    factor:
        The factor name this editor revises (e.g. ``"contact_region"``).
    centroids:
        ``{class_idx: centroid (d_slot,)}`` for this factor — the
        vision-grounded latent representatives, used both to look up the stale
        value's slot vector and to nearest-centroid-decode the revised one.
    tokens:
        ``(token0, token1, ...)`` indexed by class.
    head:
        A residual-conditioned :class:`ReviseHead`
        (``fp_dim = FP_VECTOR_DIM_RESIDUAL``).
    """

    def __init__(self, *, factor: str, centroids: dict[int, np.ndarray],
                 tokens: tuple[str, ...], head: ReviseHead) -> None:
        self.factor = factor
        self.centroids = {int(k): np.asarray(v, dtype=np.float32)
                          for k, v in centroids.items()}
        self.tokens = tuple(tokens)
        self.head = head.eval()
        self._tok2cls = {t: i for i, t in enumerate(self.tokens)}

    def __call__(self, current_value: str, residual_xy,
                 predicate: Optional[str]) -> Optional[str]:
        if current_value not in self._tok2cls:
            return None
        g_slot = self.centroids[self._tok2cls[current_value]]
        rec = {
            "revision": {"factor": self.factor},
            "failure_packet": {
                "failure_predicate": predicate or "direction_error"},
        }
        fp_vec = vectorize_failure_packet_residual(rec, residual_xy)
        with torch.no_grad():
            g_rev = self.head(
                torch.tensor(g_slot, dtype=torch.float32).unsqueeze(0),
                torch.tensor(fp_vec, dtype=torch.float32).unsqueeze(0),
            ).numpy()[0]
        cls = decode_slot(g_rev, self.centroids)
        return self.tokens[cls]

    @classmethod
    def from_pack(cls, pack_dir, residual_head_path, *,
                  factor: str = "contact_region") -> "ResidualSlotEditor":
        """Load a trained 4-way LatentPack + standalone residual head.

        Used by the GPU run; the artifacts (``models/``) are gitignored, so
        sim-free tests construct a synthetic editor directly instead."""
        from babysteps.schemas import INTENT_FIELDS
        from babysteps.stage4.latent_policy import load_latent_pack
        from babysteps.stage4.revise_head import load_revise_head

        pack = load_latent_pack(pack_dir)
        head = load_revise_head(Path(residual_head_path))
        fi = INTENT_FIELDS.index(factor)
        if fi not in pack.centroids:
            raise ValueError(f"pack {pack_dir} has no centroids for {factor}")
        return cls(factor=factor, centroids=pack.centroids[fi],
                   tokens=pack.label_tokens[fi], head=head)
