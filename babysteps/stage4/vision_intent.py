"""Stage-5 — VisionIntentExtractor: decode a Stage-0 Intent straight from a
frozen-encoder demo feature, replacing the scripted `scripted_demo_to_intent`.

This is the top of the full-vision closed loop:

    demo RGB clip -> [frozen encoder (DINOv2 / V-JEPA 2.1)] -> feature z
      -> [StandardScaler]            (essential for V-JEPA; ~no-op for DINOv2)
      -> [IntentHead]                -> slot tensor G (F, d_slot)
      -> [per-factor nearest-centroid] -> grounded factor tokens
      -> Intent (grounded factors decoded; task-constant factors from a template)

Only the factors the pack actually grounds (centroids present — for PushCube:
object_motion, contact_region, approach_direction) are vision-decoded; the
task-constant factors (goal_state, constraint_region, embodiment_mapping,
direction_grounding — `trivially_constant` in the cert) come from the template,
because they carry no per-episode pixel signal. The push controller acts on the
decoded `contact_region`, so this fully replaces the JSON intent for the loop.

The StandardScaler is persisted in the pack (`scaler.npz`: mean/scale) by
`stage5_train_4way_pack.py --standardize`; when absent the extractor uses an
identity transform. This fixes the V-JEPA feature-norm underfit (DINOv2 norm ~24
vs V-JEPA ~43 -> un-normalized Adam@1e-2 collapses V-JEPA to ~0.54;
standardized ~0.86; see reports/stage5/vjepa_object_motion/FINDINGS.md).

Sim-free: numpy + CPU torch on a cached feature; the encoder forward (frames -> z)
is the offline GPU step (scripts/stage5_cache_{dinov2,vjepa}.py).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from babysteps.schemas import INTENT_FIELDS, Intent
from babysteps.stage4.latent_policy import load_latent_pack
from babysteps.stage4.slot_decode import decode_slot

# motion token -> short, path-safe filename tag (avoids '+' in feature filenames).
MOTION_TAG: dict[str, str] = {
    "translate_+x": "px", "translate_-x": "nx",
    "translate_+y": "py", "translate_-y": "ny",
}
TAG_MOTION: dict[str, str] = {v: k for k, v in MOTION_TAG.items()}


def demo_feature_path(features_dir, seed: int, motion: str, suffix: str) -> Path:
    """Path of the cached per-(seed, direction) demo feature.

    Matches the `--each-seed-all-dirs` render -> cache layout:
    `seed_NNNN_<tag>.npz` -> `seed_NNNN_<tag>_<suffix>.npy`."""
    return Path(features_dir) / f"seed_{seed:04d}_{MOTION_TAG[motion]}_{suffix}.npy"


@dataclass
class VisionIntentExtractor:
    """Frozen-encoder feature -> Stage-0 Intent via the trained pack."""

    intent_head: torch.nn.Module
    centroids: dict[int, dict[int, np.ndarray]]
    label_tokens: dict[int, tuple[str, ...]]
    mean: np.ndarray          # StandardScaler mean (0 if identity)
    scale: np.ndarray         # StandardScaler scale (1 if identity)
    template: Intent          # task-constant factors live here

    @classmethod
    def from_pack(cls, pack_dir, template: Intent) -> "VisionIntentExtractor":
        pack_dir = Path(pack_dir)
        pack = load_latent_pack(pack_dir)
        scaler_path = pack_dir / "scaler.npz"
        if scaler_path.exists():
            s = np.load(scaler_path)
            mean = s["mean"].astype(np.float32)
            scale = s["scale"].astype(np.float32)
        else:
            # Identity (DINOv2 pack trained on raw features).
            z_dim = pack.intent_head.net[0].in_features
            mean = np.zeros(z_dim, dtype=np.float32)
            scale = np.ones(z_dim, dtype=np.float32)
        return cls(
            intent_head=pack.intent_head,
            centroids=pack.centroids,
            label_tokens=pack.label_tokens,
            mean=mean, scale=scale, template=template,
        )

    def _forward_G(self, feature: np.ndarray) -> np.ndarray:
        """Frozen-encoder feature -> slot tensor G (F, d_slot), scaler applied."""
        z = (np.asarray(feature, dtype=np.float32) - self.mean) / self.scale
        with torch.no_grad():
            return self.intent_head(torch.from_numpy(z).unsqueeze(0))[0].numpy()

    def decode(self, feature: np.ndarray) -> Intent:
        """Decode the grounded factors from `feature`; keep template constants."""
        G = self._forward_G(feature)
        intent = self.template
        for fi, per_class in self.centroids.items():
            cls_int = decode_slot(G[fi], per_class)
            token = self.label_tokens[fi][cls_int]
            intent = replace(intent, **{INTENT_FIELDS[fi]: token})
        return intent

    def decode_factor(self, feature: np.ndarray, factor_idx: int) -> str:
        """Decode the token of ONE grounded factor from `feature`.

        The single-factor primitive the dual-view router calls to take a factor
        from the view that actually sees it (e.g. contact_region from the
        contact view, object_motion from the global view). Errors if this view's
        pack does not ground `factor_idx`.
        """
        if factor_idx not in self.centroids:
            raise KeyError(
                f"factor_idx {factor_idx} ({INTENT_FIELDS[factor_idx]}) is not "
                f"grounded by this pack (grounds {sorted(self.centroids)})")
        G = self._forward_G(feature)
        cls_int = decode_slot(G[factor_idx], self.centroids[factor_idx])
        return self.label_tokens[factor_idx][cls_int]

    def decode_from_cache(self, features_dir, seed: int, motion: str,
                          suffix: str) -> Intent:
        """Load the cached (seed, motion) demo feature and decode it."""
        path = demo_feature_path(features_dir, seed, motion, suffix)
        if not path.exists():
            raise FileNotFoundError(
                f"missing demo feature for seed {seed} motion {motion}: {path}")
        return self.decode(np.load(path))


@dataclass
class DualViewIntentExtractor:
    """Route each grounded factor to the demo VIEW that sees it.

    The Stage-5 dual-camera setup (Q1 decision — proxy preserved): TWO
    third-person demo views feed intent extraction.

      * a high-oblique GLOBAL view grounds the whole-scene factors that a single
        view reads fine (PushCube: object_motion, approach_direction; StackCube
        adds goal_state once the oblique clears the gripper);
      * a second external CONTACT view grounds contact_region — the factor a low
        oblique can lose to gripper self-occlusion at the contact instant
        (the same occlusion that makes PickCube contact_region a boundary point).

    Each view has its OWN trained pack (a `VisionIntentExtractor`); this class is
    a thin router. Per factor it decodes from the assigned view and merges into
    one Intent; task-constant factors come from the shared template. The routing
    map is the paper's explicit "which camera grounds which factor" claim, and
    keeps the single-factor / per-factor-observability story legible — unlike a
    concat fusion, which would entangle the views.

    NOTE on the firewall (CLAUDE.md #3/#4): both views are passive third-person
    demo cameras (a human could provide them); no wrist/egocentric stream and no
    sim state enter this path. The wrist camera is execution-side only.
    """

    views: dict[str, VisionIntentExtractor]
    routing: dict[str, str]   # factor_name -> view_name
    template: Intent

    def __post_init__(self):
        for factor_name, view_name in self.routing.items():
            if factor_name not in INTENT_FIELDS:
                raise KeyError(f"routing references unknown factor {factor_name!r}")
            if view_name not in self.views:
                raise KeyError(
                    f"routing sends {factor_name!r} to unknown view {view_name!r} "
                    f"(views: {sorted(self.views)})")
            fi = INTENT_FIELDS.index(factor_name)
            if fi not in self.views[view_name].centroids:
                raise KeyError(
                    f"view {view_name!r} pack does not ground {factor_name!r} "
                    f"(grounds factors {sorted(self.views[view_name].centroids)})")

    @classmethod
    def from_packs(cls, packs: dict, routing: dict, template: Intent
                   ) -> "DualViewIntentExtractor":
        """Build from {view_name: pack_dir} + {factor_name: view_name} routing."""
        views = {
            name: VisionIntentExtractor.from_pack(p, template)
            for name, p in packs.items()
        }
        return cls(views=views, routing=routing, template=template)

    def decode(self, features_by_view: dict) -> Intent:
        """Decode each routed factor from its view's feature; merge into one Intent."""
        intent = self.template
        for factor_name, view_name in self.routing.items():
            if view_name not in features_by_view:
                raise KeyError(f"missing feature for view {view_name!r}")
            fi = INTENT_FIELDS.index(factor_name)
            token = self.views[view_name].decode_factor(
                features_by_view[view_name], fi)
            intent = replace(intent, **{factor_name: token})
        return intent

    def decode_from_cache(self, features_dirs: dict, seed: int, motion: str,
                          suffix: str) -> Intent:
        """Load each routed view's cached (seed, motion) feature and decode.

        `features_dirs` maps view_name -> that view's feature directory.
        """
        feats = {}
        for view_name in set(self.routing.values()):
            if view_name not in features_dirs:
                raise KeyError(f"missing features_dir for view {view_name!r}")
            path = demo_feature_path(features_dirs[view_name], seed, motion, suffix)
            if not path.exists():
                raise FileNotFoundError(
                    f"missing {view_name} demo feature for seed {seed} "
                    f"motion {motion}: {path}")
            feats[view_name] = np.load(path)
        return self.decode(feats)
