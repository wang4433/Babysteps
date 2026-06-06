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

    def decode(self, feature: np.ndarray) -> Intent:
        """Decode the grounded factors from `feature`; keep template constants."""
        z = (np.asarray(feature, dtype=np.float32) - self.mean) / self.scale
        with torch.no_grad():
            G = self.intent_head(torch.from_numpy(z).unsqueeze(0))[0].numpy()
        intent = self.template
        for fi, per_class in self.centroids.items():
            cls_int = decode_slot(G[fi], per_class)
            token = self.label_tokens[fi][cls_int]
            intent = replace(intent, **{INTENT_FIELDS[fi]: token})
        return intent

    def decode_from_cache(self, features_dir, seed: int, motion: str,
                          suffix: str) -> Intent:
        """Load the cached (seed, motion) demo feature and decode it."""
        path = demo_feature_path(features_dir, seed, motion, suffix)
        if not path.exists():
            raise FileNotFoundError(
                f"missing demo feature for seed {seed} motion {motion}: {path}")
        return self.decode(np.load(path))
