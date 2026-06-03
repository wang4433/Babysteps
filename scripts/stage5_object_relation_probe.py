"""Stage-5 Step-2 — StackCube object-centric VISUAL relation probe (CPU, sim-free).

Reads the object-token npz produced by ``stage5_extract_object_patch_tokens.py``
and asks the one genuinely non-circular question Step-1 could not:

    Do frozen DINOv2 patch tokens, pooled at the two cube locations (location
    used ONLY to *select* patches, NEVER as a feature value), recover
    ``object_motion`` better than global mean-pooling (the 0.42 baseline)?

The StackCube label is ``goal_direction_to_motion(cubeB_init - cubeA_init)`` —
a cardinal bin of the two cubes' resting positions. So ANY coordinate (world xy
*or* image uv) just re-feeds the label's own input. Therefore:

  * HEADLINE rungs are **appearance-only** (``A_tok``, ``B_tok``,
    ``B_tok - A_tok``) — pooled DINO token VALUES, no coordinates. The headline
    gate is ``B_tok - A_tok`` vs 0.42 (target >0.60, strong >0.80, oracle ~0.94).
  * A **random-location** control (tokens pooled at random patches) isolates
    whether *object* selection matters, or whether any lift is just DINO
    positional-encoding bleed.
  * The **image-uv** rung (centroid delta) is reported ONLY as a labeled,
    near-tautological UPPER BOUND — it is the image-space analogue of the
    Step-1 position oracle, not a representation result.

Probe protocol matches the 0.42 cell exactly (``stage5_p1_g1_cert.py``):
IntentHead F=6, d_slot=32, n_epochs=300, lr=1e-2, per-fold nested CV; plus a
direct StandardScaler+LR column. Localization is PIXEL-derived colour blobs
(``babysteps.stage5.object_blobs``) — on the deployable path, no sim privilege.

Example::

    python scripts/stage5_object_relation_probe.py \\
        --tokens datasets/stage5/object_relation/StackCube-v1/object_tokens.npz \\
        --out-dir reports/stage5/object_relation_probe/ --radius 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.schemas import INTENT_FIELDS  # noqa: E402
from babysteps.stage4.intent_head import nested_cv_probe_one_factor  # noqa: E402
from babysteps.stage5.object_blobs import (  # noqa: E402
    pixel_to_patch_rc, pool_patch_window,
)
from scripts.stage5_relation_oracle_probe import _direct_lr_probe, _gate  # noqa: E402

_OBJECT_MOTION_IDX = INTENT_FIELDS.index("object_motion")
_DINOV2_BASELINE = 0.42


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #


def _pool_at_centroids(
    patch_grids: np.ndarray, centroids: np.ndarray,
    *, grid: int, img_size: int, radius: int,
) -> np.ndarray:
    """(n, N, d) grids + (n, 2) pixel centroids -> (n, d) object-local tokens.

    Rows whose centroid is NaN come back as NaN (caller drops them).
    """
    out = np.full((patch_grids.shape[0], patch_grids.shape[2]), np.nan, np.float32)
    for i in range(patch_grids.shape[0]):
        uv = centroids[i]
        if np.isnan(uv).any():
            continue
        rc = pixel_to_patch_rc((uv[0], uv[1]), img_size=img_size, grid=grid)
        out[i] = pool_patch_window(patch_grids[i], rc, grid=grid, radius=radius)
    return out


def _pool_at_random(
    patch_grids: np.ndarray, *, grid: int, radius: int, seed: int,
) -> np.ndarray:
    """(n, N, d) grids -> (n, d) tokens pooled at a per-row RANDOM patch.

    Control: if this lifts as much as the object-located token, the signal is
    positional-encoding bleed, not object appearance.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((patch_grids.shape[0], patch_grids.shape[2]), np.float32)
    for i in range(patch_grids.shape[0]):
        rc = (int(rng.integers(grid)), int(rng.integers(grid)))
        out[i] = pool_patch_window(patch_grids[i], rc, grid=grid, radius=radius)
    return out


def build_appearance_features(
    A_tok: np.ndarray, B_tok: np.ndarray,
) -> dict[str, np.ndarray]:
    """Appearance-ONLY rungs — pooled token VALUES, never coordinates.

    Coordinates are *near-tautological* here: the StackCube label IS
    ``goal_direction_to_motion(cubeB_init - cubeA_init)``, so feeding image uv or
    world xy would just re-state the label's own input. Hence these rungs carry
    only pooled DINO token values. The leakage test pins the signature to
    ``(A_tok, B_tok)`` so no centroid/box can be concatenated in here, and
    ``_assert_no_coord_leak`` re-checks widths at the call site.
    """
    return {
        "A_tok (cubeA local)": A_tok.astype(np.float32),
        "B_tok (cubeB local)": B_tok.astype(np.float32),
        "[A_tok;B_tok]": np.concatenate([A_tok, B_tok], axis=1).astype(np.float32),
        "B_tok-A_tok (HEADLINE)": (B_tok - A_tok).astype(np.float32),
    }


def build_uv_upper_bound(cA: np.ndarray, cB: np.ndarray) -> np.ndarray:
    """Image-plane centroid delta (cubeB - cubeA), the near-tautological rung.

    This IS the label's input in image space (a near-affine map of world xy);
    reported only as an upper bound, NEVER as a headline.
    """
    return (cB - cA).astype(np.float32)


def _assert_no_coord_leak(feats: dict[str, np.ndarray], token_dim: int) -> None:
    """Defense-in-depth: appearance/control rungs must be token-width only.

    Every non-uv rung is either one token (``token_dim``) or a pair
    (``2*token_dim``). A coordinate concat would show up as ``token_dim + 2``
    (e.g. 770), betraying a latent leak in feature assembly. The uv upper-bound
    rung is built and probed separately, never passed through here.
    """
    for name, Z in feats.items():
        w = int(Z.shape[1])
        if w not in (token_dim, 2 * token_dim):
            raise AssertionError(
                f"feature {name!r} has width {w}, not token_dim {token_dim} nor "
                f"{2 * token_dim}; a coordinate may have leaked into a non-uv rung")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _step2_verdict(headline_mean: float) -> str:
    if headline_mean > 0.80:
        return ("STRONG — object-local frozen DINO tokens recover the relation; "
                "the fix is object-centric POOLING of frozen features (cheap).")
    if headline_mean > 0.60:
        return ("USEFUL — frozen DINO tokens carry the relation when pooled "
                "object-locally; object-centric pooling helps. Push toward the "
                "~0.94 oracle next.")
    if headline_mean > _DINOV2_BASELINE + 0.05:
        return ("WEAK LIFT over the 0.42 baseline. Before concluding, check "
                "localization / patch radius / cube identity, and expand n to "
                "~200 (n=40 may underpower high-dim tokens).")
    return ("NO LIFT over the 0.42 baseline — frozen DINO *appearance* at object "
            "locations does not carry the relation. Points to an explicitly "
            "SPATIAL representation (Transporter keypoints / slot positions / "
            "SORNet), not better appearance features.")


def _write_report(out_dir: Path, *, task: str, n: int, n_dropped: int,
                  label_dist: dict, radius: int, results: dict,
                  uv_result: dict, headline_key: str,
                  d_slot: int = 32, n_epochs: int = 300) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    majority = max(label_dist.values()) / n
    headline = results[headline_key]["intent"]["probe_acc_mean"]

    groups = {
        "global_dino (768)": "baseline",
        "A_tok (cubeA local)": "appearance",
        "B_tok (cubeB local)": "appearance",
        "[A_tok;B_tok]": "appearance",
        "B_tok-A_tok (HEADLINE)": "appearance",
        "A_tok@rand": "control",
        "[rand;rand]": "control",
        "B_tok-A_tok @rand": "control",
    }

    L = [
        f"# Stage-5 Step-2 — StackCube object-centric VISUAL relation probe — {task}",
        "",
        "**Question:** do frozen DINOv2 patch tokens, pooled at the two cube",
        "locations (location selects patches, never used as a feature value),",
        f"recover `object_motion` better than global mean-pooling (**{_DINOV2_BASELINE:.2f}**)?",
        "",
        f"- n={n} (dropped {n_dropped} for missing blob), label dist {label_dist} "
        f"(majority {majority:.3f})",
        f"- Localization: pixel colour-blob centroids (cubeA=red, cubeB=green); "
        f"patch pooling radius={radius}.",
        f"- Probe: same protocol as the 0.42 cell (IntentHead F=6, d_slot={d_slot}, "
        f"n_epochs={n_epochs}) + direct StandardScaler+LR.",
        "- Label = `goal_direction_to_motion(cubeB_init - cubeA_init)`, so any "
        "COORDINATE re-feeds the label; headline rungs are appearance-only.",
        "",
        "## Feature ladder",
        "",
        "| feature | group | dim | majority | shuffled | direct LR ± std | "
        "IntentHead-CV ± std | gate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        d, h = r["direct"], r["intent"]
        L.append(
            f"| `{name}` | {groups.get(name, '')} | {r['dim']} | "
            f"{d['majority_class_acc']:.3f} | {d['shuffled_features_acc']:.3f} | "
            f"{d['probe_acc_mean']:.3f} ± {d['probe_acc_std']:.3f} | "
            f"{h['probe_acc_mean']:.3f} ± {h['probe_acc_std']:.3f} | "
            f"{_gate(h['probe_acc_mean'], h['majority_class_acc'], h['shuffled_features_acc'])} |"
        )
    uh = uv_result["intent"]
    L.append(
        f"| `uv(B-A) image` | UPPER-BOUND | {uv_result['dim']} | "
        f"{uh['majority_class_acc']:.3f} | {uh['shuffled_features_acc']:.3f} | "
        f"{uv_result['direct']['probe_acc_mean']:.3f} ± "
        f"{uv_result['direct']['probe_acc_std']:.3f} | "
        f"{uh['probe_acc_mean']:.3f} ± {uh['probe_acc_std']:.3f} | "
        "(near-tautological) |"
    )

    rand_rel = results.get("B_tok-A_tok @rand", {}).get("intent", {}).get(
        "probe_acc_mean", float("nan"))
    L += [
        "",
        "## Verdict",
        "",
        f"- **Headline** (`{headline_key}`, IntentHead-CV): **{headline:.3f}** "
        f"vs baseline **{_DINOV2_BASELINE:.2f}**, majority **{majority:.3f}**, "
        f"image-uv upper bound **{uh['probe_acc_mean']:.3f}**.",
        f"- {_step2_verdict(headline)}",
        f"- **Random-location control** relation = {rand_rel:.3f}: if this ≈ the "
        "object-located headline, the lift is DINO positional-encoding bleed, "
        "not object appearance; if the headline ≫ control, object selection "
        "carries the signal.",
        "",
        "## Caveats",
        "",
        "- **The claim is object-local pooling vs GLOBAL pooling** (both frozen "
        "DINOv2). If the headline ≫ 0.42, global mean-pooling destroys a relation "
        "that object-local pooling preserves → object-centric pooling is the fix. "
        "This holds whether the retained signal is cube *appearance* or DINOv2 "
        "*positional encoding* at the selected patches.",
        "- **Appearance vs positional encoding:** a lift could be pos-encoding "
        "retained at the object patches, not appearance (both are 'object-located'; "
        "the random control only rules out *arbitrary*-location signal). Read the "
        f"headline against the `uv(B-A)` rung ({uh['probe_acc_mean']:.3f}): if "
        "headline ≈ uv, the signal is largely object *location* (which a detector "
        "provides) — still supports object-centric pooling, but is not a claim "
        "that appearance alone encodes the relation.",
        "- Headline rungs contain pooled token VALUES only — no centroid / box / "
        "coordinate is fed to the probe (signature-pinned + width-asserted). The "
        "`uv(B-A)` rung is the lone coordinate feature, flagged near-tautological.",
        "- Localization is pixel-derived (deployable path); only patch SELECTION "
        "uses it. DINOv2 patch tokens carry some positional encoding, so the "
        "random-location control is the guard against reading that as success.",
        f"- n={n} is small for {results['A_tok (cubeA local)']['dim']}-dim tokens; "
        "if the headline is an ambiguous WEAK LIFT, expand to n≈200 before "
        "swapping representations.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    (out_dir / "results.json").write_text(json.dumps({
        "task": task, "n": n, "n_dropped": n_dropped, "radius": radius,
        "label_dist": label_dist, "dinov2_baseline": _DINOV2_BASELINE,
        "headline_key": headline_key,
        "headline_intent_acc": headline,
        "uv_upper_bound": {"direct": uv_result["direct"], "intent": uv_result["intent"],
                           "dim": uv_result["dim"]},
        "results": results,
    }, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _probe_both(Z: np.ndarray, y: np.ndarray, *, d_slot: int, n_epochs: int,
                seed: int) -> dict:
    direct = _direct_lr_probe(Z, y, seed=seed)
    intent = nested_cv_probe_one_factor(
        Z, y, factor_idx=_OBJECT_MOTION_IDX, d_slot=d_slot,
        n_epochs=n_epochs, seed=seed,
    )
    return {"dim": int(Z.shape[1]), "direct": direct, "intent": intent}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=Path, required=True,
                   help="object_tokens.npz from stage5_extract_object_patch_tokens.py")
    p.add_argument("--task", default="StackCube-v1")
    p.add_argument("--out-dir", type=Path,
                   default=Path("reports/stage5/object_relation_probe"))
    p.add_argument("--radius", type=int, default=1, help="patch-window pool radius")
    p.add_argument("--d-slot", type=int, default=32)
    p.add_argument("--n-epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    z = np.load(args.tokens, allow_pickle=False)
    grid = int(z["grid"]); img_size = int(z["img_size"])
    labels_all = [str(s) for s in z["labels"]]
    patch_start = z["patch_start"]
    cA, cB = z["centroid_A_start"], z["centroid_B_start"]

    A_tok = _pool_at_centroids(patch_start, cA, grid=grid, img_size=img_size,
                               radius=args.radius)
    B_tok = _pool_at_centroids(patch_start, cB, grid=grid, img_size=img_size,
                               radius=args.radius)

    # Drop rows with a missing blob in either cube (NaN token).
    valid = ~(np.isnan(A_tok).any(axis=1) | np.isnan(B_tok).any(axis=1))
    n_dropped = int((~valid).sum())
    A_tok, B_tok = A_tok[valid], B_tok[valid]
    cA_v, cB_v = cA[valid], cB[valid]
    glob = z["global_spatial_mean"][valid]
    grids_v = patch_start[valid]
    y_str = [labels_all[i] for i in range(len(labels_all)) if valid[i]]

    classes = sorted(set(y_str))
    y = np.asarray([classes.index(v) for v in y_str], dtype=np.int64)
    label_dist = {c: int((np.asarray(y_str) == c).sum()) for c in classes}
    n = len(y_str)
    print(f"loaded n={n} (dropped {n_dropped}); labels {label_dist}; "
          f"grid={grid} img={img_size} radius={args.radius}")

    rand_A = _pool_at_random(grids_v, grid=grid, radius=args.radius, seed=args.seed)
    rand_B = _pool_at_random(grids_v, grid=grid, radius=args.radius, seed=args.seed + 1)

    feats: dict[str, np.ndarray] = {"global_dino (768)": glob.astype(np.float32)}
    feats.update(build_appearance_features(A_tok, B_tok))
    feats["A_tok@rand"] = rand_A.astype(np.float32)
    feats["[rand;rand]"] = np.concatenate([rand_A, rand_B], axis=1).astype(np.float32)
    feats["B_tok-A_tok @rand"] = (rand_B - rand_A).astype(np.float32)
    _assert_no_coord_leak(feats, int(A_tok.shape[1]))

    results: dict[str, dict] = {}
    for name, Z in feats.items():
        results[name] = _probe_both(Z, y, d_slot=args.d_slot,
                                    n_epochs=args.n_epochs, seed=args.seed)
        r = results[name]
        print(f"  {name:24s} dim={r['dim']:4d}  direct={r['direct']['probe_acc_mean']:.3f}"
              f"  intentCV={r['intent']['probe_acc_mean']:.3f}")

    uv = build_uv_upper_bound(cA_v, cB_v)
    uv_result = _probe_both(uv, y, d_slot=args.d_slot, n_epochs=args.n_epochs,
                            seed=args.seed)
    print(f"  {'uv(B-A) image [upper]':24s} dim={uv_result['dim']:4d}  "
          f"direct={uv_result['direct']['probe_acc_mean']:.3f}  "
          f"intentCV={uv_result['intent']['probe_acc_mean']:.3f}")

    _write_report(args.out_dir, task=args.task, n=n, n_dropped=n_dropped,
                  label_dist=label_dist, radius=args.radius, results=results,
                  uv_result=uv_result, headline_key="B_tok-A_tok (HEADLINE)",
                  d_slot=args.d_slot, n_epochs=args.n_epochs)
    print(f"\nwrote {args.out_dir}/report.md\nwrote {args.out_dir}/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
