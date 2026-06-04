"""Stage-5 — PlugCharger-v1 latent-groundability SCOUT (GPU render + CPU probe).

A cheap groundability scout for a candidate SECOND latent end-to-end task (the
first, and so far only, clean one is PushCube). NO BabySteps loop here — the one
decision-relevant question is:

    Does any PlugCharger-v1 randomized factor have a pixel signature that the
    DEPLOYED frozen encoder (DINOv2 ViT-B/14, spatial_mean) can linearly
    separate from a third-person demo frame, clearing the 0.90 IntentHead-CV
    gate — robustly, not just as a clean static artifact (the StackCube
    goal_state lesson: config 0.99 -> deployed-clip 0.65)?

The structural gotcha that pre-decides most of the answer
--------------------------------------------------------
PlugCharger's human render camera is ``mount=self.receptacle`` (plug_charger.py
line 74): the camera is rigid in the receptacle's body frame, so it co-rotates
and co-translates with the receptacle. Consequences:

  * The receptacle renders at a FIXED pixel pose every episode -> its world yaw
    (pi +- 22.5 deg) and world XY carry ~zero image variance.
  * ``goal_pose = receptacle.pose * Rz(pi)`` is rigidly tied to the receptacle
    -> the goal / insertion axis / goal-alignment are ALSO constant in-frame.
  * The peg is sub-cm (half-width 0.75 mm) -> mm-scale insertion alignment is
    below DINOv2 patch resolution.

So everything receptacle/goal-defined is camera-cancelled. What REMAINS visible
and variable is purely CHARGER-relative-to-receptacle. The scout therefore
probes three cells, all on the SAME rendered frames (they differ only in label):

  charger_yaw     (-> object_motion)     PRIMARY CANDIDATE. The asymmetric
                  charger base box (40x30 mm, multi-patch) visibly rotates over
                  +-60 deg; it is NOT camera-cancelled (the mount fixes only the
                  receptacle). INITIAL-STATE: fully visible from frame 0, so a
                  single reset frame IS the deployed signal (no StackCube-style
                  config->clip discount applies to initial-state factors).
  charger_xy      (-> approach_direction) SECONDARY. Optically the largest
                  signal (the charger blob sweeps the frame), but predicted to
                  FAIL under the deployed spatial_mean, which averages out
                  absolute centroid position — the PlugCharger analogue of the
                  StackCube pooling dilution. Informative either way.
  receptacle_yaw  (-> constraint_region) NEGATIVE CONTROL. EXPECTED to read
                  ~chance (the receptacle was assumed camera-cancelled by the
                  mount). FINDING (2026-06-04): it reads ~0.96 instead — the
                  mount makes the BACKGROUND counter-rotate with receptacle yaw,
                  so receptacle yaw leaks into the image globally. This falsifies
                  the cancellation assumption and shows the deployed charger
                  numbers are confounded, which is exactly why ``--fix-receptacle``
                  (below) is the decisive control.

Findings (2026-06-04) — see reports/stage5/plugcharger_probe/SUMMARY.md
----------------------------------------------------------------------
Deployed (mounted camera, background rotates): charger_yaw 0.93/0.91 PASS,
charger_xy 0.86/0.88 FAIL, receptacle_yaw 0.96 PASS (the confound). With
``--fix-receptacle`` (background frozen, only the charger varies): charger_yaw
DROPS to 0.83/0.86 FAIL, charger_xy 0.84(@224)/0.94(@518). Conclusion: the
deployed charger_yaw PASS was largely the rotating background; the charger's own
+-60 deg orientation is NOT reliably groundable by frozen DINOv2 (the asymmetric
base is too small/coarse), and charger position only clears the gate with a
non-default fixed camera at 518. PlugCharger is therefore NOT a clean second
latent task under the deployed encoder; PushCube stays the single clean one.

Label hygiene
-------------
Because the receptacle is constant in-frame, labels are CHARGER-RELATIVE (the
position is de-rotated into the receptacle/camera frame; the yaw is the apparent
camera-frame yaw = charger_world_yaw - receptacle_world_yaw). This is both the
deployment-faithful quantity (what the demo camera actually shows) and a guard
against the probe exploiting a fixed-background artifact. Poses are read ONLY to
author the class labels; the encoder reads pixels and no coordinate is ever fed
to the probe (CLAUDE.md invariant #4).

Example::

    python scripts/stage5_plugcharger_probe.py \\
        --task PlugCharger-v1 --seeds 0-119 \\
        --factors charger_yaw,charger_xy,receptacle_yaw \\
        --resolutions 224,518 \\
        --out-dir reports/stage5/plugcharger_probe

GPU/Vulkan node required (login-node Vulkan init fails).
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
from scripts.stage5_relation_oracle_probe import _direct_lr_probe, _gate  # noqa: E402


# --------------------------------------------------------------------------- #
# Pure geometry (sim-free, tested)
# --------------------------------------------------------------------------- #


def quat_yaw(quat) -> float:
    """z-yaw (rad) from a ``[qw, qx, qy, qz]`` quaternion (ManiSkill raw_pose
    order). The charger/receptacle are locked in x,y rotation, so this reduces
    to the pure in-plane yaw."""
    qw, qx, qy, qz = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def Rz(theta: float) -> np.ndarray:
    """2x2 rotation matrix about +z by ``theta`` radians."""
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def charger_rel_xy(charger_p_xy, recep_p_xy, recep_yaw: float) -> tuple[float, float]:
    """Charger position in the receptacle's (camera) frame.

    ``rel = Rz(-recep_yaw) @ (charger_xy - recep_xy)``. This is the
    deployment-faithful "where the charger appears" coordinate (the camera is
    mounted on the receptacle), and de-rotating prevents the probe from
    exploiting the constant in-frame receptacle/background.
    """
    d = np.asarray(charger_p_xy, dtype=np.float64)[:2] - np.asarray(recep_p_xy, dtype=np.float64)[:2]
    rel = Rz(-recep_yaw) @ d
    return float(rel[0]), float(rel[1])


def charger_rel_yaw(charger_quat, recep_quat) -> float:
    """Charger apparent (camera-frame) in-plane yaw = charger world yaw minus
    receptacle world yaw — exactly what the receptacle-mounted camera renders.

    Receptacle yaw ~pi and charger yaw in [-pi/3, pi/3], so the raw difference
    lands in a contiguous interval (~[-262, -98] deg) with no wrap ambiguity for
    a median split; no angle wrapping is needed.
    """
    return quat_yaw(charger_quat) - quat_yaw(recep_quat)


def median_split_labels(values, *, lo: str, hi: str) -> list[str]:
    """Balanced 2-class labels at the batch median (``< median`` -> ``lo``, else
    ``hi``). Median-thresholding guarantees an ~50/50 split for any continuous
    factor, so majority-class accuracy stays ~0.5 and the gate is not
    underpowered by class imbalance."""
    v = np.asarray(values, dtype=np.float64)
    thr = float(np.median(v))
    return [hi if float(x) >= thr else lo for x in v]


# --------------------------------------------------------------------------- #
# Factor specs — each cell differs ONLY in its label (same frames, same Z)
# --------------------------------------------------------------------------- #


def _value_charger_yaw(charger_raw: np.ndarray, recep_raw: np.ndarray) -> float:
    return charger_rel_yaw(charger_raw[3:7], recep_raw[3:7])


def _value_charger_xy(charger_raw: np.ndarray, recep_raw: np.ndarray) -> float:
    recep_yaw = quat_yaw(recep_raw[3:7])
    _, rel_y = charger_rel_xy(charger_raw[0:3], recep_raw[0:3], recep_yaw)
    return rel_y  # lateral coord in the receptacle frame (left/right of slot)


def _value_receptacle_yaw(charger_raw: np.ndarray, recep_raw: np.ndarray) -> float:
    return quat_yaw(recep_raw[3:7])  # ABSOLUTE receptacle yaw (camera-cancelled)


# name -> (intent factor, value fn, lo/hi class names, role, description)
FACTOR_SPECS: dict[str, dict] = {
    "charger_yaw": {
        "intent": "object_motion",
        "value": _value_charger_yaw,
        "lo": "cw",
        "hi": "ccw",
        "role": "primary candidate",
        "desc": "charger apparent (camera-frame) in-plane yaw, +-60 deg",
    },
    "charger_xy": {
        "intent": "approach_direction",
        "value": _value_charger_xy,
        "lo": "left",
        "hi": "right",
        "role": "secondary",
        "desc": "charger lateral position in the receptacle frame (which side of the slot)",
    },
    "receptacle_yaw": {
        "intent": "constraint_region",
        "value": _value_receptacle_yaw,
        "lo": "neg",
        "hi": "pos",
        "role": "negative control",
        "desc": "receptacle ABSOLUTE world yaw (camera-cancelled by mount=receptacle)",
    },
}


# --------------------------------------------------------------------------- #
# Probe ladder (CPU, sim-free, tested)
# --------------------------------------------------------------------------- #


def run_probe(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    factor_idx: int,
    d_slot: int = 32,
    n_epochs: int = 300,
    seed: int = 0,
) -> dict:
    """Score features on the SAME axis as the G1 cells: a direct
    StandardScaler+LR column and the IntentHead-mediated nested CV."""
    direct = _direct_lr_probe(Z, y, seed=seed)
    intent = nested_cv_probe_one_factor(
        Z, y, factor_idx=factor_idx, d_slot=d_slot, n_epochs=n_epochs, seed=seed,
    )
    return {"dim": int(Z.shape[1]), "direct": direct, "intent": intent}


def _verdict(
    intent_mean: float, majority: float, shuffled: float, *, role: str = "primary candidate",
) -> str:
    gate = _gate(intent_mean, majority, shuffled)
    if role == "negative control":
        if gate == "PASS":
            return (
                "UNEXPECTED PASS — the negative control is separable, so the probe may be "
                "exploiting the constant in-frame receptacle/background rather than the "
                "charger. Treat every PlugCharger PASS as suspect until this is explained."
            )
        return (
            "FAIL as required — the camera is mounted on the receptacle, so its absolute yaw "
            "is cancelled in-frame and reads ~chance. This validates that the charger cells "
            "read the charger, not the constant background."
        )
    if gate == "PASS":
        return (
            "PASS — frozen DINOv2 spatial_mean separates this factor at the DEPLOYED "
            "representation. It is an INITIAL-STATE factor (fully visible from frame 0), so "
            "this config number IS the deployed number — no StackCube config->clip discount "
            "applies. A genuine second latent-viable factor beyond PushCube."
        )
    if intent_mean >= max(majority, shuffled) + 0.10:
        return (
            "WEAK LIFT over baselines but below the 0.90 gate. Partially groundable; run the "
            "--resolutions 518 scale follow-up before deciding. Not yet a clean latent cell."
        )
    return (
        "FAIL — frozen DINOv2 cannot separate this factor at the deployed representation. "
        "Not latent-viable with this encoder."
    )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _write_report(
    out_dir: Path,
    *,
    task: str,
    factor_name: str,
    spec: dict,
    n_seeds: int,
    n: int,
    label_dist: dict,
    result: dict,
    encoder: str,
    resolution: int,
    d_slot: int,
    n_epochs: int,
    fix_receptacle: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    majority = max(label_dist.values()) / n
    d, h = result["direct"], result["intent"]
    intent_mean = h["probe_acc_mean"]
    gate = _gate(intent_mean, h["majority_class_acc"], h["shuffled_features_acc"])
    role = spec["role"]
    L = [
        f"# Stage-5 — PlugCharger latent scout: `{factor_name}` ({role}) — {task}",
        "",
        f"**Factor:** {spec['desc']} → maps to intent factor `{spec['intent']}`.",
        "**Question:** can the DEPLOYED frozen encoder (DINOv2 ViT-B/14, spatial_mean,",
        f"resolution {resolution}) read this factor from the natural third-person reset frame",
        "(the demo's opening view)? This is the necessary-condition ceiling for latent-grounding",
        "the factor as a candidate second latent end-to-end task.",
        "",
        f"- Encoder: `{encoder}` spatial_mean @ resolution {resolution} "
        "(224 = the deployed P1 encoder exactly, so a PASS is deployment-honest).",
        f"- n={n} reset frames ({n_seeds} seeds), 2-class label {label_dist} "
        f"(majority {majority:.3f}).",
        "- Labels are charger-RELATIVE (de-rotated into the receptacle/camera frame) so the "
        "probe cannot exploit the constant in-frame receptacle/background.",
        f"- Probe: G1 protocol (IntentHead F=6, d_slot={d_slot}, n_epochs={n_epochs}, "
        f"factor_idx=`{spec['intent']}`) + direct StandardScaler+LR.",
        "",
        "## Result",
        "",
        "| feature | dim | majority | shuffled | direct LR ± std | IntentHead-CV ± std | gate |",
        "|---|---|---|---|---|---|---|",
        f"| `global_dino (spatial_mean)` | {result['dim']} | "
        f"{d['majority_class_acc']:.3f} | {d['shuffled_features_acc']:.3f} | "
        f"{d['probe_acc_mean']:.3f} ± {d['probe_acc_std']:.3f} | "
        f"{h['probe_acc_mean']:.3f} ± {h['probe_acc_std']:.3f} | {gate} |",
        "",
        "## Verdict",
        "",
        f"- IntentHead-CV **{intent_mean:.3f}** vs majority **{majority:.3f}**, "
        f"shuffled **{h['shuffled_features_acc']:.3f}**, gate **0.90**.",
        f"- {_verdict(intent_mean, h['majority_class_acc'], h['shuffled_features_acc'], role=role)}",
        "",
        "## Honesty boundary",
        "",
    ]
    if fix_receptacle:
        L += [
            "- **CONTROL mode (`--fix-receptacle`):** the receptacle is pinned to a canonical "
            "pose, so the receptacle-mounted camera — and the entire background (table + robot) "
            "— is IDENTICAL in every frame; only the charger varies. Any separability here MUST "
            "come from the charger itself, not the background-rotation cue. Compare to the "
            "deployed run to read off how much of the deployed number was background.",
        ]
    else:
        L += [
            "- **Camera mounted on the receptacle** (`mount=self.receptacle`) — and the "
            "background is NOT cancelled. When the receptacle rotates (yaw ±22.5°) the camera "
            "co-rotates, so the table + robot appear to counter-rotate: receptacle yaw leaks "
            "into the image as a global background rotation. The `receptacle_yaw` negative "
            "control reads ~0.96, FALSIFYING the naive camera-cancellation assumption. Because "
            "charger-relative labels share a partial correlation with this cue, the DEPLOYED "
            "charger numbers are CONFOUNDED — see the `--fix-receptacle` control, which removes "
            "the background and is the honest measure of charger groundability.",
        ]
    L += [
        "- **Initial-state factor:** the charger sits on the table at reset, fully visible from "
        "frame 0, so the single reset frame IS the deployed signal — there is NO StackCube-style "
        "config→clip discount (that trap was specific to final-state factors like goal_state).",
        "- **No sim privilege in the encoded signal:** poses are read only to author the class "
        "labels; the encoder reads pixels and no coordinate is fed to the probe "
        "(CLAUDE.md invariant #4).",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    (out_dir / "results.json").write_text(json.dumps({
        "task": task, "factor": factor_name, "intent_factor": spec["intent"],
        "role": role, "encoder": encoder, "resolution": resolution,
        "n_seeds": n_seeds, "n": n, "label_dist": label_dist,
        "d_slot": d_slot, "n_epochs": n_epochs, "gate": gate,
        "result": result,
    }, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# Env (GPU/Vulkan)
# --------------------------------------------------------------------------- #


def _make_env(task: str):
    """ManiSkill render env (state_dict obs, pd_ee_delta_pose, cpu backend,
    rgb_array render). PlugCharger-v1 only supports the panda_wristcam robot, so
    robot_uids must be set (unlike StackCube)."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 — registers tasks

    kwargs = dict(
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        sim_backend="cpu",
        render_mode="rgb_array",
    )
    if task == "PlugCharger-v1":
        kwargs["robot_uids"] = "panda_wristcam"  # SUPPORTED_ROBOTS = ['panda_wristcam']
        kwargs["max_episode_steps"] = 200
    return gym.make(task, **kwargs)


def _capture(env) -> np.ndarray:
    """Refresh the render scene from sim state and grab one third-person frame."""
    from babysteps.render.common import render_frame

    try:
        env.unwrapped.scene.update_render()
    except Exception:
        pass
    return render_frame(env)


# Canonical receptacle pose for the --fix-receptacle control (yaw = pi, a fixed
# point inside the receptacle's natural xy range, natural z). Holding the
# receptacle constant freezes the receptacle-mounted camera, so the background
# (table + robot) no longer counter-rotates with receptacle yaw — any residual
# charger separability must then come from the CHARGER, not a background cue.
_FIXED_RECEP_P = (0.055, 0.0, 0.1)
_FIXED_RECEP_Q = (0.0, 0.0, 0.0, 1.0)  # [qw,qx,qy,qz] for yaw = pi


def _collect_frames(
    task: str, seeds: list[int], *, fix_receptacle: bool = False,
    save_frames_dir: Path | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Render the reset frame per seed and read the charger/receptacle raw poses.

    With ``fix_receptacle=False`` (default) this is the DEPLOYED config: the
    natural randomized reset, receptacle-mounted camera, background counter-
    rotates with receptacle yaw. With ``fix_receptacle=True`` the receptacle is
    pinned to a canonical pose after reset (the camera, mounted on it, is thus
    identical every frame; only the charger varies) — the controlled experiment
    that removes the background-rotation confound exposed by the negative control.

    All factor cells share these frames (they differ only in label), so we render
    once and encode once per resolution downstream. Poses are read from the sim
    actors AFTER the optional override so labels reflect what is actually
    rendered. Returns (frames, charger_raw (n,7), receptacle_raw (n,7)).
    """
    from babysteps.render.common import to_np

    env = _make_env(task)
    frames: list[np.ndarray] = []
    charger_raw: list[np.ndarray] = []
    recep_raw: list[np.ndarray] = []
    try:
        for si, seed in enumerate(seeds):
            env.reset(seed=int(seed))
            if fix_receptacle:
                import sapien
                env.unwrapped.receptacle.set_pose(
                    sapien.Pose(p=list(_FIXED_RECEP_P), q=list(_FIXED_RECEP_Q)))
            ch = np.asarray(
                to_np(env.unwrapped.charger.pose.raw_pose), dtype=np.float64)
            rc = np.asarray(
                to_np(env.unwrapped.receptacle.pose.raw_pose), dtype=np.float64)
            frame = _capture(env)
            frames.append(frame)
            charger_raw.append(ch)
            recep_raw.append(rc)
            if save_frames_dir is not None and si < 4:
                save_frames_dir.mkdir(parents=True, exist_ok=True)
                try:
                    from PIL import Image
                    Image.fromarray(frame).save(save_frames_dir / f"seed_{seed:04d}.png")
                except Exception:
                    pass
            print(f"  seed {seed:04d} ({si + 1}/{len(seeds)}) rendered", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
    return frames, np.stack(charger_raw, axis=0), np.stack(recep_raw, axis=0)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def _parse_int_list(spec: str) -> list[int]:
    return [int(s) for s in spec.split(",") if s.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="PlugCharger-v1")
    p.add_argument("--factors", default="charger_yaw,charger_xy,receptacle_yaw",
                   help="comma list of factor cells (see FACTOR_SPECS); "
                        "receptacle_yaw is the negative control.")
    p.add_argument("--seeds", default="0-119", help="Inclusive seed range A-B.")
    p.add_argument("--resolutions", default="224",
                   help="comma list of DINOv2 input resolutions; 224 = the deployed "
                        "encoder, 518 = scale-ceiling follow-up. Frames are rendered once "
                        "and encoded at each resolution.")
    p.add_argument("--encoder", default="dinov2_vitb14")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", type=Path,
                   default=Path("reports/stage5/plugcharger_probe"))
    p.add_argument("--d-slot", type=int, default=32)
    p.add_argument("--n-epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-frames", action="store_true",
                   help="Dump the first few rendered reset frames as PNGs for sanity.")
    p.add_argument("--fix-receptacle", action="store_true",
                   help="CONTROL mode: pin the receptacle to a canonical pose so the "
                        "receptacle-mounted camera (hence the background) is constant. "
                        "Isolates the charger signal from the background-rotation cue "
                        "that the negative control exposed. (receptacle_yaw becomes "
                        "constant and is auto-skipped.)")
    args = p.parse_args(argv)

    factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    unknown = [f for f in factors if f not in FACTOR_SPECS]
    if unknown:
        p.error(f"unknown factor(s) {unknown}; choices: {sorted(FACTOR_SPECS)}")
    resolutions = _parse_int_list(args.resolutions)
    seeds = _parse_seeds(args.seeds)
    out_root = args.out_dir / args.task
    frames_dir = (out_root / "frames") if args.save_frames else None

    mode = "fix-receptacle CONTROL" if args.fix_receptacle else "deployed"
    print(f"PlugCharger scout [{mode}]: {args.task} seeds {seeds[0]}-{seeds[-1]} "
          f"(n={len(seeds)}), factors={factors}, resolutions={resolutions}")

    from babysteps.stage4.vision_features import extract_vision_features

    frames, charger_raw, recep_raw = _collect_frames(
        args.task, seeds, fix_receptacle=args.fix_receptacle,
        save_frames_dir=frames_dir)
    print(f"rendered {len(frames)} frames")

    for resolution in resolutions:
        Z = np.stack([
            extract_vision_features(
                [f], encoder=args.encoder, pool="spatial_mean",
                device=args.device, resolution=resolution,
            )
            for f in frames
        ], axis=0).astype(np.float32)
        print(f"\n[resolution {resolution}] Z={Z.shape}")

        for factor_name in factors:
            spec = FACTOR_SPECS[factor_name]
            vals = [spec["value"](charger_raw[i], recep_raw[i]) for i in range(len(frames))]
            y_str = median_split_labels(vals, lo=spec["lo"], hi=spec["hi"])
            classes = sorted(set(y_str))
            if len(classes) < 2:
                print(f"  {factor_name:14s} SKIPPED — only one class "
                      f"(factor is constant in this mode, e.g. fix-receptacle)")
                continue
            y = np.asarray([classes.index(v) for v in y_str], dtype=np.int64)
            label_dist = {c: int((np.asarray(y_str) == c).sum()) for c in classes}
            factor_idx = INTENT_FIELDS.index(spec["intent"])

            result = run_probe(Z, y, factor_idx=factor_idx, d_slot=args.d_slot,
                               n_epochs=args.n_epochs, seed=args.seed)
            sub = factor_name if len(resolutions) == 1 else f"{factor_name}_res{resolution}"
            out_dir = out_root / sub
            _write_report(out_dir, task=args.task, factor_name=factor_name, spec=spec,
                          n_seeds=len(seeds), n=len(y_str), label_dist=label_dist,
                          result=result, encoder=args.encoder, resolution=resolution,
                          d_slot=args.d_slot, n_epochs=args.n_epochs,
                          fix_receptacle=args.fix_receptacle)
            im = result["intent"]
            print(f"  {factor_name:14s} [{spec['role']:17s}] "
                  f"direct={result['direct']['probe_acc_mean']:.3f} "
                  f"intentCV={im['probe_acc_mean']:.3f} "
                  f"(maj {im['majority_class_acc']:.3f}, shuf {im['shuffled_features_acc']:.3f}) "
                  f"-> {_gate(im['probe_acc_mean'], im['majority_class_acc'], im['shuffled_features_acc'])}"
                  f"  [{out_dir}]")

    print(f"\nwrote reports under {out_root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
