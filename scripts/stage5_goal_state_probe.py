"""Stage-5 — StackCube ``goal_state`` pixel-separability probe (GPU render + CPU probe).

The one decision-relevant question for the latent-intent pivot on StackCube:

    Does ``goal_state`` have a pixel signature that frozen DINOv2 can read?
    i.e. can the SAME encoder used everywhere (DINOv2 ViT-B/14, spatial_mean)
    linearly separate the two goal CONFIGURATIONS —
      * ``cubeA_on_cubeB``  : cubeA resting ON TOP of cubeB (a stacked tower)
      * ``cube_at_target``  : cubeA resting NEAR/beside cubeB on the table
        (the "place-near" reading of the under-specified goal)?

This is the necessary-condition CEILING for latent-grounding ``goal_state``.
The relation-oracle work already showed StackCube ``object_motion`` is
representation-blocked (frozen DINOv2 0.42->0.68, object-local pooling does not
close it) and PickCube ``contact_region`` is invisible (symbolic-only). This
probe asks whether ``goal_state`` is the surviving latent-viable factor.

What this measures — and its limit (read this)
-----------------------------------------------
We render the two goal CONFIGS directly (cubeA pose-injected on-top vs beside
cubeB), capture one third-person frame each (the deployed *demo* view), and
DINOv2-encode them. The near-offset DIRECTION is balanced across the four
cardinals per seed so "cubeA is shifted +x" can't stand in for the label —
the probe must learn *co-located-and-raised* (stack) vs *displaced* (near),
which is exactly the ``goal_state`` semantics.

  * PASS (>=0.90): ``goal_state`` is pixel-groundable in principle. Unlike
    object_motion, this is a WHOLE-IMAGE config difference (tower vs two cubes
    side by side), so global spatial_mean should suffice. Green-lights a
    faithful clip-based latent goal_state collection.
  * FAIL: ``goal_state`` is not even separable at the clean-config level →
    consolidate on PushCube as the single end-to-end latent task.

Honesty boundary
----------------
This is a CONFIG ceiling, not yet the deployed-demo result. Two caveats the
report restates:
  1. The deployed demo is a CLIP, and the Stage-0 StackCube demo deliberately
     HIDES vertical motion in its 2D trajectory ("controlled information loss",
     Sub-project C goal-ambiguity). So a PASS means the ambiguity is a
     demo-DESIGN choice, not a representation limit: a goal-disambiguating demo
     render would make goal_state latent-viable. A FAIL kills it outright.
  2. cube poses are injected (privileged sim obs) ONLY to author the two
     canonical configs to encode; the encoder still reads pixels only
     (CLAUDE.md invariant #4). No coordinate is ever fed to the probe.

Example::

    python scripts/stage5_goal_state_probe.py \\
        --task StackCube-v1 --seeds 0-59 \\
        --out-dir reports/stage5/goal_state_probe/

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

_GOAL_STATE_IDX = INTENT_FIELDS.index("goal_state")

# StackCube-v1 geometry (mani_skill .../tabletop/stack_cube.py): cube
# half_size = 0.02, so a cube edge is 0.04. cubeA-on-cubeB raises cubeA's
# center by 2*half_size = 0.04 above cubeB's center.
_CUBE_TOP_DZ = 0.04
# Center-to-center offset for "place near": 0.06 leaves a 0.02 gap between
# faces (clearly adjacent, never overlapping/stacked).
_NEAR_OFFSET = 0.06
# Balanced near-offset directions so absolute "+x/-x" shift can't proxy the
# label — only the stacked-vs-displaced relation separates the classes.
_NEAR_DIRS = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))

# label tokens (the Stage-0 GOAL_STATES used by StackCube / Sub-project C)
_STACK_TOKEN = "cubeA_on_cubeB"
_NEAR_TOKEN = "cube_at_target"


# --------------------------------------------------------------------------- #
# Pure geometry (sim-free, tested)
# --------------------------------------------------------------------------- #


def build_goal_configs(
    cubeB_p,
    cubeA_q,
    *,
    offset: float = _NEAR_OFFSET,
    direction_idx: int = 0,
    top_dz: float = _CUBE_TOP_DZ,
) -> dict[str, tuple[tuple[float, float, float], tuple[float, ...]]]:
    """The two canonical ``goal_state`` configs as (position, quaternion).

    ``stack_on``   : cubeA at cubeB.xy, raised by ``top_dz`` (resting on cubeB).
    ``place_near`` : cubeA at cubeB.xy + ``offset`` along cardinal
                     ``direction_idx`` (mod 4), at cubeB's resting height.

    cubeA keeps its own (upright, yaw-random) quaternion in both. Positions
    are RELATIVE to cubeB, so the discriminator is the stacked-vs-displaced
    relation, not absolute position.
    """
    bx, by, bz = float(cubeB_p[0]), float(cubeB_p[1]), float(cubeB_p[2])
    dx, dy = _NEAR_DIRS[direction_idx % len(_NEAR_DIRS)]
    q = tuple(float(v) for v in cubeA_q)
    return {
        "stack_on": ((bx, by, bz + top_dz), q),
        "place_near": ((bx + offset * dx, by + offset * dy, bz), q),
    }


def clip_drop_xy(
    cubeB_xy,
    goal_token: str,
    *,
    direction_idx: int = 0,
    offset: float = _NEAR_OFFSET,
) -> tuple[float, float]:
    """The xy the stack skill should drive cubeA to, per goal_state.

    For ``cubeA_on_cubeB`` cubeA goes onto the real cubeB (drop xy = cubeB.xy).
    For ``cube_at_target`` (place-near) cubeA is dropped at cubeB.xy + ``offset``
    along a balanced cardinal — a free table spot BESIDE cubeB. Fed to the demo
    executor via ``scene.extra["cubeB_xy"]`` (the field _build_translate_waypoints
    reads), so the rendered CLIP genuinely shows place-near vs stack-on.
    """
    bx, by = float(cubeB_xy[0]), float(cubeB_xy[1])
    if goal_token == _STACK_TOKEN:
        return (bx, by)
    dx, dy = _NEAR_DIRS[direction_idx % len(_NEAR_DIRS)]
    return (bx + offset * dx, by + offset * dy)


def clip_pool_frame_indices(n_frames: int) -> dict[str, list[int]]:
    """Which clip frames feed each temporal pooling (pure, for the clip-pool
    diagnostic). ``goal_state`` is a FINAL-STATE factor, so the question is
    whether the deployed mean-over-all-frames pooling dilutes a signal that a
    final-state-aware pooling preserves."""
    T = int(n_frames)
    if T <= 0:
        raise ValueError("clip_pool_frame_indices needs at least one frame")
    last = T - 1
    return {
        "spatial_mean (all)": list(range(T)),   # deployed default
        "final_frame": [last],                  # final-state pooling
        "first_last": [0, last],                # start-vs-end delta
        "last5_mean": list(range(max(0, T - 5), T)),
    }


# --------------------------------------------------------------------------- #
# Probe ladder (CPU, sim-free, tested)
# --------------------------------------------------------------------------- #


def run_probe(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    factor_idx: int = _GOAL_STATE_IDX,
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


def _verdict(intent_mean: float, majority: float, shuffled: float) -> str:
    if _gate(intent_mean, majority, shuffled) == "PASS":
        return (
            "PASS — goal_state IS pixel-groundable: frozen DINOv2 spatial_mean "
            "separates the stacked vs place-near configs. Unlike object_motion "
            "(representation-blocked) this is a whole-image config difference. "
            "GREEN-LIGHT a faithful clip-based latent goal_state collection "
            "(render goal-DISAMBIGUATING demos so the encoder can read it)."
        )
    if intent_mean >= max(majority, shuffled) + 0.10:
        return (
            "WEAK LIFT over baselines but below the 0.90 gate. Inspect the "
            "rendered configs / expand n before committing; goal_state may be "
            "only partially groundable."
        )
    return (
        "FAIL — frozen DINOv2 cannot separate the two goal_state configs even "
        "at the clean-config ceiling. goal_state is NOT latent-groundable with "
        "this encoder → consolidate on PushCube as the single end-to-end "
        "latent task."
    )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _write_report(
    out_dir: Path,
    *,
    task: str,
    mode: str,
    n_seeds: int,
    n: int,
    label_dist: dict,
    offset: float,
    result: dict,
    encoder: str,
    d_slot: int,
    n_epochs: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    majority = max(label_dist.values()) / n
    d, h = result["direct"], result["intent"]
    intent_mean = h["probe_acc_mean"]
    if mode == "clip":
        unit = "demo CLIPS"
        setup = (
            f"- Demo CLIPS (real Stage-0 executor), DINOv2 spatial_mean pooled over "
            f"TIME+patches = the DEPLOYED representation: stack = cubeA_on_cubeB onto "
            f"cubeB; near = cube_at_target dropped at cubeB.xy + {offset} along a "
            "per-seed BALANCED cardinal. Tests whether clip-pooling dilutes the "
            "signal the static-config probe saw at 0.99."
        )
    else:
        unit = "static configs"
        setup = (
            f"- Configs are pose-injected: stack = cubeB.xy at z+{_CUBE_TOP_DZ}; near = "
            f"cubeB.xy + {offset} along a per-seed BALANCED cardinal (so absolute shift "
            "can't proxy the label). cubeB resting pose varies per seed."
        )
    L = [
        f"# Stage-5 — StackCube `goal_state` pixel-separability probe ({mode}) — {task}",
        "",
        "**Question:** can frozen DINOv2 (ViT-B/14, spatial_mean) separate the two",
        "`goal_state` cases — cubeA stacked ON cubeB (`cubeA_on_cubeB`) vs cubeA",
        "placed NEAR/beside cubeB (`cube_at_target`, the place-near reading)?",
        "This is the necessary-condition ceiling for latent-grounding `goal_state`.",
        "",
        f"- Encoder: `{encoder}` spatial_mean (the deployed P1 encoder).",
        f"- n={n} ({n_seeds} seeds x 2 {unit}), label dist {label_dist} "
        f"(majority {majority:.3f}).",
        setup,
        f"- Probe: same protocol as the G1 cells (IntentHead F=6, d_slot={d_slot}, "
        f"n_epochs={n_epochs}, factor_idx=goal_state) + direct StandardScaler+LR.",
        "",
        "## Result",
        "",
        "| feature | dim | majority | shuffled | direct LR ± std | IntentHead-CV ± std | gate |",
        "|---|---|---|---|---|---|---|",
        f"| `global_dino (spatial_mean)` | {result['dim']} | "
        f"{d['majority_class_acc']:.3f} | {d['shuffled_features_acc']:.3f} | "
        f"{d['probe_acc_mean']:.3f} ± {d['probe_acc_std']:.3f} | "
        f"{h['probe_acc_mean']:.3f} ± {h['probe_acc_std']:.3f} | "
        f"{_gate(intent_mean, h['majority_class_acc'], h['shuffled_features_acc'])} |",
        "",
        "## Verdict",
        "",
        f"- IntentHead-CV **{intent_mean:.3f}** vs majority **{majority:.3f}**, "
        f"shuffled **{h['shuffled_features_acc']:.3f}**, gate **0.90**.",
        f"- {_verdict(intent_mean, h['majority_class_acc'], h['shuffled_features_acc'])}",
        "",
        "## Honesty boundary",
        "",
        "- **Config ceiling, not the deployed demo.** This encodes the clean "
        "canonical goal configs. The deployed StackCube demo is a CLIP that "
        "deliberately HIDES vertical motion (2D-trajectory info loss = "
        "Sub-project C goal-ambiguity). A PASS means the ambiguity is a "
        "demo-DESIGN choice, not a representation limit — a goal-disambiguating "
        "demo would make goal_state latent-viable. A FAIL kills it regardless.",
        "- **No sim privilege in the encoded signal.** Cube poses are injected "
        "ONLY to author the two configs to render; the encoder reads pixels and "
        "no coordinate is ever fed to the probe (CLAUDE.md invariant #4).",
        "- **Two-class, balanced** (majority 0.5) — a clean binary ceiling, not "
        "the 4-way object_motion problem.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    (out_dir / "results.json").write_text(json.dumps({
        "task": task, "mode": mode, "encoder": encoder,
        "n_seeds": n_seeds, "n": n,
        "label_dist": label_dist, "near_offset": offset,
        "d_slot": d_slot, "n_epochs": n_epochs,
        "gate": _gate(intent_mean, h["majority_class_acc"], h["shuffled_features_acc"]),
        "result": result,
    }, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# Env (GPU/Vulkan)
# --------------------------------------------------------------------------- #


def _make_env(task: str):
    """ManiSkill render env, matching scripts/stage5_render_demo_frames._make_env
    (state_dict obs, pd_ee_delta_pose, cpu backend, rgb_array render)."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 — registers tasks

    kwargs = dict(
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        sim_backend="cpu",
        render_mode="rgb_array",
    )
    if task == "StackCube-v1":
        kwargs["max_episode_steps"] = 200
    return gym.make(task, **kwargs)


def _capture(env) -> np.ndarray:
    """Refresh the render scene from the (pose-injected) sim state and grab one
    third-person frame."""
    from babysteps.render.common import render_frame

    try:
        env.unwrapped.scene.update_render()
    except Exception:
        pass
    return render_frame(env)


def _collect_features(
    task: str, seeds: list[int], *, offset: float, encoder: str, device: str,
    save_frames_dir: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Render the two goal configs per seed and DINOv2-encode each.

    Returns (Z (2n, d), y_str (2n,)). Class order per seed: place_near, stack_on.
    """
    import sapien

    from babysteps.render.common import to_np
    from babysteps.stage4.vision_features import extract_vision_features

    env = _make_env(task)
    Z: list[np.ndarray] = []
    y: list[str] = []
    try:
        for si, seed in enumerate(seeds):
            obs, _info = env.reset(seed=int(seed))
            cubeA_raw = np.asarray(to_np(obs["extra"]["cubeA_pose"]), dtype=np.float64)
            cubeB_raw = np.asarray(to_np(obs["extra"]["cubeB_pose"]), dtype=np.float64)
            cubeB_p = cubeB_raw[0:3]
            cubeA_q = cubeA_raw[3:7]  # raw_pose quat order [qw,qx,qy,qz]
            configs = build_goal_configs(
                cubeB_p, cubeA_q, offset=offset, direction_idx=si,
            )
            # Encode place_near then stack_on (balanced, deterministic order).
            for token, key in ((_NEAR_TOKEN, "place_near"), (_STACK_TOKEN, "stack_on")):
                (px, py, pz), q = configs[key]
                env.unwrapped.cubeA.set_pose(
                    sapien.Pose(p=[px, py, pz], q=list(q))
                )
                frame = _capture(env)
                z = extract_vision_features(
                    [frame], encoder=encoder, pool="spatial_mean", device=device,
                )
                Z.append(z)
                y.append(token)
                if save_frames_dir is not None and si < 3:
                    save_frames_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        from PIL import Image
                        Image.fromarray(frame).save(
                            save_frames_dir / f"seed_{seed:04d}_{key}.png"
                        )
                    except Exception:
                        pass
            print(f"  seed {seed:04d} ({si + 1}/{len(seeds)}) encoded", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
    return np.stack(Z, axis=0).astype(np.float32), y


def _collect_features_clip(
    task: str, seeds: list[int], *, offset: float, encoder: str, device: str,
    save_frames_dir: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Render a full demo CLIP per goal_state and DINOv2-encode the WHOLE clip.

    This is the deployed representation: spatial_mean pools over time AND
    patches, so it tests whether clip-pooling (early frames where cubeA is at
    its start, ~identical across classes) DILUTES the goal_state signal that
    the static-config probe saw at 0.99. Clips use the real Stage-0 demo
    executor (``babysteps.render.stackcube._execute_stack``):
      * stack_on   : cubeA_on_cubeB (5-wp) onto the real cubeB -> tower.
      * place_near : cube_at_target (4-wp) with the drop target redirected to
        cubeB.xy + offset (a free table spot) -> clean place-beside.
    """
    from dataclasses import replace

    from babysteps.envs.task_registry import get_task_entry
    from babysteps.render.stackcube import _execute_stack, _read_stack_obs
    from babysteps.schemas import SceneState
    from babysteps.skills.stack import CUBE_HALF_SIZE
    from babysteps.stage4.vision_features import extract_vision_features

    adapter = get_task_entry(task).adapter_cls()
    env = _make_env(task)
    Z: list[np.ndarray] = []
    y: list[str] = []
    try:
        for si, seed in enumerate(seeds):
            obs, _info = env.reset(seed=int(seed))
            tcp, cubeA_xy0, cubeA_z, cubeB_xy, cubeB_z = _read_stack_obs(obs)
            cubeB_top_z = cubeB_z + 2 * CUBE_HALF_SIZE
            for token, key in ((_NEAR_TOKEN, "place_near"), (_STACK_TOKEN, "stack_on")):
                drop_xy = clip_drop_xy(cubeB_xy, token, direction_idx=si, offset=offset)
                scene = SceneState(
                    cube_xy=(float(cubeA_xy0[0]), float(cubeA_xy0[1])),
                    cube_z=cubeA_z,
                    goal_xy=(float(cubeB_xy[0]), float(cubeB_xy[1])),
                    tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
                    blocked_sides=(),
                    extra={
                        "cubeB_xy": drop_xy,  # drop target for the stack skill
                        "cubeB_z": cubeB_z,
                        "cubeB_top_z": cubeB_top_z,
                    },
                )
                intent = replace(adapter.oracle_correct_intent(scene), goal_state=token)
                frames: list = []
                _execute_stack(env, intent, scene, frames, seed=int(seed))
                if not frames:
                    raise RuntimeError(f"empty clip for seed {seed} ({key})")
                z = extract_vision_features(
                    frames, encoder=encoder, pool="spatial_mean", device=device,
                )
                Z.append(z)
                y.append(token)
                if save_frames_dir is not None and si < 3:
                    save_frames_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        from PIL import Image
                        Image.fromarray(frames[-1]).save(
                            save_frames_dir / f"seed_{seed:04d}_{key}_last.png"
                        )
                    except Exception:
                        pass
            print(f"  seed {seed:04d} ({si + 1}/{len(seeds)}) clips encoded", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            adapter.close()
        except Exception:
            pass
    return np.stack(Z, axis=0).astype(np.float32), y


def _collect_clip_multipool(
    task: str, seeds: list[int], *, offset: float, encoder: str, device: str,
    save_frames_dir: Path | None = None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Render a demo CLIP per goal_state and DINOv2-encode it under SEVERAL
    temporal poolings (deployed spatial_mean vs final-state-aware poolings).

    Returns ({pooling_name: Z (n, d)}, y_str (n,)). Diagnoses whether the clip
    FAIL is mean-pooling DILUTION (final-state poolings recover it) or a true
    representation block (all poolings fail).
    """
    from dataclasses import replace

    from babysteps.envs.task_registry import get_task_entry
    from babysteps.render.stackcube import _execute_stack, _read_stack_obs
    from babysteps.schemas import SceneState
    from babysteps.skills.stack import CUBE_HALF_SIZE
    from babysteps.stage4.vision_features import extract_vision_features

    adapter = get_task_entry(task).adapter_cls()
    env = _make_env(task)
    Zs: dict[str, list[np.ndarray]] = {}
    y: list[str] = []
    try:
        for si, seed in enumerate(seeds):
            obs, _info = env.reset(seed=int(seed))
            tcp, cubeA_xy0, cubeA_z, cubeB_xy, cubeB_z = _read_stack_obs(obs)
            cubeB_top_z = cubeB_z + 2 * CUBE_HALF_SIZE
            for token, key in ((_NEAR_TOKEN, "place_near"), (_STACK_TOKEN, "stack_on")):
                drop_xy = clip_drop_xy(cubeB_xy, token, direction_idx=si, offset=offset)
                scene = SceneState(
                    cube_xy=(float(cubeA_xy0[0]), float(cubeA_xy0[1])),
                    cube_z=cubeA_z,
                    goal_xy=(float(cubeB_xy[0]), float(cubeB_xy[1])),
                    tcp_start_pose=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
                    blocked_sides=(),
                    extra={"cubeB_xy": drop_xy, "cubeB_z": cubeB_z,
                           "cubeB_top_z": cubeB_top_z},
                )
                intent = replace(adapter.oracle_correct_intent(scene), goal_state=token)
                frames: list = []
                _execute_stack(env, intent, scene, frames, seed=int(seed))
                if not frames:
                    raise RuntimeError(f"empty clip for seed {seed} ({key})")
                idx_by_pool = clip_pool_frame_indices(len(frames))
                for pool_name, idxs in idx_by_pool.items():
                    z = extract_vision_features(
                        [frames[i] for i in idxs], encoder=encoder,
                        pool="spatial_mean", device=device,
                    )
                    Zs.setdefault(pool_name, []).append(z)
                y.append(token)
                if save_frames_dir is not None and si < 3:
                    save_frames_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        from PIL import Image
                        Image.fromarray(frames[-1]).save(
                            save_frames_dir / f"seed_{seed:04d}_{key}_last.png"
                        )
                    except Exception:
                        pass
            print(f"  seed {seed:04d} ({si + 1}/{len(seeds)}) multipool encoded",
                  flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            adapter.close()
        except Exception:
            pass
    return ({k: np.stack(v, axis=0).astype(np.float32) for k, v in Zs.items()}, y)


def _write_multipool_report(
    out_dir: Path, *, task: str, n_seeds: int, n: int, label_dist: dict,
    offset: float, results_by_pool: dict, encoder: str, d_slot: int, n_epochs: int,
) -> None:
    """Report the temporal-pooling ladder for the clip goal_state signal."""
    out_dir.mkdir(parents=True, exist_ok=True)
    majority = max(label_dist.values()) / n
    deployed = results_by_pool.get("spatial_mean (all)", {})
    dep_mean = deployed.get("intent", {}).get("probe_acc_mean", float("nan"))
    # Best final-state-aware pooling (anything but the deployed mean).
    alt = {k: v for k, v in results_by_pool.items() if k != "spatial_mean (all)"}
    best_name = max(alt, key=lambda k: alt[k]["intent"]["probe_acc_mean"]) if alt else None
    best_mean = alt[best_name]["intent"]["probe_acc_mean"] if best_name else float("nan")

    if best_name is not None and _gate(
        best_mean, alt[best_name]["intent"]["majority_class_acc"],
        alt[best_name]["intent"]["shuffled_features_acc"]) == "PASS" and dep_mean < 0.90:
        verdict = (
            f"POOLING DILUTION — goal_state IS pixel-groundable, but the DEPLOYED "
            f"spatial_mean-over-clip pooling dilutes it ({dep_mean:.3f}); a "
            f"final-state-aware pooling (`{best_name}` {best_mean:.3f}) recovers it. "
            "goal_state is a FINAL-STATE factor; mean-over-trajectory is the wrong "
            "pooling for it. Latent-viable ONLY with a per-factor pooling change "
            "(caveat: the deployed encoder uses spatial_mean uniformly across tasks "
            "/ factors -> a StackCube-specific pooling is a reviewer-visible "
            "inconsistency unless framed as principled final-state pooling)."
        )
    elif best_name is not None and best_mean >= max(majority, 0.0) + 0.10 and best_mean < 0.90:
        verdict = (
            f"PARTIAL — even the best pooling (`{best_name}` {best_mean:.3f}) lifts "
            "over chance but misses the 0.90 gate. goal_state is only weakly "
            "clip-groundable; not a clean latent cell."
        )
    else:
        verdict = (
            "REPRESENTATION BLOCK — no temporal pooling clears the gate on the "
            "demo clip. goal_state is not latent-groundable from clips with this "
            "encoder -> consolidate on PushCube as the single end-to-end latent task."
        )

    L = [
        f"# Stage-5 — StackCube `goal_state` clip temporal-pooling ladder — {task}",
        "",
        "**Question:** the static final config separates at 0.99 but the deployed",
        "spatial_mean-over-clip pooling fell to 0.65. Is that mean-pooling DILUTION",
        "(a final-state-aware pooling recovers it) or a representation block?",
        "",
        f"- Encoder: `{encoder}`; same clips, several temporal poolings (frame",
        "  subsets pooled by DINOv2 spatial_mean).",
        f"- n={n} ({n_seeds} seeds x 2 demo CLIPS), label dist {label_dist} "
        f"(majority {majority:.3f}); place-near drop = cubeB.xy + {offset} (balanced).",
        f"- Probe: G1 protocol (IntentHead F=6, d_slot={d_slot}, n_epochs={n_epochs}, "
        "factor_idx=goal_state) + direct LR.",
        "",
        "## Temporal-pooling ladder",
        "",
        "| pooling | dim | majority | shuffled | direct LR ± std | IntentHead-CV ± std | gate |",
        "|---|---|---|---|---|---|---|",
    ]
    # Stable, informative ordering.
    order = ["spatial_mean (all)", "last5_mean", "first_last", "final_frame"]
    for name in [o for o in order if o in results_by_pool] + \
            [k for k in results_by_pool if k not in order]:
        r = results_by_pool[name]
        d, h = r["direct"], r["intent"]
        L.append(
            f"| `{name}` | {r['dim']} | {h['majority_class_acc']:.3f} | "
            f"{h['shuffled_features_acc']:.3f} | "
            f"{d['probe_acc_mean']:.3f} ± {d['probe_acc_std']:.3f} | "
            f"{h['probe_acc_mean']:.3f} ± {h['probe_acc_std']:.3f} | "
            f"{_gate(h['probe_acc_mean'], h['majority_class_acc'], h['shuffled_features_acc'])} |"
        )
    L += [
        "",
        "## Verdict",
        "",
        f"- Deployed `spatial_mean (all)` IntentHead-CV **{dep_mean:.3f}** "
        f"(gate 0.90); best final-state pooling "
        f"**`{best_name}` {best_mean:.3f}**." if best_name else
        f"- Deployed `spatial_mean (all)` IntentHead-CV **{dep_mean:.3f}**.",
        f"- {verdict}",
        "",
        "## Honesty boundary",
        "",
        "- Cube poses / drop targets are injected ONLY to author the two demo "
        "clips; the encoder reads pixels, no coordinate is fed to the probe.",
        "- A final-state pooling that 'works' here is NOT free: the deployed P1 "
        "encoder pools spatial_mean uniformly. Adopting a goal_state-specific "
        "pooling needs a principled justification (final-state vs motion/contact "
        "factors), or it reads as per-task tuning.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    (out_dir / "results.json").write_text(json.dumps({
        "task": task, "mode": "clip-pool", "encoder": encoder,
        "n_seeds": n_seeds, "n": n, "label_dist": label_dist, "near_offset": offset,
        "d_slot": d_slot, "n_epochs": n_epochs,
        "deployed_intent_acc": dep_mean,
        "best_alt_pooling": best_name, "best_alt_intent_acc": best_mean,
        "results_by_pool": results_by_pool,
    }, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="StackCube-v1")
    p.add_argument("--mode", choices=("config", "clip", "clip-pool"),
                   default="config",
                   help="config: static pose-injected goal configs (ceiling). "
                        "clip: full demo clips, spatial_mean over time+patches "
                        "(the DEPLOYED representation). clip-pool: clips under "
                        "several temporal poolings (dilution vs representation "
                        "diagnostic).")
    p.add_argument("--seeds", default="0-59", help="Inclusive seed range A-B.")
    p.add_argument("--offset", type=float, default=_NEAR_OFFSET,
                   help="Center-to-center xy offset for the place-near config.")
    p.add_argument("--encoder", default="dinov2_vitb14")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", type=Path,
                   default=Path("reports/stage5/goal_state_probe"))
    p.add_argument("--d-slot", type=int, default=32)
    p.add_argument("--n-epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-frames", action="store_true",
                   help="Dump the first few rendered configs as PNGs for sanity.")
    args = p.parse_args(argv)

    seeds = _parse_seeds(args.seeds)
    out_dir = args.out_dir / args.task
    frames_dir = (out_dir / "frames") if args.save_frames else None
    print(f"goal_state probe ({args.mode}): {args.task} seeds "
          f"{seeds[0]}-{seeds[-1]} (n={len(seeds)} x2), offset={args.offset}")

    if args.mode == "clip-pool":
        Z_by_pool, y_str = _collect_clip_multipool(
            args.task, seeds, offset=args.offset, encoder=args.encoder,
            device=args.device, save_frames_dir=frames_dir,
        )
        classes = sorted(set(y_str))
        y = np.asarray([classes.index(v) for v in y_str], dtype=np.int64)
        label_dist = {c: int((np.asarray(y_str) == c).sum()) for c in classes}
        results_by_pool = {}
        for pool_name, Zp in Z_by_pool.items():
            results_by_pool[pool_name] = run_probe(
                Zp, y, factor_idx=_GOAL_STATE_IDX, d_slot=args.d_slot,
                n_epochs=args.n_epochs, seed=args.seed,
            )
            print(f"  {pool_name:20s} direct={results_by_pool[pool_name]['direct']['probe_acc_mean']:.3f}"
                  f"  intentCV={results_by_pool[pool_name]['intent']['probe_acc_mean']:.3f}")
        _write_multipool_report(
            out_dir, task=args.task, n_seeds=len(seeds), n=len(y_str),
            label_dist=label_dist, offset=args.offset,
            results_by_pool=results_by_pool, encoder=args.encoder,
            d_slot=args.d_slot, n_epochs=args.n_epochs,
        )
        print(f"\nwrote {out_dir}/report.md\nwrote {out_dir}/results.json")
        return 0

    collect = _collect_features_clip if args.mode == "clip" else _collect_features
    Z, y_str = collect(
        args.task, seeds, offset=args.offset, encoder=args.encoder,
        device=args.device, save_frames_dir=frames_dir,
    )

    classes = sorted(set(y_str))
    y = np.asarray([classes.index(v) for v in y_str], dtype=np.int64)
    label_dist = {c: int((np.asarray(y_str) == c).sum()) for c in classes}
    print(f"encoded n={len(y_str)}; labels {label_dist}; Z={Z.shape}")

    result = run_probe(Z, y, factor_idx=_GOAL_STATE_IDX, d_slot=args.d_slot,
                       n_epochs=args.n_epochs, seed=args.seed)
    print(f"  direct LR  = {result['direct']['probe_acc_mean']:.3f}")
    print(f"  IntentCV   = {result['intent']['probe_acc_mean']:.3f} "
          f"(majority {result['intent']['majority_class_acc']:.3f}, "
          f"shuffled {result['intent']['shuffled_features_acc']:.3f})")

    _write_report(out_dir, task=args.task, mode=args.mode, n_seeds=len(seeds),
                  n=len(y_str), label_dist=label_dist, offset=args.offset,
                  result=result, encoder=args.encoder, d_slot=args.d_slot,
                  n_epochs=args.n_epochs)
    print(f"\nwrote {out_dir}/report.md\nwrote {out_dir}/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
