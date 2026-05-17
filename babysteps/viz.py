"""Sim-free 2D top-down rendering for Stage-0 episodes.

Produces MP4s directly from the cube xy trajectories already in
samples.jsonl-style records. Useful for previewing the loop on the login
node, where ManiSkill's RGB rendering isn't available (no Vulkan ICD).

For real (RGB) ManiSkill recordings see `scripts/render_stage0_maniskill.py`,
which runs on a Vulkan-capable compute node.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Defer heavy imports — viz is optional in the rest of the package.


def _import_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return matplotlib, plt


def _new_figure(plt, world_extent: float):
    # Wide figure: left half is the top-down scene, right half holds the
    # intent annotation boxes (which are placed with transform=ax.transAxes
    # at x > 1.0).
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.set_xlim(-world_extent, world_extent)
    ax.set_ylim(-world_extent, world_extent)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25)
    # Leave 55% of the figure width for annotations on the right.
    fig.subplots_adjust(left=0.07, right=0.45, top=0.88, bottom=0.10)
    return fig, ax


def _draw_static(ax, *, cube0_xy, goal_xy, goal_radius, cube_half_size,
                 blocked_sides: Sequence[str]):
    # Table outline.
    extent = max(abs(goal_xy[0]), abs(goal_xy[1])) + 0.05
    ax.add_patch(
        _patch_rect(ax, -extent, -extent, 2 * extent, 2 * extent,
                    edgecolor="0.7", facecolor="none", linewidth=1)
    )
    # Goal disc.
    ax.add_patch(
        _patch_circle(ax, goal_xy, goal_radius,
                      edgecolor="tab:green", facecolor="tab:green", alpha=0.25)
    )
    ax.plot(goal_xy[0], goal_xy[1], marker="+", color="tab:green",
            markersize=14, markeredgewidth=2, label="goal")
    # Initial cube position (faint).
    ax.add_patch(
        _patch_rect(ax,
                    cube0_xy[0] - cube_half_size, cube0_xy[1] - cube_half_size,
                    2 * cube_half_size, 2 * cube_half_size,
                    edgecolor="0.5", facecolor="0.85", linewidth=1, alpha=0.5)
    )
    # Blocked-side indicators (red bars on the side of the cube the approach
    # is blocked from).
    bar_len = cube_half_size * 6
    bar_off = cube_half_size * 3
    for side in blocked_sides:
        if side == "from_minus_x":
            ax.plot(
                [cube0_xy[0] - bar_off, cube0_xy[0] - bar_off],
                [cube0_xy[1] - bar_len / 2, cube0_xy[1] + bar_len / 2],
                color="tab:red", linewidth=4, alpha=0.85,
            )
        elif side == "from_plus_x":
            ax.plot(
                [cube0_xy[0] + bar_off, cube0_xy[0] + bar_off],
                [cube0_xy[1] - bar_len / 2, cube0_xy[1] + bar_len / 2],
                color="tab:red", linewidth=4, alpha=0.85,
            )
        elif side == "from_minus_y":
            ax.plot(
                [cube0_xy[0] - bar_len / 2, cube0_xy[0] + bar_len / 2],
                [cube0_xy[1] - bar_off, cube0_xy[1] - bar_off],
                color="tab:red", linewidth=4, alpha=0.85,
            )
        elif side == "from_plus_y":
            ax.plot(
                [cube0_xy[0] - bar_len / 2, cube0_xy[0] + bar_len / 2],
                [cube0_xy[1] + bar_off, cube0_xy[1] + bar_off],
                color="tab:red", linewidth=4, alpha=0.85,
            )


def _patch_rect(ax, x, y, w, h, **kw):
    from matplotlib.patches import Rectangle
    r = Rectangle((x, y), w, h, **kw)
    return r


def _patch_circle(ax, center, radius, **kw):
    from matplotlib.patches import Circle
    c = Circle(center, radius, **kw)
    return c


def _draw_cube_frame(ax, cube_xy, cube_half_size, color, alpha=1.0):
    """Add a cube patch at xy. Returns the patch so it can be removed."""
    r = _patch_rect(ax,
                    cube_xy[0] - cube_half_size, cube_xy[1] - cube_half_size,
                    2 * cube_half_size, 2 * cube_half_size,
                    edgecolor=color, facecolor=color, alpha=alpha, linewidth=1.5)
    ax.add_patch(r)
    return r


def _draw_arrow(ax, from_xy, to_xy, color, label=None):
    ax.annotate(
        "", xy=to_xy, xytext=from_xy,
        arrowprops=dict(arrowstyle="->", color=color, lw=2),
    )
    if label:
        mid = (0.5 * (from_xy[0] + to_xy[0]), 0.5 * (from_xy[1] + to_xy[1]))
        ax.text(mid[0], mid[1] + 0.01, label, color=color, fontsize=8,
                ha="center")


def render_episode_topdown(
    *,
    out_path: Path,
    episode_id: str,
    cube0_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    blocked_sides: Sequence[str],
    demo_trajectory: Sequence[tuple[float, float]],
    initial_intent: dict,
    revised_intent: Optional[dict],
    attempt1_trajectory: Sequence[tuple[float, float]],
    attempt1_planner_failed: bool,
    attempt2_trajectory: Optional[Sequence[tuple[float, float]]],
    attempt2_success: Optional[bool],
    fps: int = 12,
    cube_half_size: float = 0.02,
    goal_radius: float = 0.025,
) -> Path:
    """Render one MP4 with three phases: demo → blocked attempt → revised retry.

    Each phase is rendered as a sequence of frames; static elements (goal,
    blocked-side indicators) stay across all phases.
    """
    _, plt = _import_mpl()
    import imageio.v2 as imageio  # ffmpeg available via imageio_ffmpeg

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    world_extent = max(abs(goal_xy[0]), abs(goal_xy[1])) + 0.08
    pad_frames = max(1, fps // 3)

    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=8,
        macro_block_size=1,
    )

    def _emit_phase(label: str, traj: Sequence[tuple[float, float]],
                    cube_color: str, retry: bool = False):
        for i in range(len(traj)):
            fig, ax = _new_figure(plt, world_extent)
            _draw_static(ax, cube0_xy=cube0_xy, goal_xy=goal_xy,
                         goal_radius=goal_radius, cube_half_size=cube_half_size,
                         blocked_sides=blocked_sides)
            xy = traj[i]
            _draw_cube_frame(ax, xy, cube_half_size, color=cube_color)
            if i > 0:
                xs = [p[0] for p in traj[: i + 1]]
                ys = [p[1] for p in traj[: i + 1]]
                ax.plot(xs, ys, color=cube_color, linewidth=1, alpha=0.6)
            ax.set_title(f"{episode_id}\n{label}", fontsize=9, loc="left")
            _annotate_intents(ax, initial_intent, revised_intent, retry=retry)
            # No tight_layout — we set subplots_adjust manually so the
            # right-side annotations have room.
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            writer.append_data(buf)
            plt.close(fig)

    def _emit_planner_failed_frames(label: str, n_frames: int):
        for _ in range(n_frames):
            fig, ax = _new_figure(plt, world_extent)
            _draw_static(ax, cube0_xy=cube0_xy, goal_xy=goal_xy,
                         goal_radius=goal_radius, cube_half_size=cube_half_size,
                         blocked_sides=blocked_sides)
            _draw_cube_frame(ax, cube0_xy, cube_half_size, color="tab:red")
            ax.text(
                0, 0.07,
                "approach BLOCKED\nplanner_failed=true",
                color="tab:red", fontsize=11, ha="center",
                fontweight="bold",
            )
            ax.set_title(f"{episode_id}\n{label}", fontsize=9, loc="left")
            _annotate_intents(ax, initial_intent, revised_intent, retry=False)
            # No tight_layout — we set subplots_adjust manually so the
            # right-side annotations have room.
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            writer.append_data(buf)
            plt.close(fig)

    # Phase 1: demo proxy.
    _emit_phase("phase 1/3 — third-person demo proxy (oracle)",
                demo_trajectory, cube_color="tab:blue")
    for _ in range(pad_frames):
        # Brief still on the demo's final frame.
        fig, ax = _new_figure(plt, world_extent)
        _draw_static(ax, cube0_xy=cube0_xy, goal_xy=goal_xy,
                     goal_radius=goal_radius, cube_half_size=cube_half_size,
                     blocked_sides=blocked_sides)
        _draw_cube_frame(ax, demo_trajectory[-1], cube_half_size,
                          color="tab:blue", alpha=0.4)
        ax.set_title(
            f"{episode_id}\nphase 1/3 — demo done (contact_region="
            f"{initial_intent['contact_region']})",
            fontsize=9, loc="left",
        )
        _annotate_intents(ax, initial_intent, revised_intent, retry=False)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(buf)
        plt.close(fig)

    # Phase 2: attempt 1 (blocked → planner_failed).
    if attempt1_planner_failed:
        _emit_planner_failed_frames(
            "phase 2/3 — Franka attempt 1: approach_blocked",
            fps * 2,
        )
    else:
        _emit_phase("phase 2/3 — Franka attempt 1",
                    attempt1_trajectory, cube_color="tab:red")

    # Phase 3: retry with revised intent.
    if attempt2_trajectory is not None:
        if revised_intent:
            label = (
                f"phase 3/3 — Franka retry (success={attempt2_success})\n"
                f"approach_substitution: "
                f"{initial_intent['approach_direction']} → "
                f"{revised_intent['approach_direction']}"
            )
        else:
            label = "phase 3/3 — Franka retry"
        _emit_phase(label, attempt2_trajectory,
                    cube_color="tab:green" if attempt2_success else "tab:red",
                    retry=True)

    writer.close()
    return out_path


def _annotate_intents(ax, initial_intent: dict,
                      revised_intent: Optional[dict], *, retry: bool) -> None:
    """Two text boxes on the right side: initial intent + revised intent."""
    lines_init = [
        "initial intent:",
        f"  goal_state:        {initial_intent['goal_state']}",
        f"  object_motion:     {initial_intent['object_motion']}",
        f"  contact_region:    {initial_intent['contact_region']}",
        f"  approach_dir:      {initial_intent['approach_direction']}",
    ]
    if revised_intent is None:
        lines_rev = []
    else:
        diff = []
        for k in ("goal_state", "object_motion", "contact_region",
                  "approach_direction"):
            if revised_intent[k] != initial_intent[k]:
                diff.append(k)
        lines_rev = [
            "revised intent:",
            f"  approach_dir:      {revised_intent['approach_direction']}"
            + ("   ← REVISED" if "approach_direction" in diff else ""),
        ]
    ax.text(
        1.02, 0.99, "\n".join(lines_init),
        transform=ax.transAxes, fontsize=8,
        verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="0.95",
                  edgecolor="0.7"),
    )
    if lines_rev:
        ax.text(
            1.02, 0.55, "\n".join(lines_rev),
            transform=ax.transAxes, fontsize=8,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#e8f5e9" if retry else "0.95",
                      edgecolor="tab:green" if retry else "0.7"),
        )
