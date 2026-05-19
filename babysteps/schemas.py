"""Pure data contracts for BABYSTEPS Stage 0.

All dataclasses are frozen and JSON-roundtrippable. Whitelist-validation in
__post_init__ catches schema drift early. The shape of EpisodeRecord matches
goal.md §"Episode Data Format" — see test_schemas.py for the snapshot guard.

No simulator, no privileged-state side channels: this module only defines
shapes. The privileged-firewall (goal.md §5: "Keep simulator privileged state
out of the demo-to-intent input path") is enforced by where these are passed
in, not by this module — see test_episode.py for the call-shape guard.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

# ---------- Module-level constants & whitelists -------------------------- #

INTENT_FIELDS: tuple[str, ...] = (
    "goal_state",
    "object_motion",
    "contact_region",
    "approach_direction",
    "constraint_region",
    "embodiment_mapping",
)

CONTACT_REGIONS: frozenset[str] = frozenset({
    "minus_x_face", "plus_x_face", "minus_y_face", "plus_y_face",
    "faucet_base",   # D: TurnFaucet — static body of the faucet (wrong contact)
    "handle_grip",   # D: TurnFaucet — rotating handle (correct contact)
})
APPROACH_DIRECTIONS: frozenset[str] = frozenset({
    "from_minus_x", "from_plus_x", "from_minus_y", "from_plus_y", "from_above",
})
OBJECT_MOTIONS: frozenset[str] = frozenset({
    "translate_+x", "translate_-x", "translate_+y", "translate_-y",
    "lift_up",   # B: PickCube — cube lifted along +z
    "place_on",  # C: StackCube — cube placed on top of another cube
    "turn",      # D: TurnFaucet — handle rotated around joint axis
})
EMBODIMENT_MAPPINGS: frozenset[str] = frozenset({
    "proxy_contact_to_franka_push",
    "proxy_contact_to_franka_grasp",   # B: PickCube — parallel-jaw grasp
    "proxy_contact_to_franka_pick_and_place",  # C: StackCube — pick + place sequence
    "proxy_contact_to_franka_turn",    # D: deprecated, kept in whitelist
    "proxy_contact_to_franka_grasp_turn",      # D: NEW — initial intent
    "proxy_contact_to_franka_poke_turn",       # D: NEW — revised intent
})
GOAL_STATES: frozenset[str] = frozenset({
    "cube_at_target",
    "cube_lifted_at_target",           # B: PickCube — cube lifted to goal xyz
    "cubeA_on_cubeB",                  # C: StackCube — cubeA resting atop cubeB
    "faucet_turned",                   # D: TurnFaucet — handle rotated past target
})
CONSTRAINT_REGIONS: frozenset[str] = frozenset({
    "none",
    "faucet_base_static",   # D: TurnFaucet — body must not be displaced
})
DIRECTION_GROUNDINGS: frozenset[str] = frozenset({
    "actor_frame",       # E: cross-view — egocentric (identity) grounding; the bug
    "observer_frame",    # E: cross-view — account for the observer camera yaw; the fix
    "object_frame",      # E: reserved (later cut)
    "world_frame",       # E: default for non-cross-view tasks; inert
})

FAILURE_PREDICATES: frozenset[str] = frozenset({
    "none",
    "approach_blocked",
    "direction_error",
    "contact_failure",
    "no_motion",
    "goal_not_satisfied",
    "grasp_slip",                      # B: PickCube — grip lost during lift
    "constraint_violation",            # D: deprecated, kept in whitelist
    "grasp_infeasible",                # D: NEW — handle too thick to close jaw
})
REVISION_OPERATORS: frozenset[str] = frozenset({
    "approach_substitution",
    "contact_substitution",            # B: PickCube — rotate gripper axis
    "goal_refinement",                 # C: StackCube — sharpen under-specified goal
    "constraint_introduction",         # D: deprecated, kept in whitelist
    "embodiment_substitution",         # D: NEW — swap grasp_turn → poke_turn
    "grounding_substitution",          # E: cross-view — swap actor_frame → observer_frame
})

CLAIM_BOUNDARY: str = "third_person_demo_proxy_not_human_demo"
"""Stage-0 paper-claim guard string. Must appear on every EpisodeRecord."""


def _validate(value: str, allowed: frozenset[str], field_name: str) -> None:
    if value not in allowed:
        raise ValueError(
            f"{field_name} must be one of {sorted(allowed)}, got {value!r}"
        )


# ---------- Intent ------------------------------------------------------- #


@dataclass(frozen=True)
class Intent:
    """Stage-0 object-centric intent. The six factors are from goal.md
    §"Stage 0 Intent Factors". No task-specific fields."""

    goal_state: str
    object_motion: str
    contact_region: str
    approach_direction: str
    constraint_region: str
    embodiment_mapping: str

    def __post_init__(self) -> None:
        _validate(self.goal_state, GOAL_STATES, "goal_state")
        _validate(self.object_motion, OBJECT_MOTIONS, "object_motion")
        _validate(self.contact_region, CONTACT_REGIONS, "contact_region")
        _validate(self.approach_direction, APPROACH_DIRECTIONS, "approach_direction")
        _validate(self.constraint_region, CONSTRAINT_REGIONS, "constraint_region")
        _validate(self.embodiment_mapping, EMBODIMENT_MAPPINGS, "embodiment_mapping")

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in INTENT_FIELDS}

    @classmethod
    def from_dict(cls, d: dict) -> "Intent":
        return cls(**{f: d[f] for f in INTENT_FIELDS})


# ---------- DemoEvidence ------------------------------------------------- #


@dataclass(frozen=True)
class DemoEvidence:
    """What the proxy demo hands forward to the (scripted) intent extractor.

    No privileged fields: no goal_xy, no tcp pose, no blocked_sides. The
    contact_region_label is allowed because it is observable from the
    third-person view of the demo (it labels what was demonstrated)."""

    camera: str
    demonstrator_type: str
    object_trajectory: tuple[tuple[float, float], ...]
    contact_region_label: str
    final_state: str
    rgbd_video_path: Optional[str]

    def to_dict(self) -> dict:
        return {
            "camera": self.camera,
            "demonstrator_type": self.demonstrator_type,
            "object_trajectory": [list(p) for p in self.object_trajectory],
            "contact_region_label": self.contact_region_label,
            "final_state": self.final_state,
            "rgbd_video_path": self.rgbd_video_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DemoEvidence":
        return cls(
            camera=d["camera"],
            demonstrator_type=d["demonstrator_type"],
            object_trajectory=tuple(
                (float(p[0]), float(p[1])) for p in d["object_trajectory"]
            ),
            contact_region_label=d["contact_region_label"],
            final_state=d["final_state"],
            rgbd_video_path=d.get("rgbd_video_path"),
        )


# ---------- SceneState (every field is privileged) ----------------------- #


@dataclass(frozen=True)
class SceneState:
    """Simulator-side ground truth + feasibility flags.

    Every field is privileged — must not flow into demo_to_intent. Consumed
    only by the skill compiler (waypoint geometry + blocked_sides feasibility
    check) and by metric computation (oracle labels).

    `extra` is an adapter-owned payload for forward compatibility with non-
    push tasks (PickCube populates gripper_width etc.; StackCube populates a
    second cube's pose). It is serialized only when non-empty so PushCube
    records remain byte-identical to pre-A snapshots."""

    cube_xy: tuple[float, float]
    cube_z: float
    goal_xy: tuple[float, float]
    tcp_start_pose: tuple[float, float, float, float, float, float, float]
    blocked_sides: tuple[str, ...]
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "cube_xy": list(self.cube_xy),
            "cube_z": float(self.cube_z),
            "goal_xy": list(self.goal_xy),
            "tcp_start_pose": list(self.tcp_start_pose),
            "blocked_sides": list(self.blocked_sides),
        }
        if self.extra:
            d["extra"] = dict(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SceneState":
        cube_xy = tuple(float(v) for v in d["cube_xy"])
        goal_xy = tuple(float(v) for v in d["goal_xy"])
        tcp = tuple(float(v) for v in d["tcp_start_pose"])
        if len(cube_xy) != 2 or len(goal_xy) != 2:
            raise ValueError("cube_xy and goal_xy must have length 2")
        if len(tcp) != 7:
            raise ValueError(f"tcp_start_pose must have length 7, got {len(tcp)}")
        return cls(
            cube_xy=cube_xy,          # type: ignore[arg-type]
            cube_z=float(d["cube_z"]),
            goal_xy=goal_xy,          # type: ignore[arg-type]
            tcp_start_pose=tcp,        # type: ignore[arg-type]
            blocked_sides=tuple(d["blocked_sides"]),
            extra=dict(d.get("extra", {})),
        )


# ---------- AttemptResult ------------------------------------------------ #


@dataclass(frozen=True)
class AttemptResult:
    """Per-attempt evidence emitted by the env_runner. JSON-able.

    `trajectory_xy` is the cube xy path over the rollout — used by
    `generate_proxy_demo` to populate DemoEvidence.object_trajectory.
    Empty tuple is the default (planner_failed attempts and the schema
    test fixtures use that)."""

    initial_obj_xy: tuple[float, float]
    final_obj_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    reached_contact: bool
    object_moved: bool
    planner_failed: bool
    collision: bool
    grasp_slip: bool
    rollout_log_path: Optional[str]
    success: bool
    trajectory_xy: tuple[tuple[float, float], ...] = ()

    def to_dict(self) -> dict:
        return {
            "initial_obj_xy": list(self.initial_obj_xy),
            "final_obj_xy": list(self.final_obj_xy),
            "goal_xy": list(self.goal_xy),
            "reached_contact": bool(self.reached_contact),
            "object_moved": bool(self.object_moved),
            "planner_failed": bool(self.planner_failed),
            "collision": bool(self.collision),
            "grasp_slip": bool(self.grasp_slip),
            "rollout_log_path": self.rollout_log_path,
            "success": bool(self.success),
            "trajectory_xy": [list(p) for p in self.trajectory_xy],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AttemptResult":
        return cls(
            initial_obj_xy=tuple(float(v) for v in d["initial_obj_xy"]),  # type: ignore[arg-type]
            final_obj_xy=tuple(float(v) for v in d["final_obj_xy"]),      # type: ignore[arg-type]
            goal_xy=tuple(float(v) for v in d["goal_xy"]),                # type: ignore[arg-type]
            reached_contact=bool(d["reached_contact"]),
            object_moved=bool(d["object_moved"]),
            planner_failed=bool(d["planner_failed"]),
            collision=bool(d["collision"]),
            grasp_slip=bool(d["grasp_slip"]),
            rollout_log_path=d.get("rollout_log_path"),
            success=bool(d["success"]),
            trajectory_xy=tuple(
                (float(p[0]), float(p[1])) for p in d.get("trajectory_xy", [])
            ),
        )


# ---------- FailurePacket ------------------------------------------------ #


@dataclass(frozen=True)
class FailurePacket:
    """The structured failure observation, per goal.md §"Build Failure Packet"."""

    chosen_intent: Intent
    execution_trace: dict
    failure_predicate: str
    object_displacement: Optional[float]
    direction_alignment: Optional[float]

    def __post_init__(self) -> None:
        _validate(self.failure_predicate, FAILURE_PREDICATES, "failure_predicate")

    def to_dict(self) -> dict:
        return {
            "chosen_intent": self.chosen_intent.to_dict(),
            "execution_trace": dict(self.execution_trace),
            "failure_predicate": self.failure_predicate,
            "object_displacement": self.object_displacement,
            "direction_alignment": self.direction_alignment,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FailurePacket":
        return cls(
            chosen_intent=Intent.from_dict(d["chosen_intent"]),
            execution_trace=dict(d["execution_trace"]),
            failure_predicate=d["failure_predicate"],
            object_displacement=d.get("object_displacement"),
            direction_alignment=d.get("direction_alignment"),
        )


# ---------- Revision ----------------------------------------------------- #


@dataclass(frozen=True)
class Revision:
    """One factor-local edit, per goal.md §"Revise and Retry"."""

    operator: str
    factor: str
    old_value: str
    new_value: str
    frozen_factors: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate(self.operator, REVISION_OPERATORS, "operator")

    def to_dict(self) -> dict:
        return {
            "operator": self.operator,
            "factor": self.factor,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "frozen_factors": list(self.frozen_factors),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Revision":
        return cls(
            operator=d["operator"],
            factor=d["factor"],
            old_value=d["old_value"],
            new_value=d["new_value"],
            frozen_factors=tuple(d["frozen_factors"]),
        )


# ---------- EpisodeRecord ------------------------------------------------ #


@dataclass(frozen=True)
class EpisodeRecord:
    """One Stage-0 episode. Shape matches goal.md §"Episode Data Format".

    Top-level dict children (demo/execution/failure_packet/revision/retry/
    metrics) are plain dicts rather than typed dataclasses, deliberately:
    they collect heterogeneous fields whose shape is dictated by goal.md and
    will evolve faster than the typed inner dataclasses (Intent, DemoEvidence,
    …) that live inside them. The typed pieces are still validated where they
    are constructed (see Intent.__post_init__ etc.)."""

    episode_id: str
    stage: str
    task: str
    claim_boundary: str
    demo: dict
    execution: dict
    failure_packet: dict
    revision: Optional[dict]
    retry: Optional[dict]
    metrics: dict

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "stage": self.stage,
            "task": self.task,
            "claim_boundary": self.claim_boundary,
            "demo": dict(self.demo),
            "execution": dict(self.execution),
            "failure_packet": dict(self.failure_packet),
            "revision": dict(self.revision) if self.revision is not None else None,
            "retry": dict(self.retry) if self.retry is not None else None,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeRecord":
        return cls(
            episode_id=d["episode_id"],
            stage=d["stage"],
            task=d["task"],
            claim_boundary=d["claim_boundary"],
            demo=dict(d["demo"]),
            execution=dict(d["execution"]),
            failure_packet=dict(d["failure_packet"]),
            revision=dict(d["revision"]) if d.get("revision") is not None else None,
            retry=dict(d["retry"]) if d.get("retry") is not None else None,
            metrics=dict(d.get("metrics", {})),
        )

    def to_jsonl_line(self) -> str:
        """One JSON object per line; sort_keys for deterministic diffs."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_jsonl_line(cls, line: str) -> "EpisodeRecord":
        return cls.from_dict(json.loads(line))
