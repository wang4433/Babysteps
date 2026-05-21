# babysteps/skills/ — intent → skill compilers

The deterministic interface between a structured `Intent` and an executable
robot skill. Each compiler takes an `Intent` (+ scene) and returns a
parameterized skill (waypoints / motion-plan parameters / contact poses). No
torque, no learning — explicit and deterministic.

## Files

| File | Job |
| --- | --- |
| `push.py` | Push compiler — `Intent` → executable `PushSkill` (PushCube / CrossViewPush). |
| `pick.py` | Pick compiler — `Intent` → executable `PickSkill` (PickCube). |
| `stack.py` | Stack compiler — `Intent` → executable `StackSkill` (StackCube). |
| `turn.py` | Turn compiler — Sub-project D embodiment dispatch: grasp-turn vs poke-turn (embodiment_substitution). |

## Rules

- Compilers are **deterministic**. Same intent + scene → same skill.
- A factor revision changes the compiled skill *only through the changed
  factor* — e.g. flipping `approach_direction` changes the contact side, not
  the goal. This is what makes single-factor revision observable in execution.
- Grasp failures are fixed with a real grasp (OBB grasp pose + orientation +
  tangent pull), **never** by widening waypoint tolerances.
