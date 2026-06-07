"""PokeCube-v1 adapter — the SECOND ``contact_region`` family (build-order step 3).

PokeCube is a tool-mediated push: the Franka grasps a peg and POKES the cube to
a goal. It is the leave-one-task-family-out partner for PushCube because it
shares ``contact_region`` BOTH ways — the same 4 cube-face candidate vocab AND
the same 2D-residual→face revision rule with identical sign (to move the cube
+x you contact the −x face, in poke exactly as in push). That shared candidate
SEMANTICS + revision RULE is what makes a frozen shared policy's transfer to
PokeCube a genuine generalisation test rather than task memorisation.

The poke-vs-push difference is entirely in EXECUTION (the GPU runner grasps and
pokes via a peg); the intent factors — including ``contact_region`` and its
oracle/residual logic — are identical to PushCube. So this adapter is a thin
subclass that reuses PushCubeAdapter's contact_region semantics verbatim and
only changes the task id and the env-runner binding.

Sim-free (no mani_skill import at module load). The real PokeCubeEnvRunner is
build-order step-3 GPU work, gated behind the oracle-face poke-feasibility check.
"""
from __future__ import annotations

from babysteps.envs.pushcube_adapter import PushCubeAdapter
from babysteps.envs.task_adapter import EnvRunner


class PokeCubeAdapter(PushCubeAdapter):
    task_id = "PokeCube-v1"

    def make_env_runner(self) -> EnvRunner:
        # GPU grasp+poke runner — build-order step-3 GPU work (behind the
        # poke-feasibility kill-gate). Sim-free paths use FakePokeEnvRunner via
        # the task registry; this only fires for a real rollout.
        from babysteps.envs.pokecube_runner import PokeCubeEnvRunner
        return PokeCubeEnvRunner()
