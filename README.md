# BABYSTEPS

Failure-guided structured-intent revision for Franka manipulation.

The Stage-0 goal (per `goal.md`) is to validate the data contract and the
factor-local revision loop on `ManiSkill PushCube-v1` before adding learned
perception, real human demonstrations, or real-robot execution.

> Stage-0 boundary: data inputs are **third-person demonstration proxies**,
> not human demonstrations. Read `goal.md` and the design spec before adding
> new modules or schema fields.

## Authoritative documents

1. `goal.md` — Stage-0 boundary and data contract (**authority**).
2. `CLAUDE.md` — high-level project instructions and working invariants.
3. `CODE_MAP.md` — one-screen map of every directory (each also has a `CLAUDE.md`).
4. `RUNBOOK.md` — copy-paste operational commands (render / collect / summarize / tests).
5. `docs/superpowers/specs/2026-05-15-stage0-pushcube-blocked-design.md` and the
   matching `docs/superpowers/plans/...-plan.md` — the design + TDD plan this
   implementation follows.
6. `docs/cold_start_handover_archive.md` — historical long-form handover; its
   schema has drifted, so **`goal.md` is the Stage-0 authority** if anything disagrees.

## Quickstart

```bash
conda activate handover    # Gilbreth's pre-existing env with ManiSkill 3
cd /home/wang4433/scratch/babysteps

# 1. Unit tests (sim-free; runs on the login node).
python -m pytest tests/ -q

# 2. End-to-end data collection with the fake env (no Vulkan / GPU needed).
python scripts/stage0_collect.py \
    --out_dir datasets/stage0_pushcube_blocked \
    --n_episodes 5 --seed_start 0 --fake-env

# 3. Real ManiSkill collection (needs a GPU+Vulkan-capable compute node).
# On Gilbreth: allocate one first, e.g.
#   salloc --gres=gpu:1 --time=00:30:00
# then on the allocated node:
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
python scripts/stage0_collect.py \
    --out_dir datasets/stage0_pushcube_blocked \
    --n_episodes 5 --seed_start 0

# 4. Re-summarize an existing samples.jsonl into report.{md,json}.
python scripts/stage0_summarize.py \
    --samples datasets/stage0_pushcube_blocked/samples.jsonl \
    --out_dir datasets/stage0_pushcube_blocked
```

## Layout

```
babysteps/                  # Python package
  schemas.py                # Intent, FailurePacket, EpisodeRecord, ...
  demo.py                   # demo evidence → Intent (privileged firewall)
  execution.py              # push skill compiler + feasibility check
  failure.py                # FailurePacket builder + attribution rules
  revision.py               # approach_substitution operator (Stage 0)
  episode.py                # the run_episode loop (sim-agnostic)
  eval.py                   # dataset-level metrics + report writer
  envs/
    scene.py                # face / approach / motion helpers (pure)
    pushcube_runner.py      # ManiSkill env_runner (ONLY mani_skill import)

scripts/
  smoke_pushcube.py         # ManiSkill loadability check (compute node only)
  stage0_collect.py         # CLI: N episodes → samples.jsonl + report
  stage0_summarize.py       # CLI: samples.jsonl → report

tests/                      # 85 sim-free unit tests; one file per pure module
datasets/
  stage0_pushcube_blocked/  # run output (gitignored)
```

## Stage-0 acceptance criteria (spec §13)

A Stage-0 run is good when:

- `pytest tests/` passes (sim-free).
- `samples.jsonl` has one line per episode, all with:
  - `claim_boundary == "third_person_demo_proxy_not_human_demo"`,
  - `demo.demonstrator_type == "proxy_oracle"`,
  - `failure_packet.failure_predicate == "approach_blocked"`,
  - `revision.operator == "approach_substitution"`,
  - `retry.success == True`.
- `report.md` reports `passed_acceptance: True` (`delta_pp >= 10`).

## Limitations (Stage 0)

- The "block" is a privileged feasibility flag, not a physical obstacle.
  A future stage will add a real obstacle via a PushCube subclass — see
  spec §14.
- No RGB / DINO / VLM yet. The demo proxy carries only object trajectory +
  contact-region label.
- One failure predicate (`approach_blocked`). Other predicate paths
  (`contact_failure`, `direction_error`, …) are reserved in the schema
  but raise `NotImplementedError` in the Stage-0 reviser.

See spec §14 ("Out-of-scope, captured for follow-on stages") for the full
list.
