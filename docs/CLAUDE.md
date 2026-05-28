# docs/ — specs, plans, claim, archive

Written design history. The flow for each sub-project is **spec → plan → code**:
a dated design spec, then a TDD implementation plan, then the code under
`babysteps/`.

## Layout

| Path | Job |
| --- | --- |
| `superpowers/specs/` | Design spec per sub-project (dated `YYYY-MM-DD-...-design.md`). The intended behavior, schema deltas, and acceptance gate. Kept as long as code/tests/reports cite them. |
| `superpowers/plans/` | TDD implementation plan per sub-project (dated `...-plan.md`). The ordered task list that produced the code. Deleted after the work lands — git history preserves them. |
| `milestone1_locked_claim.md` | The locked paper claim (thesis, intent factors, failure predicates, table design). |
| `cold_start_handover_archive.md` | Historical long-form handover. **Drifted schema** — reference only; `goal.md` is authoritative. |

## Sub-project ↔ spec map

The plans for completed sub-projects have been pruned (git history retains
them); the specs below remain because code, tests, and reports cite them.

| ID | Spec stem | Status |
| --- | --- | --- |
| A PushCube | `2026-05-15-stage0-pushcube-blocked-design` | done (Stage-0 clutter render; paper-figure render reframed to `contact_region` — see `redesign_failure_paradigm.md`) |
| B PickCube | `2026-05-17-stage0-four-scene-roadmap-design` | done |
| C StackCube | `2026-05-17-stage0-stackcube-c-design` | done |
| D TurnFaucet | `2026-05-17-stage0-turnfaucet-d-design`, superseded by `2026-05-18-stage0-turnfaucet-embodiment-design` | done; poke-turn fix is the open follow-up |
| E CrossViewPush | `2026-05-19-stage0-crossview-grounding-design` | done |
| Stage-0 baselines | `2026-05-20-stage0-baselines-design` | done |
| Stage-4 varied intent cut | `2026-05-22-stage4-varied-intent-cut-design` | done |
| Stage-4 M2 slot encoder | `2026-05-23-stage4-m2-slot-encoder-design` | done |
| Stage-4 M2.5 attribution head | `2026-05-23-stage4-m2.5-attribution-head-design` | done |
| Stage-5 P1 vision encoder swap | `2026-05-24-stage5-vision-encoder-swap-design` | done (PushCube held-out latent matches oracle) |
| Stage-5 P3 world model | `2026-05-26-stage5-p3-world-model-design` (+ matching plan) | **active** |

## Rules

- **Follow an existing spec when one covers the work** — skip re-brainstorming
  and go straight to the plan/code.
- A new sub-project gets a spec *and* a plan before code. Date the filenames.
- **Specs**: when superseded by a newer spec, leave a supersession note in
  the old file rather than deleting it (see the TurnFaucet pair). Delete only
  when no code/tests/reports reference the spec by path.
- **Plans**: delete once the implementation has landed and the work is
  reflected in code + commit history. Plans are ephemeral TDD task lists; the
  surviving artifacts are the spec, the commits, and the running code.
