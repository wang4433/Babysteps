# docs/ — specs, plans, claim, archive

Written design history. The flow for each sub-project is **spec → plan → code**:
a dated design spec, then a TDD implementation plan, then the code under
`babysteps/`.

## Layout

| Path | Job |
| --- | --- |
| `superpowers/specs/` | Design spec per sub-project (dated `YYYY-MM-DD-...-design.md`). The intended behavior, schema deltas, and acceptance gate. |
| `superpowers/plans/` | TDD implementation plan per sub-project (dated `...-plan.md`). The ordered task list that produced the code. |
| `milestone1_locked_claim.md` | The locked paper claim (thesis, intent factors, failure predicates, table design). |
| `cold_start_handover_archive.md` | Historical long-form handover. **Drifted schema** — reference only; `goal.md` is authoritative. |

## Sub-project ↔ spec/plan map

| ID | Spec / plan stem |
| --- | --- |
| A PushCube | `2026-05-15-stage0-pushcube-blocked-*` |
| (refactor) | `2026-05-16-stage0-task-adapter-refactor-*` |
| B PickCube | `2026-05-17-stage0-pickcube-b-plan` (roadmap: `2026-05-17-stage0-four-scene-roadmap-design`) |
| C StackCube | `2026-05-17-stage0-stackcube-c-*` |
| D TurnFaucet | `2026-05-17-stage0-turnfaucet-d-*`, superseded by `2026-05-18-stage0-turnfaucet-embodiment-*` |
| E CrossViewPush | `2026-05-19-stage0-crossview-grounding-*` |

## Rules

- **Follow an existing spec when one covers the work** — skip re-brainstorming
  and go straight to the plan/code.
- A new sub-project gets a spec *and* a plan before code. Date the filenames.
- When a spec is superseded, leave a supersession note in the old file rather
  than deleting it (see the TurnFaucet pair).
