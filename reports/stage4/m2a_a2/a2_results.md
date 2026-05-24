# Stage-4 M2a Stage A2 — Joint IntentHead + ReviseHead

Per task: joint IntentHead training across all non-trivial intent factors; per-factor centroid `slot_decode`; single-slot ReviseHead trained on counterfactual (g_pre, fp) → centroid[revised_class] pairs.

IntentHead: F=6, d_slot=16, hidden=64, n_epochs=200, lr=0.01.
ReviseHead: d_slot=16, fp_dim=15, hidden=64, n_epochs=400, L2 loss.

Cert metrics:
- **G2 (frozen-slot preservation)**: max ℓ2 drift of unedited slots after `apply_revision`. Spec gate: ≤ ε. Deterministic encoder → ε = 0 by construction (`apply_revision` only writes the implicated slot).
- **Revised-slot decode acc**: on held-out test folds, does `decode_slot(apply_revision(G, factor_idx, fp)[slot])` match the ground-truth `revision.new_value`? This is the more interesting number — measures whether ReviseHead actually moves the slot to the right centroid, not just that the type signature works.

## Per-task headline

| task | n_episodes | n_revisions | certable | uncertable | revised-slot decode acc | G2 max drift | G2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PushCube-v1 | 20 | 20 | 20 | 0 | 1.000 | 0.00e+00 | PASS |
| StackCube-v1 | 40 | 40 | 0 | 40 | n/a (0 certable) | 0.00e+00 | PASS |

## Revision factor distribution (per task)

- **PushCube-v1**: {'approach_direction': 20}
- **StackCube-v1**: {'approach_direction': 6, 'goal_state': 34}

## Per-fold detail (machine-readable in JSON)
