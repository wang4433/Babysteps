# Stage-4 Schema-Recoverability Probe

Gate: non-trivial cells must reach probe_acc_mean >= 0.90.
Cells: 18 total | 0 pass | 1 fail | 17 trivially constant.

### PickCube-v1

| factor | n_unique | n_episodes | majority | shuffled | probe ± std | gate |
| --- | --- | --- | --- | --- | --- | --- |
| approach_direction | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| constraint_region | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| contact_region | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| embodiment_mapping | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| goal_state | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| object_motion | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |

### PushCube-v1

| factor | n_unique | n_episodes | majority | shuffled | probe ± std | gate |
| --- | --- | --- | --- | --- | --- | --- |
| approach_direction | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| constraint_region | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| contact_region | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| embodiment_mapping | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| goal_state | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| object_motion | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |

### StackCube-v1

| factor | n_unique | n_episodes | majority | shuffled | probe ± std | gate |
| --- | --- | --- | --- | --- | --- | --- |
| approach_direction | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| constraint_region | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| contact_region | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| embodiment_mapping | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| goal_state | 1 | 24 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| object_motion | 4 | 24 | 0.42 | 0.54 | 0.75 ± 0.43 | FAIL |

**Failing cells (need a notes.md explanation):** StackCube-v1/object_motion

