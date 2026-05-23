# Stage-4 Schema-Recoverability Probe

Gate: geometric cells must reach probe_acc_mean >= 0.90 AND beat chance & shuffled each by 0.10.
Cells: 12 total | 2 geometric (1 pass / 1 fail) | 2 label-identity | 8 trivially constant.

### PushCube-v1

| factor | class | n_unique | n_episodes | majority | shuffled | probe ± std | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| approach_direction | label_identity | 2 | 20 | 0.50 | 0.65 | 1.00 ± 0.00 | label_identity |
| constraint_region | trivially_constant | 1 | 20 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| contact_region | label_identity | 2 | 20 | 0.50 | 0.65 | 1.00 ± 0.00 | label_identity |
| embodiment_mapping | trivially_constant | 1 | 20 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| goal_state | trivially_constant | 1 | 20 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| object_motion | geometric | 3 | 20 | 0.50 | 0.60 | 0.95 ± 0.22 | PASS |

### StackCube-v1

| factor | class | n_unique | n_episodes | majority | shuffled | probe ± std | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| approach_direction | trivially_constant | 1 | 40 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| constraint_region | trivially_constant | 1 | 40 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| contact_region | trivially_constant | 1 | 40 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| embodiment_mapping | trivially_constant | 1 | 40 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| goal_state | trivially_constant | 1 | 40 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| object_motion | geometric | 4 | 40 | 0.25 | 0.23 | 0.72 ± 0.05 | FAIL |

**Failing geometric cells (need a notes.md explanation):** StackCube-v1/object_motion

