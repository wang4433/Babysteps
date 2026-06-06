# Stage-5 P1 — Vision-grounded G1 (V-JEPA 2.1 ViT-L/16 (clip token-mean))

Input Z: 1024-dim V-JEPA 2.1 ViT-L/16 (clip token-mean) features (token_mean pool over demo frames).
IntentHead: F=6, d_slot=32, hidden=256, n_epochs=300, lr=0.01.
Outer CV: per-fold IntentHead training; frozen LogisticRegression on G_train, evaluated on G_test.

### StackCube-v1

| factor | class | n_unique | n_episodes | majority | shuffled | probe ± std | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| approach_direction | trivially_constant | 1 | 200 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| constraint_region | trivially_constant | 1 | 200 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| contact_region | trivially_constant | 1 | 200 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| embodiment_mapping | trivially_constant | 1 | 200 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| goal_state | trivially_constant | 1 | 200 | 1.00 | 1.00 | 1.00 ± 0.00 | trivial |
| object_motion | geometric | 4 | 200 | 0.27 | 0.20 | 0.54 ± 0.24 | FAIL |


**Gate:** all geometric cells >= 90% (margin 10% over majority & shuffled) -> **FAIL**

Cells: 6 total | 1 geometric (0 pass / 1 fail) | 0 label-identity | 5 trivially constant.

**Failing geometric cells:** StackCube-v1/object_motion

