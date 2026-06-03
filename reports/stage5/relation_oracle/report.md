# Stage-5 — StackCube `object_motion` label well-posedness (position oracle) — StackCube-v1

> **n=200 update (2026-06-02).** The DINOv2 spatial_mean reference cited below
> (**0.42**) is n=40; at **n=200** it is **0.68** (`reports/stage5/p1_vision_g1_n200`).
> So the deficit vs this position oracle is real but ~half what 0.42 implied, and
> object-LOCAL DINO pooling does not close it
> (`reports/stage5/object_relation_probe_n200`). This well-posedness conclusion
> (label is a noiseless function of positions) is unaffected.

**Question:** is `object_motion` a clean function of object POSITIONS — i.e. is
the frozen-DINOv2 spatial_mean **0.42** a representation failure or a label/data
problem? (This is a well-posedness check; the relation == the label's own input,
so it is near-tautological by construction — see Interpretation.)

- Source: p2_vlm_wrist/StackCube-v1 rollouts (initial_obj_xy, goal_xy) + episodes.jsonl
- Cut: n=50, seeds 100-149, label dist {'translate_+x': 5, 'translate_+y': 17, 'translate_-x': 17, 'translate_-y': 11} (majority 0.340)
- Feature: cubeA **start** + cubeB resting positions (privileged sim obs). The label is `goal_direction_to_motion(cubeB_init - cubeA_init)` (`scene.py:cubeA_to_cubeB_motion`, assigned at reset), so `(cubeB - cubeA_start)`
  IS the label's own input — this ladder is a **well-posedness / zero-label-noise** check, not a non-circular recovery.
- Reference: DINOv2 spatial_mean object_motion = **0.42** (`reports/stage5/p1_vision_g1`).

## Feature ladder

| feature | dim | majority | shuffled | direct LR ± std | gate | IntentHead-CV ± std | gate |
|---|---|---|---|---|---|---|---|
| `A0 (cubeA start)` | 2 | 0.340 | 0.280 | 0.540 ± 0.049 | FAIL | 0.500 ± 0.063 | FAIL |
| `B (cubeB)` | 2 | 0.340 | 0.340 | 0.560 ± 0.162 | FAIL | 0.520 ± 0.117 | FAIL |
| `[A0;B] (concat)` | 4 | 0.340 | 0.320 | 0.860 ± 0.120 | FAIL | 0.940 ± 0.049 | PASS |
| `B-A0 (relative)` | 2 | 0.340 | 0.360 | 0.980 ± 0.040 | PASS | 0.940 ± 0.049 | PASS |

**Parameter-free dominant-axis rule on `(cubeB - cubeA_start)`: 0.940** (no fitting, full cut; this is literally the label's own `goal_direction_to_motion` applied to its own input → ≈1.0 on an exact cut).

## Interpretation

- The label is a **deterministic, parameter-free, noiseless** function of the two
  cubes' resting positions: the dominant-axis rule on `(cubeB - cubeA_start)` matches it at **0.940** (≈1.0 on an exact cut; any gap here is only the
  cross-cut `goal_xy`≈cubeB approximation). There is **no label noise**.
- Neither cube's *absolute* position predicts direction (`A0` 0.540, `B` 0.560 ≈ chance); only the **relation** `B - A0` does (0.980) — the
  object-centric hypothesis, stated in coordinates.
- **Conclusion:** because the label is a clean, noiseless geometric function of
  resting positions, the frozen-DINOv2 **0.42** is a **representation**
  failure, not a label/data problem.
- **But note the limit:** this ladder uses privileged coordinates and is
  near-tautological by construction (the relation == the label input). It does
  NOT show that any *learned feature* recovers the relation. The non-trivial
  representation evidence is the **blob image-position probe ~0.80 vs global
  DINO 0.42** on the same frames; Step-2 (`stage5_object_relation_probe.py`)
  tests object-local DINO tokens from pixels, no coordinates fed.

## Caveats

- **Near-tautological by construction:** the label IS `goal_direction_to_motion(cubeB - cubeA)`, so the relational feature and the rule
  re-derive it. This certifies the label is noiseless & well-posed and that
  absolute single-object position is insufficient — it is NOT evidence that a
  learned representation recovers the relation. That is Step-2's job.
- **Cross-cut** when `--source p2_wrist`: this cut is seeds 100-149 (P2 eval, imbalanced), not the balanced 0-39 cut that produced 0.42. The conclusion is
  cut-agnostic; the exactly-matched 0-39 path is
  `stage5_extract_cube_positions.py` (GPU) → `--source dir`.
- Positions are privileged sim obs, used here only as a well-posedness check for
  representation development (CLAUDE.md invariant #4 — they must NOT become the
  deployable demo→intent path; Step-2 extracts the relation from pixels).

