# Stage-5 Step-2 — StackCube object-centric VISUAL relation probe — StackCube-v1

**Question:** do frozen DINOv2 patch tokens, pooled at the two cube
locations (location selects patches, never used as a feature value),
recover `object_motion` better than global mean-pooling (**0.685** at this
n=200; the original n=40 cell read 0.42)?

- n=200 (dropped 0 for missing blob), label dist {'translate_+x': 49, 'translate_+y': 50, 'translate_-x': 54, 'translate_-y': 47} (majority 0.270)
- Localization: pixel colour-blob centroids (cubeA=red, cubeB=green); patch pooling radius=1.
- Probe: same protocol as the original n=40 0.42 gate cell (IntentHead F=6, d_slot=32, n_epochs=300) + direct StandardScaler+LR.
- Label = `goal_direction_to_motion(cubeB_init - cubeA_init)`, so any COORDINATE re-feeds the label; headline rungs are appearance-only.

## Feature ladder

| feature | group | dim | majority | shuffled | direct LR ± std | IntentHead-CV ± std | gate |
|---|---|---|---|---|---|---|---|
| `global_dino (768)` | baseline | 768 | 0.270 | 0.220 | 0.670 ± 0.051 | 0.685 ± 0.104 | FAIL |
| `A_tok (cubeA local)` | appearance | 768 | 0.270 | 0.290 | 0.495 ± 0.046 | 0.355 ± 0.070 | FAIL |
| `B_tok (cubeB local)` | appearance | 768 | 0.270 | 0.305 | 0.715 ± 0.090 | 0.690 ± 0.093 | FAIL |
| `[A_tok;B_tok]` | appearance | 1536 | 0.270 | 0.310 | 0.680 ± 0.087 | 0.610 ± 0.080 | FAIL |
| `B_tok-A_tok (HEADLINE)` | appearance | 768 | 0.270 | 0.260 | 0.535 ± 0.103 | 0.560 ± 0.054 | FAIL |
| `A_tok@rand` | control | 768 | 0.270 | 0.290 | 0.255 ± 0.037 | 0.265 ± 0.080 | FAIL |
| `[rand;rand]` | control | 1536 | 0.270 | 0.300 | 0.250 ± 0.065 | 0.220 ± 0.076 | FAIL |
| `B_tok-A_tok @rand` | control | 768 | 0.270 | 0.275 | 0.265 ± 0.046 | 0.160 ± 0.066 | FAIL |
| `uv(B-A) image` | UPPER-BOUND | 2 | 0.270 | 0.260 | 0.495 ± 0.132 | 0.420 ± 0.048 | (near-tautological) |

## Verdict

- **Headline** (`B_tok-A_tok (HEADLINE)`, IntentHead-CV): **0.560** — *below* the n=200 `global_dino` baseline **0.685** (majority 0.270, image-uv upper bound 0.420).
- **NO LIFT.** At adequate n the global baseline rises to 0.685 (the 0.42 figure was an n=40 small-sample artifact), and object-local pooling does not beat it: `B_tok` alone (0.690) ≈ global, the `B_tok-A_tok` relation (0.560) is *worse*, and `A_tok` (0.355) is near chance. The object-centric-pooling hypothesis is **not supported**.
- **Random-location control** relation = 0.160 (≈ chance, well below the headline): the controls are clean, so the non-result is not masked by positional-encoding bleed — object-local pooling simply does not recover the relation that global pooling misses.

## Caveats

- **The claim is object-local pooling vs GLOBAL pooling** (both frozen DINOv2). The decision rule was: if the headline ≫ the n=200 global baseline (0.685), global mean-pooling destroys a relation that object-local pooling preserves → object-centric pooling is the fix. It does not (headline 0.560 < 0.685), so the claim fails here.
- **Appearance vs positional encoding:** a lift could be pos-encoding retained at the object patches, not appearance (both are 'object-located'; the random control only rules out *arbitrary*-location signal). Read the headline against the `uv(B-A)` rung (0.420): if headline ≈ uv, the signal is largely object *location* (which a detector provides) — still supports object-centric pooling, but is not a claim that appearance alone encodes the relation.
- Headline rungs contain pooled token VALUES only — no centroid / box / coordinate is fed to the probe (signature-pinned + width-asserted). The `uv(B-A)` rung is the lone coordinate feature, flagged near-tautological.
- Localization is pixel-derived (deployable path); only patch SELECTION uses it. DINOv2 patch tokens carry some positional encoding, so the random-location control is the guard against reading that as success.
- This is the n=200 adequate-data run; the n=40 pilot (`reports/stage5/object_relation_probe`) showed a spurious 0.625 vs 0.425 lift that vanishes here. No representation swap is warranted on this evidence.

