# Stage-5 Step-2 — StackCube object-centric VISUAL relation probe — StackCube-v1

> **SUPERSEDED by the n=200 run (2026-06-02).** This is the **n=40** probe. At
> **n=200** (`reports/stage5/object_relation_probe_n200`): `global_dino` rises to
> **0.685** (the 0.42 baseline was small-n), `B_tok` ≈ global, and the headline
> `B_tok−A_tok` = **0.560 < global** — object-local pooling shows **NO lift**.
> The modest n=40 lift below (0.625 vs 0.425) was small-sample noise. Controls
> stay clean (random ≈ chance). Net: the object-centric-pooling hypothesis is not
> supported once the probe has adequate data.

**Question:** do frozen DINOv2 patch tokens, pooled at the two cube
locations (location selects patches, never used as a feature value),
recover `object_motion` better than global mean-pooling (**0.42**)?

- n=40 (dropped 0 for missing blob), label dist {'translate_+x': 10, 'translate_+y': 10, 'translate_-x': 10, 'translate_-y': 10} (majority 0.250)
- Localization: pixel colour-blob centroids (cubeA=red, cubeB=green); patch pooling radius=1.
- Probe: same protocol as the 0.42 cell (IntentHead F=6, d_slot=32, n_epochs=300) + direct StandardScaler+LR.
- Label = `goal_direction_to_motion(cubeB_init - cubeA_init)`, so any COORDINATE re-feeds the label; headline rungs are appearance-only.

## Feature ladder

| feature | group | dim | majority | shuffled | direct LR ± std | IntentHead-CV ± std | gate |
|---|---|---|---|---|---|---|---|
| `global_dino (768)` | baseline | 768 | 0.250 | 0.225 | 0.350 ± 0.094 | 0.425 ± 0.100 | FAIL |
| `A_tok (cubeA local)` | appearance | 768 | 0.250 | 0.225 | 0.250 ± 0.137 | 0.275 ± 0.146 | FAIL |
| `B_tok (cubeB local)` | appearance | 768 | 0.250 | 0.225 | 0.575 ± 0.127 | 0.625 ± 0.158 | FAIL |
| `[A_tok;B_tok]` | appearance | 1536 | 0.250 | 0.250 | 0.500 ± 0.209 | 0.425 ± 0.257 | FAIL |
| `B_tok-A_tok (HEADLINE)` | appearance | 768 | 0.250 | 0.250 | 0.525 ± 0.242 | 0.625 ± 0.177 | FAIL |
| `A_tok@rand` | control | 768 | 0.250 | 0.400 | 0.325 ± 0.100 | 0.250 ± 0.079 | FAIL |
| `[rand;rand]` | control | 1536 | 0.250 | 0.350 | 0.275 ± 0.094 | 0.250 ± 0.079 | FAIL |
| `B_tok-A_tok @rand` | control | 768 | 0.250 | 0.325 | 0.250 ± 0.112 | 0.275 ± 0.094 | FAIL |
| `uv(B-A) image` | UPPER-BOUND | 2 | 0.250 | 0.225 | 0.475 ± 0.122 | 0.400 ± 0.166 | (near-tautological) |

## Verdict

- **Headline** (`B_tok-A_tok (HEADLINE)`, IntentHead-CV): **0.625** vs baseline **0.42**, majority **0.250**, image-uv upper bound **0.400**.
- ~~USEFUL — frozen DINO tokens carry the relation when pooled object-locally; object-centric pooling helps.~~ **Superseded (see header): this 0.625 vs 0.425 lift was small-sample noise. At n=200 the global baseline rises to 0.685 and object-local pooling shows NO lift (headline 0.560 < 0.685).**
- **Random-location control** relation = 0.275: if this ≈ the object-located headline, the lift is DINO positional-encoding bleed, not object appearance; if the headline ≫ control, object selection carries the signal.

## Caveats

- **The claim is object-local pooling vs GLOBAL pooling** (both frozen DINOv2). If the headline ≫ 0.42, global mean-pooling destroys a relation that object-local pooling preserves → object-centric pooling is the fix. This holds whether the retained signal is cube *appearance* or DINOv2 *positional encoding* at the selected patches.
- **Appearance vs positional encoding:** a lift could be pos-encoding retained at the object patches, not appearance (both are 'object-located'; the random control only rules out *arbitrary*-location signal). Read the headline against the `uv(B-A)` rung (0.400): if headline ≈ uv, the signal is largely object *location* (which a detector provides) — still supports object-centric pooling, but is not a claim that appearance alone encodes the relation.
- Headline rungs contain pooled token VALUES only — no centroid / box / coordinate is fed to the probe (signature-pinned + width-asserted). The `uv(B-A)` rung is the lone coordinate feature, flagged near-tautological.
- Localization is pixel-derived (deployable path); only patch SELECTION uses it. DINOv2 patch tokens carry some positional encoding, so the random-location control is the guard against reading that as success.
- n=40 is small for 768-dim tokens; if the headline is an ambiguous WEAK LIFT, expand to n≈200 before swapping representations.

