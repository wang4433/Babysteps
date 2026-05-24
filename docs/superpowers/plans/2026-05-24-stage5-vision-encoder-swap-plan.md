# Stage 5 P1 — Vision Encoder Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-05-24-stage5-vision-encoder-swap-design.md`

**Goal:** Replace Stage-4's handcrafted 20-dim demo features with frozen DINOv2 features extracted from third-person demo RGB frames, then re-run the M2a G1 / G4 / G5 protocol to confirm vision-grounded slot intents recover all six Stage-0 factors at ≥ 90% and preserve the latent revision loop.

**Architecture:** A new sim-free module `babysteps/stage4/vision_features.py` wraps frozen DINOv2 ViT-B/14, taking the same uint8 RGB arrays `render_frame(env)` already produces and returning a `(768,)` pooled float32 vector. A GPU re-render script captures demo frames per seed from the existing varied-intent collection (the demo is deterministic — same seed, same oracle), and a one-off GPU cache job extracts and saves DINOv2 features. The existing `IntentHead` accepts any `z_dim`, so M2a's nested-CV G1 cert, ReviseHead training, and sim-rollout G4/G5 eval all generalize unchanged — only the input pack changes.

**Tech Stack:** Python 3.11, PyTorch 2.11 + torchvision 0.26 (DINOv2 via `torch.hub`), NumPy, scikit-learn, ManiSkill (re-render only), pytest.

---

## File Structure

**New files:**
- `babysteps/stage4/vision_features.py` — frozen DINOv2 extractor (sim-free except for one optional GPU `torch.hub.load` path; pure-tensor pipeline is CPU-runnable for tests)
- `tests/test_stage5_vision_features.py` — sim-free unit tests using an injected fake encoder
- `scripts/stage5_render_demo_frames.py` — GPU job that re-runs the oracle demo per seed and saves `(T, H, W, 3) uint8` frame stacks
- `scripts/stage5_cache_dinov2.py` — GPU job that runs `vision_features.extract_vision_features` per seed and caches `(768,) float32` features
- `scripts/stage5_p1_g1_cert.py` — nested-CV G1 probe on cached DINOv2 features (mirrors `stage4_m2a_g1_cert.py`)
- `scripts/stage5_p1_train_pack.py` — train IntentHead + ReviseHead on cached DINOv2 features (mirrors `stage4_m2a_train_pack.py`)
- `scripts/stage5_p1_run_eval.py` — sim-rollout G4/G5 eval on the vision-grounded LatentPack (mirrors `stage4_m2a_run_eval.py`)

**New data directories:**
- `datasets/stage5/varied_intent/<task>/frames/seed_NNNN.npz` — saved demo frames
- `datasets/stage5/varied_intent/<task>/features/seed_NNNN_dinov2.npy` — cached DINOv2 features
- `reports/stage5/p1_vision_g1/` — G1 report (markdown + json)
- `models/stage5/p1_vision/<task>/` — saved vision-grounded LatentPack

**Modified files:** none in S1–S3. Optional small edits in S4–S5 only if the existing M2a CLIs need a `--features-dir` flag plumbed through.

**Why parallel `datasets/stage5/` instead of overlaying `datasets/stage4/`:** stage4's `samples.jsonl` is the authoritative episode record (unchanged); frame .npz and feature .npy are *additive caches* keyed by the same seed. Keeping them under `datasets/stage5/` makes them regeneratable and disposable without risk to the stage-4 ground truth.

---

## Section S1 — `vision_features.py` module (sim-free, TDD-first)

The headline deliverable. ~80 lines of code + ~6 unit tests. Sim-free unit tests use a fake encoder so the suite stays GPU-free; a single integration smoke test on real DINOv2 is gated to GPU.

### Task S1.1: Add the failing shape test for `_preprocess_frames`

**Files:**
- Create: `tests/test_stage5_vision_features.py`

- [ ] **Step 1: Write the failing test**

```python
"""Stage-5 P1 — vision_features module tests.

Sim-free: CPU-only torch. The real DINOv2 model is never loaded in this
suite; instead an injected fake encoder verifies the pre/post pipeline.
A separate GPU smoke test (scripts/stage5_cache_dinov2.py --check) loads
the real encoder.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_preprocess_frames_shape_and_dtype():
    """List[(H,W,3) uint8] -> (T, 3, 224, 224) float32 ImageNet-normalized."""
    from babysteps.stage4.vision_features import _preprocess_frames

    frames = [
        (255 * np.random.rand(512, 512, 3)).astype(np.uint8)
        for _ in range(4)
    ]
    x = _preprocess_frames(frames, resolution=224)
    assert x.shape == (4, 3, 224, 224)
    assert x.dtype == torch.float32
    # ImageNet-normalized: mean roughly in [-2.2, 2.7] for random pixels.
    assert -3.0 < float(x.mean()) < 3.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /scratch/gilbreth/wang4433/babysteps
python -m pytest tests/test_stage5_vision_features.py::test_preprocess_frames_shape_and_dtype -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'babysteps.stage4.vision_features'`.

### Task S1.2: Implement `_preprocess_frames` to make Task S1.1 pass

**Files:**
- Create: `babysteps/stage4/vision_features.py`

- [ ] **Step 1: Write the minimal module**

```python
"""Stage-5 P1 — frozen vision-encoder feature extraction.

Wraps a frozen pretrained vision encoder (default: DINOv2 ViT-B/14)
applied to the third-person demo RGB frames produced by
`babysteps.render.common.render_frame(env)`. Returns a single
(d_encoder,) float32 vector per demo, suitable as drop-in `Z` for
the existing Stage-4 IntentHead.

Stage-4 firewall (carried over): this module consumes only RGB frame
arrays — DemoEvidence-shaped input — and never reads
execution.initial_intent, failure_packet, revision, retry, or any
privileged SceneState field.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ImageNet normalization constants used by DINOv2.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _preprocess_frames(
    frames: list[np.ndarray],
    *,
    resolution: int = 224,
) -> torch.Tensor:
    """List[(H, W, 3) uint8] -> (T, 3, R, R) float32 ImageNet-normalized."""
    # Stack to (T, H, W, 3) uint8, permute to (T, 3, H, W), float in [0, 1].
    arr = np.stack(frames, axis=0)
    if arr.dtype != np.uint8:
        raise ValueError(f"frames must be uint8, got {arr.dtype}")
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"frames must have shape (T, H, W, 3), got {arr.shape}")
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)
    # Resize to (R, R) via bilinear; DINOv2 ViT-B/14 needs the spatial dims
    # divisible by the patch size 14 — 224 = 16*14 is the standard.
    t = F.interpolate(t, size=(resolution, resolution),
                      mode="bilinear", align_corners=False)
    # ImageNet normalize per channel.
    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (t - mean) / std
```

- [ ] **Step 2: Run the test to verify it passes**

```bash
python -m pytest tests/test_stage5_vision_features.py::test_preprocess_frames_shape_and_dtype -v
```

Expected: PASS.

### Task S1.3: Add the `_pool_cls` test + implementation

**Files:**
- Modify: `tests/test_stage5_vision_features.py` — append test
- Modify: `babysteps/stage4/vision_features.py` — append function

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stage5_vision_features.py`:

```python
def test_pool_cls_mean_collapses_time_dim():
    """(T, d) cls tokens -> (d,) mean. Numerical identity on a hand-built case."""
    from babysteps.stage4.vision_features import _pool_cls

    cls = torch.tensor([
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
        [2.0, 2.0, 2.0],
    ])  # (3, 3)
    z = _pool_cls(cls, pool="cls_mean")
    assert z.shape == (3,)
    torch.testing.assert_close(z, torch.tensor([2.0, 2.0, 2.0]))


def test_pool_cls_unknown_strategy_raises():
    from babysteps.stage4.vision_features import _pool_cls

    with pytest.raises(ValueError, match="unknown pool"):
        _pool_cls(torch.zeros(2, 4), pool="not-a-real-strategy")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_stage5_vision_features.py::test_pool_cls_mean_collapses_time_dim -v
```

Expected: FAIL with `ImportError: cannot import name '_pool_cls'`.

- [ ] **Step 3: Implement `_pool_cls`**

Append to `babysteps/stage4/vision_features.py`:

```python
def _pool_cls(cls_tokens: torch.Tensor, *, pool: str = "cls_mean") -> torch.Tensor:
    """(T, d) -> (d,). Time-pooling strategies.

    Strategies (per the design spec § 3.2):
      - cls_mean: mean over T (default; simplest baseline).
      Future ablations (cls_first_last, spatial_mean) can be added here
      without changing the public extract_vision_features signature.
    """
    if pool == "cls_mean":
        return cls_tokens.mean(dim=0)
    raise ValueError(f"unknown pool strategy: {pool!r}")
```

- [ ] **Step 4: Run both pool tests**

```bash
python -m pytest tests/test_stage5_vision_features.py -k "pool" -v
```

Expected: 2 PASS.

### Task S1.4: Add the end-to-end `extract_vision_features` test with injected fake encoder

**Files:**
- Modify: `tests/test_stage5_vision_features.py` — append test
- Modify: `babysteps/stage4/vision_features.py` — append function

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stage5_vision_features.py`:

```python
class _FakeEncoder(torch.nn.Module):
    """Mock DINOv2 — returns a fixed (T, d) CLS embedding so the test
    verifies the pre→encode→pool→numpy pipeline without loading real weights."""

    def __init__(self, d: int = 768):
        super().__init__()
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, 3, R, R). Return a deterministic per-frame embedding
        # whose mean across T is exactly arange(d) / d * mean(x) — checkable.
        T = x.shape[0]
        base = torch.arange(self.d, dtype=torch.float32) / self.d
        # Modulate per-frame so the time-mean has a stable identity.
        per_t = x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)  # (T, 1)
        return base.unsqueeze(0) * per_t  # (T, d)


def test_extract_vision_features_end_to_end_with_fake_encoder():
    """Full path: uint8 frames -> preprocess -> encode -> pool -> numpy."""
    from babysteps.stage4.vision_features import extract_vision_features

    frames = [
        (128 * np.ones((512, 512, 3), dtype=np.uint8))
        for _ in range(5)
    ]
    z = extract_vision_features(
        frames,
        device="cpu",
        _encoder=_FakeEncoder(d=768),  # injection for test
    )
    assert isinstance(z, np.ndarray)
    assert z.shape == (768,)
    assert z.dtype == np.float32
    # Identical input frames -> identical per-frame embeddings -> mean = embedding.
    # Embedding magnitude > 0 (non-trivial signal).
    assert float(np.abs(z).sum()) > 0.0


def test_extract_vision_features_rejects_empty_frames():
    from babysteps.stage4.vision_features import extract_vision_features

    with pytest.raises(ValueError, match="at least one frame"):
        extract_vision_features([], device="cpu", _encoder=_FakeEncoder())
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_stage5_vision_features.py::test_extract_vision_features_end_to_end_with_fake_encoder -v
```

Expected: FAIL with `ImportError: cannot import name 'extract_vision_features'`.

- [ ] **Step 3: Implement `extract_vision_features`**

Append to `babysteps/stage4/vision_features.py`:

```python
# Module-level cache so successive calls in the cache-features job don't
# reload DINOv2 weights for every seed.
_MODEL_CACHE: dict[tuple[str, str], torch.nn.Module] = {}


def _load_dinov2(encoder: str, device: str) -> torch.nn.Module:
    """Load and freeze a DINOv2 model via torch.hub. Cached per (encoder, device).

    Network/disk hit happens once per process; the model is moved to
    `device`, set to eval mode, and all parameters are frozen.
    """
    key = (encoder, device)
    if key not in _MODEL_CACHE:
        model = torch.hub.load("facebookresearch/dinov2", encoder)
        model.eval()
        model.to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def extract_vision_features(
    demo_frames: list[np.ndarray],
    *,
    encoder: str = "dinov2_vitb14",
    pool: str = "cls_mean",
    device: str = "cuda",
    resolution: int = 224,
    _encoder: Optional[torch.nn.Module] = None,  # test/inject hook
) -> np.ndarray:
    """Frozen-encoder feature extraction from demo RGB frames.

    Args:
      demo_frames: list of (H, W, 3) uint8 RGB arrays — exactly what
        babysteps.render.common.render_frame(env) produces.
      encoder: torch.hub model id; default DINOv2 ViT-B/14.
      pool: time-pooling strategy (see `_pool_cls`).
      device: torch device for the encoder.
      resolution: square resize before encoding (DINOv2 wants 224).
      _encoder: optional injected encoder for unit tests (bypasses torch.hub).

    Returns:
      (d_encoder,) float32 numpy vector. d_encoder = 768 for ViT-B/14.

    The Stage-4 firewall applies: this function reads only the frame
    arrays — no labels, no privileged scene state.
    """
    if len(demo_frames) == 0:
        raise ValueError("extract_vision_features needs at least one frame")

    model = _encoder if _encoder is not None else _load_dinov2(encoder, device)
    x = _preprocess_frames(demo_frames, resolution=resolution).to(device)
    with torch.no_grad():
        cls = model(x)  # (T, d) — DINOv2's default forward returns CLS
    z = _pool_cls(cls, pool=pool)
    return z.detach().cpu().numpy().astype(np.float32)
```

- [ ] **Step 4: Run all the S1 tests**

```bash
python -m pytest tests/test_stage5_vision_features.py -v
```

Expected: 5 PASS (preprocess shape, pool mean, pool error, extract e2e, extract empty error).

### Task S1.5: Sanity-check the sim-free suite still passes

- [ ] **Step 1: Run the full sim-free suite**

```bash
python -m pytest tests/ -q
```

Expected: PASS (302 + 5 new = 307 tests). Wall time should still be < 5s.

### Task S1.6: Commit S1

- [ ] **Step 1: Commit**

```bash
cd /scratch/gilbreth/wang4433/babysteps
git add babysteps/stage4/vision_features.py tests/test_stage5_vision_features.py docs/superpowers/specs/2026-05-24-stage5-vision-encoder-swap-design.md docs/superpowers/plans/2026-05-24-stage5-vision-encoder-swap-plan.md
git commit -m "feat(stage5 p1): vision_features extractor + sim-free tests

S1 of the Stage-5 P1 vision-encoder swap. New sim-free module
babysteps/stage4/vision_features.py wraps frozen DINOv2 ViT-B/14
applied to uint8 RGB demo frames (the same arrays render_frame(env)
already produces) and returns a (768,) float32 vector. The Stage-4
firewall carries over: encoder-side inputs only, no label / privileged
fields. 5 unit tests use an injected fake encoder so the suite stays
GPU-free.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Section S2 — Re-render demo frames per seed (GPU)

Re-run the oracle demo on each seed in the existing varied-intent JSONL and save the third-person camera frames as `.npz`. The demo is deterministic (same seed, same scripted oracle), so the saved frames are bit-for-bit reproducible.

### Task S2.1: Write `scripts/stage5_render_demo_frames.py`

**Files:**
- Create: `scripts/stage5_render_demo_frames.py`

- [ ] **Step 1: Write the script**

```python
# scripts/stage5_render_demo_frames.py
"""Stage-5 P1 — re-render demo frames per seed (GPU).

For each seed in an existing varied-intent samples.jsonl, re-runs the
oracle demo on the real ManiSkill env, captures one (H, W, 3) uint8
frame per control step via babysteps.render.common.render_frame, and
saves the stack as datasets/stage5/varied_intent/<task>/frames/seed_NNNN.npz.

The demo is deterministic (same seed, same scripted oracle adapter
demo function); no state is shared with the executing-attempt phase.

Example::

    python scripts/stage5_render_demo_frames.py \\
        --jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl \\
        --out-dir datasets/stage5/varied_intent/PushCube-v1/frames/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.envs.task_registry import get_task_entry  # noqa: E402
from babysteps.render.common import render_frame  # noqa: E402
from babysteps.schemas import EpisodeRecord  # noqa: E402


def _seed_from_record(rec: dict) -> int:
    """Extract the int seed from an episode_id like 'pushcube_varied_seed_0012'."""
    eid = rec["episode_id"]
    # The trailing chunk after the last underscore is the zero-padded seed.
    return int(eid.split("_")[-1])


def _capture_demo_frames(adapter, seed: int) -> np.ndarray:
    """Re-run the oracle demo on `seed` and return a (T, H, W, 3) uint8 stack.

    Uses the same demo function the task adapter uses inside
    `babysteps.episode.run_episode`, but instead of consuming the
    DemoEvidence we capture the env-side rgb buffer per step.
    """
    runner = adapter.env_runner()
    # Reset the env to the same seed the original episode used. The
    # demo is then driven by the adapter's scripted demo program.
    scene = runner.reset(seed)  # noqa: F841  (scene used implicitly via state)
    frames: list[np.ndarray] = []
    # The adapter exposes `iter_demo_steps(scene)` per the TaskAdapter
    # interface; each yielded step is a fresh env.step() applied via
    # the scripted oracle controller. After each step, capture a frame.
    for _ in adapter.iter_demo_steps(runner):
        frames.append(render_frame(runner.env))
    if not frames:
        raise RuntimeError(f"demo produced 0 frames for seed {seed}")
    return np.stack(frames, axis=0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, required=True,
                   help="Source varied-intent samples.jsonl.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for seed_NNNN.npz files.")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of seeds (smoke test).")
    args = p.parse_args(argv)

    with args.jsonl.open() as f:
        records = [EpisodeRecord.from_jsonl_line(line).to_dict()
                   for line in f if line.strip()]
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        print("no records to render", file=sys.stderr)
        return 1

    task = records[0]["task"]
    entry = get_task_entry(task)
    adapter = entry.adapter_cls()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for rec in records:
            seed = _seed_from_record(rec)
            frames = _capture_demo_frames(adapter, seed)
            out = args.out_dir / f"seed_{seed:04d}.npz"
            np.savez_compressed(out, frames=frames)
            print(f"wrote {out} (T={frames.shape[0]}, "
                  f"H={frames.shape[1]}, W={frames.shape[2]})")
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

> **Note on `adapter.iter_demo_steps`:** if the existing `TaskAdapter` interface does not expose a `iter_demo_steps(runner)` method, the script must instead reuse whichever demo-execution function `babysteps.episode.run_episode` calls. The exact integration is one-line: find the call in `babysteps/episode.py` that drives the demo (typically inside `generate_proxy_demo()` or equivalent), and wrap it with frame capture. **First task in S2 is to verify which entry point exists** (see Task S2.2).

### Task S2.2: Verify the demo-loop entry point on PushCube

- [ ] **Step 1: Read `babysteps/episode.py` and `babysteps/envs/pushcube_runner.py` to identify which call drives the demo step loop**

```bash
grep -n "demo" /scratch/gilbreth/wang4433/babysteps/babysteps/episode.py
grep -n "def run\|def reset\|def step\|render" /scratch/gilbreth/wang4433/babysteps/babysteps/envs/pushcube_runner.py | head -40
```

Expected: identify the function name (likely `runner.demo_loop()` or `adapter.iter_demo_steps()` or a private helper in `episode.py`). If the existing interface does not yield per-step control, add a small `iter_demo_steps` generator to the runner that mirrors the existing demo logic but yields each step — keeping `run_episode` unchanged.

- [ ] **Step 2: If a new generator is needed, write it on `PushCubeEnvRunner`**

Patch shape (paste into `babysteps/envs/pushcube_runner.py` adjacent to the existing demo function):

```python
    def iter_demo_steps(self):
        """Yield once per control step of the oracle demo.

        Identical control sequence to whatever `run_episode` uses to
        produce the DemoEvidence — just exposed as a generator so a
        caller can render a frame per step.
        """
        # NOTE: copy the body of the existing demo controller here; emit
        # `yield` after each `self.env.step(...)` call. No other state
        # change vs. the existing path — the demo is deterministic.
        ...
```

Apply the same generator to `StackCubeEnvRunner` if needed.

> The minimum behavior contract: identical to the demo currently embedded in `run_episode`; only difference is the per-step `yield`.

- [ ] **Step 3: Add a sim-free smoke test that the generator exists and is iterable**

Append to `tests/test_stage4_smoke.py` (or new `tests/test_stage5_render_demo_frames_smoke.py`):

```python
def test_iter_demo_steps_exists_on_pushcube_runner():
    """Generator interface present — sim-free: we only check the attribute."""
    from babysteps.envs.pushcube_runner import PushCubeEnvRunner
    assert hasattr(PushCubeEnvRunner, "iter_demo_steps")
```

Run:

```bash
python -m pytest tests/test_stage4_smoke.py -k "iter_demo_steps" -v
```

Expected: PASS.

### Task S2.3: Run the GPU job for PushCube (smoke first, then full)

- [ ] **Step 1: Smoke test (2 seeds)**

```bash
cd /scratch/gilbreth/wang4433/babysteps
python scripts/stage5_render_demo_frames.py \
    --jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl \
    --out-dir datasets/stage5/varied_intent/PushCube-v1/frames/ \
    --limit 2
```

Expected: 2 `.npz` files written, each `T ≈ 30–50, H = W = 512`.

- [ ] **Step 2: Spot-check one file**

```bash
python -c "import numpy as np; \
  d = np.load('datasets/stage5/varied_intent/PushCube-v1/frames/seed_0000.npz'); \
  f = d['frames']; \
  print(f.shape, f.dtype, f.min(), f.max())"
```

Expected: `(T, 512, 512, 3) uint8 0 255`.

- [ ] **Step 3: Full run on PushCube (20 seeds)**

Drop `--limit`:

```bash
python scripts/stage5_render_demo_frames.py \
    --jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl \
    --out-dir datasets/stage5/varied_intent/PushCube-v1/frames/
```

Wall time: ~3–5 min on an A100 (ManiSkill render dominates).

- [ ] **Step 4: Repeat for StackCube (40 seeds)**

```bash
python scripts/stage5_render_demo_frames.py \
    --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl \
    --out-dir datasets/stage5/varied_intent/StackCube-v1/frames/
```

Wall time: ~6–10 min.

### Task S2.4: Commit S2

- [ ] **Step 1: Commit**

```bash
git add scripts/stage5_render_demo_frames.py tests/test_stage4_smoke.py
# (Only commit small runner edits if they were needed in Task S2.2 Step 2)
[ -n "$(git diff --cached --name-only babysteps/envs/)" ] && echo "(also committing runner generator)"
git commit -m "feat(stage5 p1): re-render demo frames per seed (GPU)

S2 of the Stage-5 P1 vision-encoder swap. New
scripts/stage5_render_demo_frames.py re-runs the oracle demo for each
seed in an existing varied-intent samples.jsonl, captures one
render_frame(env) per control step, and saves the (T, H, W, 3) uint8
stack as datasets/stage5/varied_intent/<task>/frames/seed_NNNN.npz.

The demo is deterministic (same seed + same scripted oracle), so this
adds no new ground truth — only a frame cache keyed to the existing
samples.jsonl. Sim-free smoke test pins the generator interface.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

> **Note:** the saved frame data is large (~40 MB per seed × 60 seeds ≈ 2 GB). Do NOT commit `datasets/stage5/`. Add `datasets/stage5/` to `.gitignore` if not already covered.

- [ ] **Step 2: Verify .gitignore covers `datasets/stage5/`**

```bash
grep -E "^datasets/" .gitignore
```

If not covered, append:

```bash
echo "datasets/stage5/" >> .gitignore
git add .gitignore
git commit --amend --no-edit
```

---

## Section S3 — Extract and cache DINOv2 features (GPU)

One-off GPU job that runs `extract_vision_features` from S1 on each seed's saved frames and caches the `(768,)` float32 result. ~2 min per task on an A100.

### Task S3.1: Write `scripts/stage5_cache_dinov2.py`

**Files:**
- Create: `scripts/stage5_cache_dinov2.py`

- [ ] **Step 1: Write the script**

```python
# scripts/stage5_cache_dinov2.py
"""Stage-5 P1 — extract and cache DINOv2 features per seed (GPU, one-off).

Reads frame stacks written by scripts/stage5_render_demo_frames.py,
runs extract_vision_features (default DINOv2 ViT-B/14, cls_mean pool),
and writes (768,) float32 features alongside as seed_NNNN_dinov2.npy.

The model is loaded once per process and cached at module level
(babysteps.stage4.vision_features._MODEL_CACHE), so the per-seed cost
is just the forward pass on (T, 3, 224, 224).

Example::

    python scripts/stage5_cache_dinov2.py \\
        --frames-dir datasets/stage5/varied_intent/PushCube-v1/frames/ \\
        --out-dir datasets/stage5/varied_intent/PushCube-v1/features/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.stage4.vision_features import extract_vision_features  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames-dir", type=Path, required=True,
                   help="Directory of seed_NNNN.npz frame stacks.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for seed_NNNN_dinov2.npy.")
    p.add_argument("--encoder", type=str, default="dinov2_vitb14")
    p.add_argument("--pool", type=str, default="cls_mean")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--check", action="store_true",
                   help="Load DINOv2 once and exit (smoke).")
    args = p.parse_args(argv)

    if args.check:
        from babysteps.stage4.vision_features import _load_dinov2
        m = _load_dinov2(args.encoder, args.device)
        print(f"loaded {args.encoder} on {args.device}: "
              f"{sum(p.numel() for p in m.parameters()):,} params")
        return 0

    frame_files = sorted(args.frames_dir.glob("seed_*.npz"))
    if not frame_files:
        print(f"no seed_*.npz under {args.frames_dir}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fp in frame_files:
        frames = list(np.load(fp)["frames"])  # list[(H, W, 3) uint8]
        z = extract_vision_features(
            frames,
            encoder=args.encoder, pool=args.pool, device=args.device,
        )
        out = args.out_dir / f"{fp.stem}_dinov2.npy"
        np.save(out, z)
        print(f"wrote {out} (shape={z.shape}, dtype={z.dtype})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Task S3.2: Run the smoke check (load DINOv2 once)

- [ ] **Step 1: Verify DINOv2 loads in the active conda env**

```bash
python scripts/stage5_cache_dinov2.py --check
```

Expected: `loaded dinov2_vitb14 on cuda: ~86,000,000 params`. If this fails with a torch.hub network error, set `TORCH_HOME` and pre-download:

```bash
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')"
```

### Task S3.3: Cache features for PushCube and StackCube

- [ ] **Step 1: Cache PushCube features**

```bash
python scripts/stage5_cache_dinov2.py \
    --frames-dir datasets/stage5/varied_intent/PushCube-v1/frames/ \
    --out-dir datasets/stage5/varied_intent/PushCube-v1/features/
```

Wall time: ~2 min for 20 seeds.

- [ ] **Step 2: Cache StackCube features**

```bash
python scripts/stage5_cache_dinov2.py \
    --frames-dir datasets/stage5/varied_intent/StackCube-v1/frames/ \
    --out-dir datasets/stage5/varied_intent/StackCube-v1/features/
```

Wall time: ~4 min for 40 seeds.

- [ ] **Step 3: Spot-check a feature file**

```bash
python -c "import numpy as np; \
  z = np.load('datasets/stage5/varied_intent/PushCube-v1/features/seed_0000_dinov2.npy'); \
  print(z.shape, z.dtype, z.mean(), z.std())"
```

Expected: `(768,) float32 <some-finite-mean> <some-positive-std>`.

### Task S3.4: Commit S3

- [ ] **Step 1: Commit**

```bash
git add scripts/stage5_cache_dinov2.py
git commit -m "feat(stage5 p1): cache DINOv2 features per seed (GPU one-off)

S3 of the Stage-5 P1 vision-encoder swap. New
scripts/stage5_cache_dinov2.py reads frame stacks from S2 and writes
(768,) float32 DINOv2 ViT-B/14 features as
datasets/stage5/varied_intent/<task>/features/seed_NNNN_dinov2.npy.
DINOv2 is loaded once per process via module-level cache so the
per-seed cost is just the forward pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Section S4 — G1 probe on DINOv2 features (go/no-go gate)

Run the same nested-CV G1 protocol as M2a, but consuming cached DINOv2 features instead of the 20-dim handcrafted vector. **This is the gate:** all non-trivially-constant (task, factor) cells must reach ≥ 90% held-out accuracy.

### Task S4.1: Write `scripts/stage5_p1_g1_cert.py`

**Files:**
- Create: `scripts/stage5_p1_g1_cert.py`

- [ ] **Step 1: Write the script (mirror `scripts/stage4_m2a_g1_cert.py`)**

```python
# scripts/stage5_p1_g1_cert.py
"""Stage-5 P1 Gate G1 — IntentHead probe recoverability on DINOv2 features.

Mirrors scripts/stage4_m2a_g1_cert.py but consumes cached DINOv2
features (Z = (768,) float32 per seed) in place of the 20-dim
handcrafted vector. Same nested-CV protocol, same three-way report
schema, same gate threshold.

Example::

    python scripts/stage5_p1_g1_cert.py \\
        --jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl \\
        --features-dir datasets/stage5/varied_intent/PushCube-v1/features/ \\
        --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl \\
        --features-dir datasets/stage5/varied_intent/StackCube-v1/features/ \\
        --out-dir reports/stage5/p1_vision_g1/ --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.schemas import INTENT_FIELDS, EpisodeRecord  # noqa: E402
from babysteps.stage4.intent_head import nested_cv_probe_one_factor  # noqa: E402
from babysteps.stage4.report import (  # noqa: E402
    GATE_THRESHOLD,
    build_report,
    markdown_table,
)


def _seed_from_record(rec: dict) -> int:
    return int(rec["episode_id"].split("_")[-1])


def _load_one_task(jsonl: Path, features_dir: Path) -> tuple[list[dict], np.ndarray]:
    """Load records and stack their cached DINOv2 features in jsonl order."""
    records: list[dict] = []
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(EpisodeRecord.from_jsonl_line(line).to_dict())
    Z_rows: list[np.ndarray] = []
    for rec in records:
        seed = _seed_from_record(rec)
        feat = features_dir / f"seed_{seed:04d}_dinov2.npy"
        Z_rows.append(np.load(feat))
    Z = np.stack(Z_rows).astype(np.float32)
    return records, Z


def _probe_rows(
    records: list[dict], Z: np.ndarray, *,
    n_factors: int, d_slot: int, hidden: int, n_epochs: int, lr: float, seed: int,
) -> list[dict]:
    """One trained-encoder probe per factor on the supplied Z."""
    rows: list[dict] = []
    task = records[0]["task"]
    for factor_idx, factor in enumerate(INTENT_FIELDS):
        labels = [r["execution"]["initial_intent"][factor] for r in records]
        y = LabelEncoder().fit_transform(labels)
        out = nested_cv_probe_one_factor(
            Z, y,
            factor_idx=factor_idx, n_factors=n_factors,
            d_slot=d_slot, n_epochs=n_epochs, lr=lr, seed=seed,
        )
        out["task"] = task
        out["factor"] = factor
        rows.append(out)
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, action="append", required=True,
                   help="One varied-intent samples.jsonl; repeat per task.")
    p.add_argument("--features-dir", type=Path, action="append", required=True,
                   help="Matching DINOv2 features directory; repeat per task.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-slot", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    args = p.parse_args(argv)

    if len(args.jsonl) != len(args.features_dir):
        print("--jsonl and --features-dir must be repeated in matching pairs",
              file=sys.stderr)
        return 1

    all_rows: list[dict] = []
    for jl, fd in zip(args.jsonl, args.features_dir):
        records, Z = _load_one_task(jl, fd)
        rows = _probe_rows(
            records, Z,
            n_factors=6, d_slot=args.d_slot, hidden=args.hidden,
            n_epochs=args.n_epochs, lr=args.lr, seed=args.seed,
        )
        all_rows.extend(rows)

    report = build_report(all_rows, gate_threshold=GATE_THRESHOLD)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    md = "\n".join([
        "# Stage-5 P1 — Vision-grounded G1 (DINOv2 ViT-B/14)",
        "",
        f"Input Z: 768-dim DINOv2 ViT-B/14 CLS, mean-pooled over demo frames.",
        f"IntentHead: F=6, d_slot={args.d_slot}, hidden={args.hidden}, "
        f"n_epochs={args.n_epochs}, lr={args.lr}.",
        f"Outer CV: per-fold IntentHead training; frozen LogisticRegression "
        f"on G_train, evaluated on G_test.",
        "",
        markdown_table(report),
        "",
        f"**Gate:** all non-trivial cells ≥ {GATE_THRESHOLD:.0%} → "
        f"**{'PASS' if report['gate']['pass'] else 'FAIL'}**",
    ])
    (args.out_dir / "report.md").write_text(md)
    print(md)
    return 0 if report["gate"]["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
```

> **Note on `d_slot=32`, `hidden=256`:** the spec § 3.4 calls for these scaled hyperparameters when `z_dim` jumps from 20 to 768. The new values are CLI defaults so future ablations can revert.

### Task S4.2: Run the G1 cert

- [ ] **Step 1: Run on both tasks**

```bash
cd /scratch/gilbreth/wang4433/babysteps
python scripts/stage5_p1_g1_cert.py \
    --jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl \
    --features-dir datasets/stage5/varied_intent/PushCube-v1/features/ \
    --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl \
    --features-dir datasets/stage5/varied_intent/StackCube-v1/features/ \
    --out-dir reports/stage5/p1_vision_g1/ --seed 0
```

Wall time: ~30 s (CPU-only).

- [ ] **Step 2: Inspect the report**

```bash
cat reports/stage5/p1_vision_g1/report.md
```

Expected: a markdown table with one row per (task, factor) and a final gate line. **Pass criterion:** all rows with `n_unique_labels > 1` reach `probe_acc_mean ≥ 0.90`.

### Task S4.3: Decision branch — pass or fail

- [ ] **Step 1: If PASS → continue to S5.**

- [ ] **Step 2: If FAIL → stop and try the spec § 3.2 / § 6 ablations:**
  - Switch pool from `cls_mean` to `cls_first_last` (concat first and last frame CLS → 1536-dim). Add a new pool strategy in `_pool_cls` + re-cache + re-cert.
  - If still fails on spatial factors (`contact_region`, `approach_direction`), switch to `spatial_mean` (mean-pool patch tokens).
  - If DINOv2 alone fails on multiple factors, swap the encoder to R3M (2048-dim Ego4D-pretrained ResNet-50). Same module interface, different `torch.hub.load`.

  Do not proceed to S5 with a failing G1 — the paper claim depends on this gate.

### Task S4.4: Commit S4

- [ ] **Step 1: Commit script + report**

```bash
git add scripts/stage5_p1_g1_cert.py reports/stage5/p1_vision_g1/
git commit -m "feat(stage5 p1): G1 probe on DINOv2 features + report

S4 of the Stage-5 P1 vision-encoder swap. New
scripts/stage5_p1_g1_cert.py runs the same nested-CV probe as M2a's
G1 cert but consumes cached DINOv2 features (768-dim CLS, mean-pooled
over demo frames) and scales IntentHead to d_slot=32, hidden=256.

Result: <PASTE THE GATE HEADLINE FROM report.md>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

> Replace `<PASTE …>` with the actual gate headline from `reports/stage5/p1_vision_g1/report.md` (e.g. `all 9 non-trivial cells ≥ 90% — PASS`).

---

## Section S5 — Retrain ReviseHead + sim rollout G4/G5

End-to-end confirmation that the latent revision loop still works on vision-grounded slots. Reuses M2a's machinery; only the input pack changes.

### Task S5.1: Write `scripts/stage5_p1_train_pack.py`

**Files:**
- Create: `scripts/stage5_p1_train_pack.py`

- [ ] **Step 1: Write the script (mirror `scripts/stage4_m2a_train_pack.py`)**

This is a small variant of `stage4_m2a_train_pack.py`. The only differences:
1. Z is loaded from cached DINOv2 files instead of `extract_episode_features`.
2. `IntentHead` is constructed with `z_dim=768, d_slot=32, hidden=256`.
3. `ReviseHead` uses `d_slot=32, hidden=256` to match.

```python
# scripts/stage5_p1_train_pack.py
"""Stage-5 P1 — train + save a vision-grounded LatentPack for one task.

Same protocol as scripts/stage4_m2a_train_pack.py, but:
  * Z is loaded from cached DINOv2 features (S3 output) instead of
    handcrafted babysteps.stage4.features.
  * IntentHead is sized to z_dim=768, d_slot=32, hidden=256.
  * ReviseHead matches the new d_slot.

Sim-free CPU-only, ~30s wall per task.

Example::

    python scripts/stage5_p1_train_pack.py \\
        --jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl \\
        --features-dir datasets/stage5/varied_intent/PushCube-v1/features/ \\
        --out-dir models/stage5/p1_vision/PushCube-v1/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from babysteps.schemas import INTENT_FIELDS, EpisodeRecord  # noqa: E402
from babysteps.stage4.intent_head import (  # noqa: E402
    IntentHead, train_intent_head_joint,
)
from babysteps.stage4.latent_policy import (  # noqa: E402
    LatentPack, save_latent_pack,
)
from babysteps.stage4.revise_head import (  # noqa: E402
    ReviseHead, train_revise_head_l2, vectorize_failure_packet,
)
from babysteps.stage4.slot_decode import build_factor_centroids  # noqa: E402


def _seed_from_record(rec: dict) -> int:
    return int(rec["episode_id"].split("_")[-1])


def _load_records(path: Path) -> list[dict]:
    with path.open() as f:
        return [EpisodeRecord.from_jsonl_line(l).to_dict()
                for l in f if l.strip()]


def _load_features(records: list[dict], features_dir: Path) -> np.ndarray:
    Zs = []
    for rec in records:
        seed = _seed_from_record(rec)
        Zs.append(np.load(features_dir / f"seed_{seed:04d}_dinov2.npy"))
    return np.stack(Zs).astype(np.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-slot", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-epochs-intent", type=int, default=300)
    p.add_argument("--n-epochs-revise", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-2)
    args = p.parse_args(argv)

    records = _load_records(args.jsonl)
    Z = _load_features(records, args.features_dir)
    print(f"loaded {len(records)} records / Z={Z.shape}")

    encoders: dict[int, LabelEncoder] = {}
    labels_per_factor: dict[int, tuple[np.ndarray, int]] = {}
    for fi, factor in enumerate(INTENT_FIELDS):
        present_vals = set(r["execution"]["initial_intent"][factor] for r in records)
        for r in records:
            rv = r.get("revision")
            if rv and rv["factor"] == factor:
                present_vals.add(rv["new_value"])
                present_vals.add(rv["old_value"])
        present_vals = sorted(present_vals)
        if len(present_vals) < 2:
            continue
        enc = LabelEncoder().fit(present_vals)
        y = enc.transform(
            [r["execution"]["initial_intent"][factor] for r in records]
        ).astype(np.int64)
        encoders[fi] = enc
        labels_per_factor[fi] = (y, len(enc.classes_))

    head = IntentHead(z_dim=Z.shape[1], n_factors=len(INTENT_FIELDS),
                      d_slot=args.d_slot, hidden=args.hidden, seed=args.seed)
    train_intent_head_joint(
        head, Z, labels_per_factor,
        n_epochs=args.n_epochs_intent, lr=args.lr, seed=args.seed,
    )
    head.eval()
    with torch.no_grad():
        G = head(torch.from_numpy(Z)).numpy()
    centroids = build_factor_centroids(
        G, {fi: y for fi, (y, _) in labels_per_factor.items()},
    )

    revisions = []
    for i, r in enumerate(records):
        rv = r.get("revision")
        if not rv or rv["factor"] not in INTENT_FIELDS:
            continue
        fi = INTENT_FIELDS.index(rv["factor"])
        if fi not in encoders:
            continue
        try:
            new_class = int(encoders[fi].transform([rv["new_value"]])[0])
        except ValueError:
            continue
        if fi not in centroids or new_class not in centroids[fi]:
            continue
        revisions.append({
            "i": i, "fi": fi, "new_class": new_class,
            "fp_vec": vectorize_failure_packet(r),
        })

    revise = ReviseHead(d_slot=args.d_slot, hidden=args.hidden, seed=args.seed)
    if revisions:
        g_pre = np.stack([G[rv["i"], rv["fi"]] for rv in revisions]).astype(np.float32)
        fp = np.stack([rv["fp_vec"] for rv in revisions]).astype(np.float32)
        g_tgt = np.stack([centroids[rv["fi"]][rv["new_class"]]
                          for rv in revisions]).astype(np.float32)
        train_revise_head_l2(
            revise, g_pre, fp, g_tgt,
            n_epochs=args.n_epochs_revise, lr=args.lr, seed=args.seed,
        )
        print(f"trained ReviseHead on {len(revisions)} pairs")
    else:
        print("WARNING: 0 certable revisions; ReviseHead at random init")

    label_tokens = {fi: tuple(enc.classes_) for fi, enc in encoders.items()
                    if fi in centroids}
    pack = LatentPack(
        intent_head=head, revise_head=revise,
        centroids=centroids, label_tokens=label_tokens,
        attribution_head=None,  # M2.5 head is encoder-input dependent; skip in P1
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_latent_pack(pack, args.out_dir)
    print(f"saved LatentPack to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

> **Note on `attribution_head=None`:** M2.5's AttributionHead was trained on the 20-dim handcrafted Z. In P1 v1 we drop it (rule-based attribution from `failure.py` is the fallback). Retraining AttributionHead on DINOv2 Z is a follow-up — not on the P1 critical path.

### Task S5.2: Train packs for PushCube and StackCube

- [ ] **Step 1: Train PushCube pack**

```bash
python scripts/stage5_p1_train_pack.py \
    --jsonl datasets/stage4/varied_intent/PushCube-v1/samples.jsonl \
    --features-dir datasets/stage5/varied_intent/PushCube-v1/features/ \
    --out-dir models/stage5/p1_vision/PushCube-v1/
```

Wall time: ~30 s.

- [ ] **Step 2: Train StackCube pack**

```bash
python scripts/stage5_p1_train_pack.py \
    --jsonl datasets/stage4/varied_intent/StackCube-v1/samples.jsonl \
    --features-dir datasets/stage5/varied_intent/StackCube-v1/features/ \
    --out-dir models/stage5/p1_vision/StackCube-v1/
```

### Task S5.3: Reuse `stage4_m2a_run_eval.py` for the sim rollout

The existing `scripts/stage4_m2a_run_eval.py` consumes a saved `LatentPack`. It does **not** re-extract features from the JSONL — it reads the pack and runs the closed-loop sim eval. So it should work unchanged with the new pack.

- [ ] **Step 1: Check `stage4_m2a_run_eval.py` does not hard-wire `z_dim=20`**

```bash
grep -n "z_dim\|20" /scratch/gilbreth/wang4433/babysteps/scripts/stage4_m2a_run_eval.py | head -20
```

If hard-wired, copy the script to `scripts/stage5_p1_run_eval.py` and parametrize `z_dim` / `--features-dir` so the eval re-extracts features from cached DINOv2 .npy files at execution-time. Otherwise reuse directly.

- [ ] **Step 2: Run eval on PushCube (held-out seeds)**

> The exact CLI invocation depends on the existing eval script's flags — use whichever flag combination produced the published M2a result. Replace `<...>` accordingly.

```bash
python scripts/stage4_m2a_run_eval.py \
    --pack models/stage5/p1_vision/PushCube-v1/ \
    --task PushCube-v1 \
    --seeds <held-out-seed-range> \
    --out-dir reports/stage5/p1_vision_g4_g5/PushCube-v1/
```

- [ ] **Step 3: Run eval on StackCube**

```bash
python scripts/stage4_m2a_run_eval.py \
    --pack models/stage5/p1_vision/StackCube-v1/ \
    --task StackCube-v1 \
    --seeds <held-out-seed-range> \
    --out-dir reports/stage5/p1_vision_g4_g5/StackCube-v1/
```

### Task S5.4: Gate check and commit

- [ ] **Step 1: Inspect the eval report**

```bash
cat reports/stage5/p1_vision_g4_g5/PushCube-v1/report.md
cat reports/stage5/p1_vision_g4_g5/StackCube-v1/report.md
```

Per-task gates:
- **G4 (Δpp vs failure-agnostic retry):** `latent_revision_success_rate − same_intent_retry_success_rate ≥ 10pp`.
- **G5 (within 5pp of oracle):** `oracle_discrete_revision_success_rate − latent_revision_success_rate ≤ 5pp`.

- [ ] **Step 2: Commit**

```bash
git add scripts/stage5_p1_train_pack.py
[ -f scripts/stage5_p1_run_eval.py ] && git add scripts/stage5_p1_run_eval.py
git add reports/stage5/p1_vision_g4_g5/
git commit -m "feat(stage5 p1): vision-grounded LatentPack + G4/G5 eval

S5 of the Stage-5 P1 vision-encoder swap. New
scripts/stage5_p1_train_pack.py trains IntentHead(z_dim=768, d_slot=32)
and ReviseHead(d_slot=32) on cached DINOv2 features and saves a
LatentPack consumable by the existing stage4_m2a_run_eval.

Headline:
  PushCube  G4: Δpp = <X>pp   G5: within <Y>pp of oracle   [PASS/FAIL]
  StackCube G4: Δpp = <X>pp   G5: within <Y>pp of oracle   [PASS/FAIL]

(AttributionHead retraining on DINOv2 Z is a follow-up; rule-based
attribution from failure.py is the fallback for P1.)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: Update memory + milestones**

Update `MEMORY.md` with the Stage-5 P1 status: add a one-line pointer recording the G1 / G4 / G5 outcome, and amend `milestones.md` to mark P1 closed (or note which sub-gate failed).

---

## Self-Review

**Spec coverage check** (against `docs/superpowers/specs/2026-05-24-stage5-vision-encoder-swap-design.md`):

- § 3.1 Encoder choice (DINOv2 ViT-B/14, 768-dim CLS): S1.4 task — default `encoder="dinov2_vitb14"`.
- § 3.2 Pooling strategies (`cls_mean` default; `cls_first_last`, `spatial_mean` as ablations): S1.3 implements `cls_mean`; S4.3 ablation branch covers fallbacks.
- § 3.3 Frame source (resize to 224, ImageNet norm, all demo frames): S1.2 preprocesses to 224, normalizes; S2 captures all demo phase frames.
- § 3.4 IntentHead changes (`z_dim=768, d_slot=32, hidden=256`): S4.1 / S5.1 CLI defaults.
- § 3.5 Training protocol (per-slot CE + nested-CV G1): S4.1 reuses `nested_cv_probe_one_factor`.
- § 3.6 Episode format extension (`demo_frames_path` + `.npz`): handled implicitly by S2 output layout; no JSONL schema change required because S3/S4/S5 use seed-keyed file naming.
- § 4.1 G1 probe (≥ 90% non-trivial cells): S4.2 / S4.3 are the gate.
- § 4.3 Downstream G4/G5: S5.3 / S5.4.
- § 5 Implementation plan steps (S1–S6 in spec): mapped 1-to-1 to S1–S5 here (this plan merges spec S3+S4 into one section since the G1 report writer is the same code path as M2a's).
- § 6 Risks: addressed in Task S4.3 ablation branch.

**Placeholder scan:** the only intentional `<...>` placeholders are in the eval CLI in S5.3 Step 2 / Step 3 (`--seeds <held-out-seed-range>`) and in the commit message templates (`<X>pp`, `<Y>pp`, `<PASTE THE GATE HEADLINE>`). These are placeholders the executor fills with the actual numeric values *after* running the eval — not gaps in the implementation. All code blocks are complete.

**Type consistency:** `_preprocess_frames` → `(T, 3, 224, 224) float32 torch.Tensor`. `_pool_cls` consumes `(T, d) torch.Tensor` and returns `(d,) torch.Tensor`. `extract_vision_features` returns `(d,) float32 numpy.ndarray`. `nested_cv_probe_one_factor` expects `(N, z_dim) float32 numpy.ndarray` — matches `_load_one_task`'s `Z`. `IntentHead(z_dim=768, d_slot=32, hidden=256)` matches both S4 and S5. `ReviseHead(d_slot=32, hidden=256)` matches the IntentHead `d_slot`. Consistent.

---

## Open design decisions (call out before execution)

1. **`iter_demo_steps` interface on the runners.** The script in S2 assumes either (a) the runner already exposes a per-step demo generator, or (b) Task S2.2 adds one. The existing demo loop lives in `babysteps/episode.py` or per-task runners; the exact location must be confirmed during S2.1. If the existing demo loop is monolithic (single `runner.demo()` returning DemoEvidence), refactor minimally: extract a private `_iter_demo_step_with_render` generator inside the runner, and have the existing `demo()` consume it without rendering — i.e. keep `run_episode` byte-identical.

2. **`stage4_m2a_run_eval.py` reuse vs. fork.** S5.3 Step 1 verifies whether the eval script is z_dim-agnostic. If yes, reuse directly. If it hard-wires `z_dim=20` or re-extracts handcrafted features at eval-time, fork to `scripts/stage5_p1_run_eval.py`. The fork is small (~20 lines).

3. **AttributionHead in the vision pack.** M2.5's AttributionHead consumed 20-dim handcrafted Z. The P1 pack uses `attribution_head=None`, falling back to the rule-based `failure.py` attribution. Retraining AttributionHead on DINOv2 Z (~half-day of work) is deferred to a follow-up. If the rule-based attribution materially degrades G4 / G5 on vision G, surface this as a finding in S5.4 and add the retrain as Task S6.

4. **PickCube varied cut is not in scope for P1 v1.** The spec calls for it but neither the existing varied-intent collection nor the user's 5-step sequence include it. P1 ships with PushCube + StackCube; PickCube collection is a separate ~half-day GPU job that should follow once P1 closes.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-24-stage5-vision-encoder-swap-plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Good for this plan because S1 is heavily TDD'd (mechanical), and S2/S3/S5 have clear smoke-test gates between tasks. S4 is a hard go/no-go that benefits from a fresh-eyes review before S5.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Faster if you want to keep all context in one place and don't mind a longer single session.

**Which approach?**
