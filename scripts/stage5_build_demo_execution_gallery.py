# scripts/stage5_build_demo_execution_gallery.py
"""Build a visual inspection gallery for BABYSTEPS demo + execution videos.

Two questions this gallery answers, side by side, per seed:

1. **What did the encoder see?** — the third-person *demo* clip the latent
   intent is decoded from.
2. **What did BabySteps do with that intent?** — the first attempt, the
   failure/diagnosis state, the single-factor-revised retry, and the outcome.

It consumes the MP4s that already exist on disk (``renders/`` and
``datasets/``); it never re-renders. For every video it extracts four
representative key frames with OpenCV — ``first``, ``mid``,
``late`` (``n-5``), ``final`` — downscales them, and stitches a labelled
contact-sheet strip. It then emits a sectioned ``index.html`` (primary visual
gallery), a git-friendly ``index.md``, and a machine-readable
``gallery_manifest.json``.

Provenance honesty (two distinct classes of success label):

* **measured** — per-seed ``final_success`` read from a results JSON
  (the Stage-5 P1 latent eval and the Stage-0 PushCube report). These are
  stated as verified outcomes.
* **by-design** — the curated ``renders/<task>`` three-phase clips, whose
  ``2_attempt_blocked`` is a real failure and ``3_retry`` a revised success
  *by construction* (see ``renders/CLAUDE.md``). Labelled as the artifact's
  designed role, with the final frame shown as the actual visual evidence —
  never asserted as a measured per-seed flag.

Frames/strips land under ``frames/`` and ``strips/`` (gitignored); the
``index.*`` + manifest at the gallery root are committable.

Sim-free: pure OpenCV / NumPy / Pillow / stdlib. Runs on the login node.

Example::

    python scripts/stage5_build_demo_execution_gallery.py \\
        --out reports/stage5/demo_execution_gallery
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Repo-relative anchoring
# --------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent


def _rel(p: Path) -> str:
    """Path relative to the repo root, for display + manifest stability."""
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------
# Intent-factor scaffold (Stage-0 schema) + per-task revised factor
# --------------------------------------------------------------------------

INTENT_FIELDS = [
    "goal_state",
    "object_motion",
    "contact_region",
    "approach_direction",
    "constraint_region",
    "embodiment_mapping",
]

# Canonical revised factor per sub-project (CLAUDE.md sub-project table).
TASK_FACTOR = {
    "PushCube-v1": "approach_direction",   # Stage-0 clutter ablation lineage
    "PickCube-v1": "contact_region",
    "StackCube-v1": "goal_state",
    "TurnFaucet-v1": "embodiment_mapping",
}

TASK_PRETTY = {
    "PushCube-v1": "PushCube-v1",
    "PickCube-v1": "PickCube-v1",
    "StackCube-v1": "StackCube-v1",
    "TurnFaucet-v1": "TurnFaucet-v1",
}


# --------------------------------------------------------------------------
# Authoritative outcome reader — the burned-in caption
# --------------------------------------------------------------------------
#
# Every render burns ``(success=True)`` / ``(success=False)`` into the title
# banner (``babysteps/render/common.annotate_frame``, DejaVuSans-Bold 16 on a
# (16,16,16) band). That caption is the *rendered outcome* of that specific
# clip — the only authoritative per-seed record (no JSON sidecar exists, and
# the iconic composites predate the current eval run, so a JSON join would
# mislabel them). We read it back with a font-exact template match: render the
# distinguishing token in the same font and correlate it against the banner
# crop. A ladder of progressively shorter tokens survives titles whose long
# policy name truncates ``False`` to ``Fals`` at the frame edge. Demo clips
# carry no ``success=`` token and correctly read back as ``None``.

_FONT_BOLD = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"
# (true_token, false_token), most specific -> most truncation-robust.
_SUCCESS_TOKEN_LADDER = [("ess=True", "ess=Fals"), ("ss=Tr", "ss=Fa"), ("=Tr", "=Fa")]
_TOKEN_TEMPLATES: Optional[list] = None


def _render_token(tok: str, size: int = 16) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype(_FONT_BOLD, size)
    except Exception:
        font = ImageFont.load_default()
    probe = Image.new("RGB", (400, 60), (16, 16, 16))
    d = ImageDraw.Draw(probe)
    bbox = d.textbbox((0, 0), tok, font=font)
    w, h = bbox[2] - bbox[0] + 2, bbox[3] - bbox[1] + 2
    im = Image.new("RGB", (w, h), (16, 16, 16))
    ImageDraw.Draw(im).text((1 - bbox[0], 1 - bbox[1]), tok, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)


def _token_templates() -> list:
    global _TOKEN_TEMPLATES
    if _TOKEN_TEMPLATES is None:
        _TOKEN_TEMPLATES = [
            (_render_token(t), _render_token(f)) for t, f in _SUCCESS_TOKEN_LADDER
        ]
    return _TOKEN_TEMPLATES


def read_burned_success(
    video_path: Path, *, n_samples: int = 9, thr: float = 0.85, margin: float = 0.15,
    banner_h: int = 60,
) -> Optional[bool]:
    """Read ``success=True/False`` from a clip's burned-in title banner.

    Returns ``True``/``False`` when one token wins confidently (peak
    correlation ``>= thr`` and lead over the other ``>= margin``), else
    ``None`` (demo clips with no flag, or an unreadable banner). Samples
    several frames because composite videos hold the outcome only in their
    final phase.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = sorted({int(n * f) for f in np.linspace(0.0, 0.999, n_samples)})
    banners: list[np.ndarray] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok and fr is not None:
            banners.append(cv2.cvtColor(fr[:banner_h], cv2.COLOR_BGR2GRAY))
    cap.release()
    return classify_banner_success(banners, thr=thr, margin=margin)


def classify_banner_success(
    banners: list, *, thr: float = 0.85, margin: float = 0.15,
) -> Optional[bool]:
    """Core matcher: given grayscale title-banner crops, return the burned-in
    ``success`` bool via the font-exact token ladder, or ``None`` if no token
    wins confidently. Factored out of :func:`read_burned_success` so it is
    testable without an MP4."""
    if not banners:
        return None
    for tt, ft in _token_templates():
        bt = bf = -1.0
        for b in banners:
            if tt.shape[0] <= b.shape[0] and tt.shape[1] <= b.shape[1]:
                bt = max(bt, float(cv2.matchTemplate(b, tt, cv2.TM_CCOEFF_NORMED).max()))
            if ft.shape[0] <= b.shape[0] and ft.shape[1] <= b.shape[1]:
                bf = max(bf, float(cv2.matchTemplate(b, ft, cv2.TM_CCOEFF_NORMED).max()))
        if max(bt, bf) >= thr and abs(bt - bf) >= margin:
            return bt > bf
    return None


# --------------------------------------------------------------------------
# Frame extraction
# --------------------------------------------------------------------------

@dataclass
class FrameSet:
    """Extracted key frames for one MP4 (paths are repo-relative strings)."""

    video: str
    n_frames: int
    fps: float
    width: int
    height: int
    frames: dict = field(default_factory=dict)   # role -> rel path
    strip: Optional[str] = None                  # rel path to contact sheet
    video_link: Optional[str] = None             # rel path to playable mp4 (symlink)


def pick_frame_indices(n: int) -> dict:
    """Map role -> frame index for an ``n``-frame clip.

    ``first`` = 0, ``mid`` = n//2, ``late`` = the start of the last-5 window
    (``n-5`` clamped), ``final`` = n-1. All clamped to ``[0, n-1]`` and
    de-collided for very short clips so the strip never repeats a frame
    needlessly.
    """
    if n <= 0:
        return {}
    last = n - 1
    idx = {
        "first": 0,
        "mid": n // 2,
        "late": max(0, n - 5),
        "final": last,
    }
    # Clamp.
    idx = {k: min(max(0, v), last) for k, v in idx.items()}
    # For tiny clips keep the ordering monotone where possible.
    order = ["first", "mid", "late", "final"]
    prev = -1
    for k in order:
        if idx[k] <= prev and prev < last:
            idx[k] = min(prev + 1, last)
        prev = idx[k]
    return idx


def _resize_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    scale = width / float(w)
    return cv2.resize(img, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def _label_band(img: np.ndarray, text: str) -> np.ndarray:
    """Draw a small label bar at the top-left of a BGR frame (in place copy)."""
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(out, (0, 0), (tw + 10, th + base + 8), (0, 0, 0), -1)
    cv2.putText(out, text, (5, th + 4), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def extract_frames(
    video_path: Path,
    out_dir: Path,
    *,
    stem: str,
    width: int,
    jpeg_quality: int,
    make_strip: bool = True,
) -> Optional[FrameSet]:
    """Extract the four key frames + a contact-sheet strip for one MP4."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # CAP_PROP_FRAME_COUNT can be unreliable; verify by reading if small/zero.
    if n <= 0:
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    idx = pick_frame_indices(n)
    if not idx:
        cap.release()
        return None

    fs = FrameSet(video=_rel(video_path), n_frames=n, fps=fps, width=w, height=h)
    grabbed: dict[str, np.ndarray] = {}
    for role, fi in idx.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            # Fallback: sequential scan to fi (some codecs dislike seeking).
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = None
            for j in range(fi + 1):
                ok2, f2 = cap.read()
                if not ok2:
                    break
                frame = f2
            if frame is None:
                continue
        frame = _resize_to_width(frame, width)
        grabbed[role] = frame
        fp = out_dir / f"{stem}__{role}.jpg"
        cv2.imwrite(str(fp), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        fs.frames[role] = _rel(fp)
    cap.release()

    if make_strip and grabbed:
        order = [r for r in ("first", "mid", "late", "final") if r in grabbed]
        labelled = [_label_band(grabbed[r], r) for r in order]
        target_h = min(im.shape[0] for im in labelled)
        resized = [
            cv2.resize(im, (int(im.shape[1] * target_h / im.shape[0]), target_h))
            for im in labelled
        ]
        gap = 4
        sep = np.full((target_h, gap, 3), 255, np.uint8)
        pieces: list[np.ndarray] = []
        for i, im in enumerate(resized):
            pieces.append(im)
            if i != len(resized) - 1:
                pieces.append(sep)
        strip = np.concatenate(pieces, axis=1)
        sp = out_dir.parent / "strips" / f"{stem}__strip.jpg"
        sp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sp), strip, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        fs.strip = _rel(sp)

    # Symlink the source MP4 into videos/ so the HTML can embed a real player
    # (zero-copy; the source clips total ~83 MB and live under gitignored dirs).
    vid_dir = out_dir.parent / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    link = vid_dir / f"{stem}.mp4"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(video_path.resolve())
        fs.video_link = _rel(link)
    except OSError:
        fs.video_link = None

    return fs


# --------------------------------------------------------------------------
# Metadata loaders
# --------------------------------------------------------------------------

def _load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_iconic_metadata() -> dict:
    """Per-seed outcome + intent metadata for the iconic PushCube latent set.

    Joins:
      * ``p1_vision_g4_g5_latent`` p1_results.json -> per-policy per-seed
        final/initial/retry success.
      * ``p2_vlm_latent`` c1_results.json -> per-seed gt_intent, decoded
        ``initial_intent``, ``revised_intent``, ``vlm_factor``,
        ``factors_changed``, attribution + preservation flags.
    Returns ``{seed: {...}}`` (intent/factor fields) and
    ``{(policy, seed): {...}}`` (per-policy success) under two keys.
    """
    p1 = _load_json(
        REPO / "reports/stage5/p1_vision_g4_g5_latent/PushCube-v1/p1_results.json"
    )
    c1 = _load_json(
        REPO / "reports/stage5/p2_vlm_latent/PushCube-v1/c1_results.json"
    )
    by_seed: dict[int, dict] = {}
    by_policy_seed: dict[tuple, dict] = {}

    if p1:
        for policy, blob in p1.get("per_policy", {}).items():
            for row in blob.get("rows", []):
                by_policy_seed[(policy, int(row["seed"]))] = {
                    "initial_success": row.get("initial_success"),
                    "retry_success": row.get("retry_success"),
                    "final_success": row.get("final_success"),
                }
    if c1:
        for ep in c1.get("per_episode", []):
            by_seed[int(ep["seed"])] = {
                "gt_intent": ep.get("gt_intent"),
                "initial_intent": ep.get("initial_intent"),
                "revised_intent": ep.get("revised_intent"),
                "vlm_factor": ep.get("vlm_factor"),
                "factors_changed": ep.get("factors_changed"),
                "attribution_correct": ep.get("attribution_correct"),
                "frozen_factor_preserved": ep.get("frozen_factor_preserved"),
                "final_success": ep.get("final_success"),
                "retry_success": ep.get("retry_success"),
                "latent_matches_stored": ep.get("latent_matches_stored"),
            }
    return {"by_seed": by_seed, "by_policy_seed": by_policy_seed}


def load_stage0_pushcube_report() -> Optional[dict]:
    return _load_json(REPO / "datasets/stage0_pushcube_blocked/report.json")


# --------------------------------------------------------------------------
# Group discovery
# --------------------------------------------------------------------------

SEED_RE = re.compile(r"seed_(\d+)")


def _seed_of(name: str) -> Optional[int]:
    m = SEED_RE.search(name)
    return int(m.group(1)) if m else None


@dataclass
class TripletRow:
    group: str
    section: str
    task: str
    seed: int
    factor: str
    demo: Optional[FrameSet] = None
    attempt1: Optional[FrameSet] = None
    retry: Optional[FrameSet] = None
    attempt1_tag: str = "2_attempt_blocked"
    retry_success: Optional[bool] = None     # read from retry clip caption
    attempt_success: Optional[bool] = None   # read from attempt clip caption
    success_label: str = ""                  # human string
    notes: str = ""
    caveat: str = ""                         # e.g. wristcam-artifact warning


@dataclass
class CompositeRow:
    group: str
    section: str
    task: str
    seed: int
    condition: str
    factor: str
    fs: Optional[FrameSet] = None
    success: Optional[bool] = None           # read from composite caption
    meta: dict = field(default_factory=dict)
    success_label: str = ""
    caveat: str = ""                         # e.g. wristcam-artifact warning


@dataclass
class IntentTableRow:
    """A measured per-seed latent-intent decode (current P1/P2 run; no video)."""
    seed: int
    final_success: Optional[bool]
    retry_success: Optional[bool]
    vlm_factor: Optional[str]
    factors_changed: Optional[list]
    attribution_correct: Optional[bool]
    frozen_factor_preserved: Optional[bool]
    latent_matches_stored: Optional[bool]
    diff: list = field(default_factory=list)   # intent_diff_rows output


@dataclass
class SingleRow:
    group: str
    section: str
    label: str
    fs: Optional[FrameSet] = None
    caption: str = ""


# The May-2x PushCube renders + iconic composites use the `panda_wristcam`
# robot variant (first-person execution view). On that variant the retry's
# success flag reads `False` even when the cube reaches the target — a
# documented controller×wristcam interaction (see renders/results/README.md),
# NOT a method failure. The standard-`panda` Stage-0 render and the measured
# eval both show the revised retry succeeding. We surface the burned-in value
# faithfully but flag it so it is not misread.
WRISTCAM_CAVEAT = (
    "⚠ First-person panda_wristcam render: the burned-in retry success flag is "
    "UNRELIABLE on this variant (reads False even when the cube reaches the "
    "target — a documented controller×wristcam artifact, see "
    "renders/results/README.md). Trust the standard-panda Stage-0 PushCube "
    "section and the measured June-3 eval (latent retry success 0.96) for the "
    "real outcome, not these flags."
)

# Declarative registry of triplet video sets.
TRIPLET_SPECS = [
    dict(
        name="stage0_pushcube_blocked",
        section="Stage-0 PushCube — blocked approach (standard-panda, third-person; trustworthy)",
        dir="datasets/stage0_pushcube_blocked/videos_maniskill",
        prefix="pushcube_blocked_approach",
        task="PushCube-v1",
        factor="approach_direction",
        attempt_tag="2_attempt_blocked",
    ),
    dict(
        name="pushcube_render",
        section="PushCube — curated three-phase clips (wristcam)",
        dir="renders/pushcube/videos_maniskill",
        prefix="pushcube_blocked_approach",
        task="PushCube-v1",
        factor="approach_direction",
        attempt_tag="2_attempt_blocked",
        caveat=WRISTCAM_CAVEAT,
    ),
    dict(
        name="pushcube_clutter",
        section="PushCube — clutter ablation clips (wristcam)",
        dir="renders/pushcube_clutter/videos_maniskill",
        prefix="pushcube_blocked_approach",
        task="PushCube-v1",
        factor="approach_direction",
        attempt_tag="2_attempt_blocked",
        caveat=WRISTCAM_CAVEAT,
    ),
    dict(
        name="pushcube_paper_figure",
        section="PushCube — paper figure (contact_region, wrong-intent attempt, wristcam)",
        dir="renders/pushcube/videos_paper_figure/2026-05-28_164447",
        prefix="pushcube_blocked_approach",
        task="PushCube-v1",
        factor="contact_region",
        attempt_tag="2_attempt_wrong_intent",
        caveat=WRISTCAM_CAVEAT,
    ),
    dict(
        name="pickcube_render",
        section="PickCube — grasp slip (contact_region)",
        dir="renders/pickcube/videos_maniskill",
        prefix="pickcube_grasp_slip",
        task="PickCube-v1",
        factor="contact_region",
        attempt_tag="2_attempt_blocked",
    ),
    dict(
        name="stackcube_render",
        section="StackCube — underspecified goal (goal_state)",
        dir="renders/stackcube/videos_maniskill",
        prefix="stackcube_underspec_goal",
        task="StackCube-v1",
        factor="goal_state",
        attempt_tag="2_attempt_blocked",
    ),
    dict(
        name="turnfaucet_render",
        section="TurnFaucet — wrong contact (grasp-turn lineage)",
        dir="renders/turnfaucet/videos_maniskill",
        prefix="turnfaucet_wrong_contact",
        task="TurnFaucet-v1",
        factor="embodiment_mapping",
        attempt_tag="2_attempt_blocked",
    ),
    dict(
        name="turnfaucet_embodiment",
        section="TurnFaucet — embodiment substitution",
        dir="renders/turnfaucet_embodiment/videos_maniskill",
        prefix="turnfaucet_wrong_contact",
        task="TurnFaucet-v1",
        factor="embodiment_mapping",
        attempt_tag="2_attempt_blocked",
    ),
]

ICONIC_SPEC = dict(
    name="iconic_latent",
    section="PushCube latent loop — iconic composites (wristcam; flags unreliable)",
    dir="renders/stage5_p1_iconic/pushcube",
    task="PushCube-v1",
    caveat=WRISTCAM_CAVEAT,
    conditions={
        "latent": "Latent input + learned slot-local edit (BabySteps, the method)",
        "babysteps_selective": "Oracle input + learned selective edit",
        "oracle_factor_revision": "Oracle single-factor revision (skyline)",
        "same_intent_retry": "Retry with unchanged intent (control — must fail)",
    },
)

SINGLE_SPECS = [
    dict(
        name="official_demo",
        section="Official ManiSkill MP demo replays (third-person demo source)",
        files=[
            ("renders/official_demo_smoke/PushCube-v1_seed_0000__official_replay.mp4", "PushCube-v1 official demo replay"),
            ("renders/official_demo_smoke/StackCube-v1_seed_0000__official_replay.mp4", "StackCube-v1 official demo replay"),
            ("renders/official_demo_smoke/PickCube-v1_seed_0000__official_replay.mp4", "PickCube-v1 official demo replay"),
        ],
        caption="Official ManiSkill motion-planning demos replayed third-person — the demo source for Scope-A 1_demo clips.",
    ),
    dict(
        name="comparison",
        section="Task comparison montages",
        files=[
            ("renders/comparison/PushCube_comparison.mp4", "PushCube comparison"),
            ("renders/comparison/PickCube_comparison.mp4", "PickCube comparison"),
            ("renders/comparison/StackCube_comparison.mp4", "StackCube comparison"),
            ("renders/comparison/TurnFaucet_comparison.mp4", "TurnFaucet comparison"),
        ],
        caption="Side-by-side montages (demo | attempt | retry composited into one frame).",
    ),
    dict(
        name="annotated_result",
        section="Annotated latent-intent result (paper deliverable)",
        files=[
            ("renders/results/stage4_latent_intent_pushcube_seed_0000.mp4", "Annotated demo→blocked→revised retry (seed 0000)"),
        ],
        caption="Burned-in captions: inferred 6-slot intent, failure predicate, learned attribution, single-factor edit.",
    ),
]


def discover_triplet_rows(spec: dict, frames_dir: Path, *, width, jpeg_quality,
                          max_seeds: Optional[int], stage0_report) -> list[TripletRow]:
    vdir = REPO / spec["dir"]
    if not vdir.is_dir():
        return []
    seeds = sorted({
        s for f in vdir.glob("*.mp4")
        if (s := _seed_of(f.name)) is not None
    })
    if max_seeds is not None:
        seeds = seeds[:max_seeds]
    rows: list[TripletRow] = []
    for seed in seeds:
        stem_base = f"{spec['prefix']}_seed_{seed:04d}"
        demo_p = vdir / f"{stem_base}__1_demo.mp4"
        att_p = vdir / f"{stem_base}__{spec['attempt_tag']}.mp4"
        retry_p = vdir / f"{stem_base}__3_retry.mp4"
        row = TripletRow(
            group=spec["name"], section=spec["section"], task=spec["task"],
            seed=seed, factor=spec["factor"], attempt1_tag=spec["attempt_tag"],
            caveat=spec.get("caveat", ""),
        )
        gid = spec["name"]
        if demo_p.exists():
            row.demo = extract_frames(demo_p, frames_dir, stem=f"{gid}_{seed:04d}_demo",
                                      width=width, jpeg_quality=jpeg_quality)
        if att_p.exists():
            row.attempt1 = extract_frames(att_p, frames_dir, stem=f"{gid}_{seed:04d}_attempt1",
                                          width=width, jpeg_quality=jpeg_quality)
            row.attempt_success = read_burned_success(att_p)
        if retry_p.exists():
            row.retry = extract_frames(retry_p, frames_dir, stem=f"{gid}_{seed:04d}_retry",
                                       width=width, jpeg_quality=jpeg_quality)
            row.retry_success = read_burned_success(retry_p)

        # Honest per-seed status: read straight from each clip's burned-in
        # caption (the rendered outcome). No "by-design" assumption — several
        # curated retries actually FAIL (e.g. PickCube/TurnFaucet seed 0000).
        def _phrase(v, fallback):
            return {True: "SUCCESS", False: "FAILURE"}.get(v, fallback)
        att = _phrase(row.attempt_success, "blocked/failed (designed failure)")
        ret = _phrase(row.retry_success, "outcome unread — see final frame")
        row.success_label = (
            f"attempt-1: {att} · revised retry: {ret}  "
            f"(read from burned-in caption)"
        )
        if spec["name"] == "stage0_pushcube_blocked" and stage0_report:
            row.notes = (
                f"Corroborated by aggregate report: initial "
                f"{stage0_report.get('initial_attempt_success_rate'):.0%} success, "
                f"retry {stage0_report.get('retry_success_rate'):.0%} success over "
                f"n={stage0_report.get('n_total')}."
            )
        rows.append(row)
    return rows


def discover_iconic_rows(spec: dict, frames_dir: Path, *, width, jpeg_quality,
                         max_seeds) -> list[CompositeRow]:
    """Iconic composites — outcome read from each video's own burned-in
    caption (these renders predate the current eval, so a JSON join would
    mislabel them). The intent transition is burned into the subtitle and is
    visible in the extracted frames; the current measured decode is reported
    separately in :func:`build_intent_table_rows`."""
    vdir = REPO / spec["dir"]
    if not vdir.is_dir():
        return []
    seeds = sorted({
        s for f in vdir.glob("*.mp4") if (s := _seed_of(f.name)) is not None
    })
    if max_seeds is not None:
        seeds = seeds[:max_seeds]
    rows: list[CompositeRow] = []
    for seed in seeds:
        for cond, cond_desc in spec["conditions"].items():
            vp = vdir / f"pushcube_seed_{seed:04d}__{cond}_full.mp4"
            if not vp.exists():
                continue
            fs = extract_frames(vp, frames_dir,
                                stem=f"iconic_{seed:04d}_{cond}",
                                width=width, jpeg_quality=jpeg_quality)
            succ = read_burned_success(vp)
            label = (
                f"{ {True:'SUCCESS', False:'FAILURE'}.get(succ, 'outcome unread') } "
                f"(read from burned-in caption; this is the May-24 iconic render)"
            )
            rows.append(CompositeRow(
                group=spec["name"], section=spec["section"], task=spec["task"],
                seed=seed, condition=cond, factor=TASK_FACTOR[spec["task"]],
                fs=fs, success=succ,
                meta={"condition_desc": cond_desc},
                success_label=label,
                caveat=spec.get("caveat", ""),
            ))
    return rows


def build_intent_table_rows(iconic_meta: dict, *, seeds: list[int]) -> list[IntentTableRow]:
    """Per-seed measured latent-intent decode from the current P1/P2 run.

    Sourced purely from ``p2_vlm_latent`` (June-3 measured eval). Kept SEPARATE
    from the May-24 iconic videos so the two runs are never conflated.
    """
    by_seed = iconic_meta["by_seed"]
    rows: list[IntentTableRow] = []
    for seed in seeds:
        m = by_seed.get(seed)
        if not m:
            continue
        rows.append(IntentTableRow(
            seed=seed,
            final_success=m.get("final_success"),
            retry_success=m.get("retry_success"),
            vlm_factor=m.get("vlm_factor"),
            factors_changed=m.get("factors_changed"),
            attribution_correct=m.get("attribution_correct"),
            frozen_factor_preserved=m.get("frozen_factor_preserved"),
            latent_matches_stored=m.get("latent_matches_stored"),
            diff=intent_diff_rows(m.get("gt_intent"), m.get("initial_intent"),
                                  m.get("revised_intent")),
        ))
    return rows


def discover_single_rows(spec: dict, frames_dir: Path, *, width, jpeg_quality) -> list[SingleRow]:
    rows: list[SingleRow] = []
    for relpath, label in spec["files"]:
        vp = REPO / relpath
        if not vp.exists():
            continue
        stem = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_").lower()
        fs = extract_frames(vp, frames_dir, stem=f"{spec['name']}_{stem}",
                            width=width, jpeg_quality=jpeg_quality)
        rows.append(SingleRow(group=spec["name"], section=spec["section"],
                              label=label, fs=fs, caption=spec.get("caption", "")))
    return rows


# --------------------------------------------------------------------------
# Intent diff helper (decoded vs ground-truth scaffold)
# --------------------------------------------------------------------------

def intent_diff_rows(gt: Optional[dict], decoded: Optional[dict],
                     revised: Optional[dict]) -> list[tuple]:
    """Return (factor, gt, decoded, revised, flag) per intent field."""
    out = []
    for f in INTENT_FIELDS:
        g = (gt or {}).get(f)
        d = (decoded or {}).get(f)
        r = (revised or {}).get(f)
        flag = ""
        if g is not None and d is not None:
            flag = "match" if g == d else "MISMATCH"
        if r is not None and d is not None and r != d:
            flag = (flag + " · " if flag else "") + "REVISED"
        out.append((f, g, d, r, flag))
    return out


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------

def _img(rel_from_root: Optional[str], gallery_root: Path, *, cls="frm") -> str:
    if not rel_from_root:
        return '<span class="missing">—</span>'
    # rel_from_root is repo-relative; HTML lives in gallery_root; compute path.
    abs_p = REPO / rel_from_root
    try:
        href = os.path.relpath(abs_p, gallery_root)
    except ValueError:
        href = str(abs_p)
    return f'<img class="{cls}" src="{html.escape(href)}" loading="lazy">'


def _frame(fs: Optional[FrameSet], role: str, gallery_root: Path) -> str:
    if fs is None:
        return '<span class="missing">(no video)</span>'
    return _img(fs.frames.get(role), gallery_root)


def _video(fs: Optional[FrameSet], gallery_root: Path, *, poster_role: str = "final") -> str:
    """Embed a playable <video> for this clip, with a key frame as poster."""
    if fs is None or not fs.video_link:
        return '<span class="missing">(no video)</span>'
    src = os.path.relpath(REPO / fs.video_link, gallery_root)
    pr = fs.frames.get(poster_role) or fs.frames.get("mid")
    poster = ""
    if pr:
        poster = f' poster="{html.escape(os.path.relpath(REPO / pr, gallery_root))}"'
    return (
        f'<video class="vid" controls muted loop playsinline preload="metadata"{poster}>'
        f'<source src="{html.escape(src)}" type="video/mp4">'
        f'<a href="{html.escape(src)}">download clip</a></video>'
    )


HTML_CSS = """
:root { --bg:#0d1117; --fg:#e6edf3; --mut:#8b949e; --card:#161b22; --bd:#30363d;
        --ok:#3fb950; --bad:#f85149; --warn:#d29922; --acc:#58a6ff; }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--fg); font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
       margin:0; padding:0 0 80px; }
header { padding:24px 32px; border-bottom:1px solid var(--bd); position:sticky; top:0;
         background:var(--bg); z-index:10; }
h1 { margin:0 0 6px; font-size:22px; }
h2 { margin:36px 32px 8px; font-size:18px; color:var(--acc); scroll-margin-top:90px; }
.sub { color:var(--mut); font-size:13px; }
nav { margin:12px 32px 0; }
nav a { color:var(--acc); text-decoration:none; margin-right:14px; font-size:12px; white-space:nowrap; }
.section { padding:0 32px; }
.row { background:var(--card); border:1px solid var(--bd); border-radius:10px;
       margin:14px 0; padding:14px; }
.rowhead { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:10px; }
.rowhead .seed { font-weight:700; font-size:15px; }
.chip { background:#21262d; border:1px solid var(--bd); border-radius:20px;
        padding:2px 10px; font-size:12px; color:var(--mut); }
.chip.factor { color:var(--acc); border-color:var(--acc); }
.badge { padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge.ok { background:rgba(63,185,80,.15); color:var(--ok); border:1px solid var(--ok); }
.badge.bad { background:rgba(248,81,73,.15); color:var(--bad); border:1px solid var(--bad); }
.badge.des { background:rgba(210,153,34,.12); color:var(--warn); border:1px solid var(--warn); }
.phases { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; }
.phases.cols3 { grid-template-columns:repeat(3,1fr); }
.phases.cols1 { grid-template-columns:minmax(0,640px); }
.phase { display:flex; flex-direction:column; gap:6px; }
.vid { width:100%; border-radius:6px; border:1px solid var(--bd); display:block; background:#000; }
.phase .cap { font-size:11px; color:var(--mut); text-transform:uppercase; letter-spacing:.04em; }
.frm { width:100%; border-radius:6px; border:1px solid var(--bd); display:block; }
.missing { color:var(--mut); font-style:italic; font-size:12px; }
.strip { width:100%; border-radius:6px; border:1px solid var(--bd); margin-top:8px; }
table.intent { border-collapse:collapse; width:100%; margin-top:10px; font-size:12px; }
table.intent th, table.intent td { border:1px solid var(--bd); padding:3px 8px; text-align:left; }
table.intent th { background:#21262d; color:var(--mut); }
.MISMATCH { color:var(--bad); font-weight:600; }
.REVISED { color:var(--warn); font-weight:600; }
.match { color:var(--ok); }
details { margin-top:8px; }
summary { cursor:pointer; color:var(--acc); font-size:12px; }
.vidlink { font-size:11px; color:var(--mut); }
.note { font-size:12px; color:var(--mut); margin-top:6px; }
.caveat { font-size:12px; color:var(--warn); background:rgba(210,153,34,.10);
          border:1px solid var(--warn); border-radius:6px; padding:6px 10px; margin-top:8px; }
.secnote { margin:0 32px 4px; font-size:13px; color:var(--warn);
           background:rgba(210,153,34,.10); border:1px solid var(--warn);
           border-radius:6px; padding:8px 12px; }
.cond { font-size:13px; color:var(--fg); font-weight:600; }
@media (max-width:900px){ .phases{ grid-template-columns:repeat(2,1fr);} }
"""


def _badge(success: Optional[bool], *, label_true="SUCCESS", label_false="FAILURE") -> str:
    if success is True:
        return f'<span class="badge ok">{label_true}</span>'
    if success is False:
        return f'<span class="badge bad">{label_false}</span>'
    return '<span class="badge des">outcome unread</span>'


def render_html(sections: list[tuple], gallery_root: Path, stats: dict) -> str:
    parts = ["<!doctype html><html><head><meta charset=utf-8>",
             "<meta name=viewport content='width=device-width,initial-scale=1'>",
             "<title>BabySteps — Demo & Execution Gallery</title>",
             f"<style>{HTML_CSS}</style></head><body>"]
    parts.append("<header><h1>BABYSTEPS — Demo &amp; Execution Inspection Gallery</h1>")
    parts.append(f"<div class=sub>{html.escape(stats['summary'])}</div>")
    parts.append(
        '<div class="secnote" style="margin:10px 0 0"><b>Each panel is a playable clip</b> '
        '— click ▶ to watch demo vs attempt-1 vs revised retry side by side (muted, loops). '
        'How to read outcomes: every '
        'SUCCESS/FAILURE badge is read back from the clip\'s own burned-in caption '
        '(font-exact template match), never assumed. Two sources are trustworthy for '
        'the method\'s real performance — the <b>Stage-0 PushCube</b> section '
        '(standard-panda, third-person) and the <b>measured June-3 intent decode</b> '
        'table. Sections marked <b>wristcam</b> use the panda_wristcam variant whose '
        'retry success flag is a documented artifact (reads False even when the cube '
        'reaches the target); their badges are neutralised to ⚠ and must not be read '
        'as method failures.</div>')
    parts.append("<nav>")
    for title, _ in sections:
        anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        parts.append(f'<a href="#{anchor}">{html.escape(title.split("—")[0].strip())}</a>')
    parts.append("</nav></header>")

    for title, rows in sections:
        anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        parts.append(f'<h2 id="{anchor}">{html.escape(title)}</h2><div class=section>')
        for row in rows:
            parts.append(render_row_html(row, gallery_root))
        parts.append("</div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def render_row_html(row, gallery_root: Path) -> str:
    if isinstance(row, TripletRow):
        return _triplet_html(row, gallery_root)
    if isinstance(row, CompositeRow):
        return _composite_html(row, gallery_root)
    if isinstance(row, IntentTableRow):
        return _intent_table_html(row)
    if isinstance(row, SingleRow):
        return _single_html(row, gallery_root)
    return ""


def _phase_block(cap: str, inner: str) -> str:
    return f'<div class="phase"><div class="cap">{html.escape(cap)}</div>{inner}</div>'


def _triplet_html(row: TripletRow, gr: Path) -> str:
    if row.caveat:
        badge = '<span class="badge des">⚠ wristcam — flag unreliable</span>'
    else:
        badge = _badge(row.retry_success, label_true="retry SUCCESS", label_false="retry FAILURE")
    h = ['<div class="row">']
    h.append('<div class="rowhead">')
    h.append(f'<span class="seed">{html.escape(row.task)} · seed {row.seed:04d}</span>')
    h.append(f'<span class="chip factor">revised: {html.escape(row.factor)}</span>')
    h.append(f'<span class="chip">{html.escape(row.group)}</span>')
    h.append(badge)
    h.append('</div>')
    h.append('<div class="phases cols3">')
    h.append(_phase_block("▶ demo (third-person)", _video(row.demo, gr, poster_role="mid")))
    h.append(_phase_block("▶ attempt 1 — initial (fails)", _video(row.attempt1, gr, poster_role="final")))
    h.append(_phase_block("▶ revised retry", _video(row.retry, gr, poster_role="final")))
    h.append('</div>')
    h.append(f'<div class="note">{html.escape(row.success_label)}</div>')
    if row.notes:
        h.append(f'<div class="note">{html.escape(row.notes)}</div>')
    if row.caveat:
        h.append(f'<div class="caveat">{html.escape(row.caveat)}</div>')
    # strips in a details
    h.append('<details><summary>key-frame strips (first · mid · late · final)</summary>')
    for nm, fs in (("demo", row.demo), ("attempt1", row.attempt1), ("retry", row.retry)):
        if fs and fs.strip:
            h.append(f'<div class="cap">{nm} — {html.escape(os.path.basename(fs.video))} '
                     f'(n={fs.n_frames})</div>')
            h.append(_img(fs.strip, gr, cls="strip"))
    h.append('</details>')
    h.append('</div>')
    return "".join(h)


def _composite_html(row: CompositeRow, gr: Path) -> str:
    m = row.meta
    badge = ('<span class="badge des">⚠ wristcam — flag unreliable</span>'
             if row.caveat else _badge(row.success))
    h = ['<div class="row">']
    h.append('<div class="rowhead">')
    h.append(f'<span class="seed">seed {row.seed:04d}</span>')
    h.append(f'<span class="cond">{html.escape(row.condition)}</span>')
    h.append(f'<span class="chip factor">revised: {html.escape(row.factor)}</span>')
    h.append(badge)
    h.append('</div>')
    h.append(f'<div class="note">{html.escape(m.get("condition_desc",""))}</div>')
    # composite is one clip covering demo -> attempt -> retry in sequence
    h.append('<div class="phases cols1">')
    h.append(_phase_block("▶ full clip: demo → attempt → retry", _video(row.fs, gr, poster_role="final")))
    h.append('</div>')
    h.append(f'<div class="note">{html.escape(row.success_label)} — '
             f'the intent transition is burned into the subtitle (visible above).</div>')
    if row.caveat:
        h.append(f'<div class="caveat">{html.escape(row.caveat)}</div>')
    if row.fs and row.fs.strip:
        h.append('<details><summary>key-frame strip</summary>')
        h.append(_img(row.fs.strip, gr, cls="strip"))
        h.append('</details>')
    h.append('</div>')
    return "".join(h)


def _intent_table_html(row: IntentTableRow) -> str:
    h = ['<div class="row">']
    h.append('<div class="rowhead">')
    h.append(f'<span class="seed">seed {row.seed:04d}</span>')
    h.append(_badge(row.final_success))
    if row.vlm_factor is not None:
        h.append(f'<span class="chip factor">VLM factor: {html.escape(str(row.vlm_factor))}</span>')
    if row.factors_changed is not None:
        ch = ", ".join(row.factors_changed) if row.factors_changed else "∅"
        h.append(f'<span class="chip">changed: {html.escape(ch)}</span>')
    if row.attribution_correct is not None:
        h.append(f'<span class="chip">attribution_correct: {row.attribution_correct}</span>')
    if row.frozen_factor_preserved is not None:
        h.append(f'<span class="chip">frozen_preserved: {row.frozen_factor_preserved}</span>')
    if row.latent_matches_stored is not None:
        h.append(f'<span class="chip">latent==stored: {row.latent_matches_stored}</span>')
    h.append('</div>')
    h.append('<table class="intent"><tr><th>factor</th><th>ground-truth scaffold</th>'
             '<th>decoded (latent)</th><th>revised</th><th>flag</th></tr>')
    for f, g, d, r, flag in row.diff:
        cls = "MISMATCH" if "MISMATCH" in flag else ("REVISED" if "REVISED" in flag else "match")
        h.append(f'<tr><td>{html.escape(f)}</td><td>{html.escape(str(g))}</td>'
                 f'<td>{html.escape(str(d))}</td><td>{html.escape(str(r))}</td>'
                 f'<td class="{cls}">{html.escape(flag)}</td></tr>')
    h.append('</table></div>')
    return "".join(h)


def _single_html(row: SingleRow, gr: Path) -> str:
    h = ['<div class="row">']
    h.append('<div class="rowhead">')
    h.append(f'<span class="seed">{html.escape(row.label)}</span>')
    h.append('</div>')
    if row.caption:
        h.append(f'<div class="note">{html.escape(row.caption)}</div>')
    h.append('<div class="phases cols1">')
    h.append(_phase_block("▶ clip", _video(row.fs, gr, poster_role="mid")))
    h.append('</div>')
    if row.fs and row.fs.strip:
        h.append('<details><summary>key-frame strip</summary>')
        h.append(_img(row.fs.strip, gr, cls="strip"))
        h.append('</details>')
    h.append('</div>')
    return "".join(h)


# --------------------------------------------------------------------------
# Markdown rendering (git-friendly index)
# --------------------------------------------------------------------------

def render_markdown(sections: list[tuple], gallery_root: Path, stats: dict) -> str:
    L = ["# BABYSTEPS — Demo & Execution Inspection Gallery", "",
         stats["summary"], "",
         "> Open `index.html` for the full side-by-side visual gallery. "
         "Frames live under `frames/`+`strips/` (gitignored); regenerate with "
         "`python scripts/stage5_build_demo_execution_gallery.py`.", "",
         "**Reading outcomes.** Every SUCCESS/FAILURE label is read back from the "
         "clip's own burned-in caption (font-exact template match), never assumed. "
         "Trustworthy for the method's real performance: the **Stage-0 PushCube** "
         "section (standard-panda, third-person) and the **measured June-3 intent "
         "decode** table. Sections marked **wristcam** use the `panda_wristcam` "
         "variant whose retry success flag is a documented artifact (reads False even "
         "when the cube reaches the target — see `renders/results/README.md`); those "
         "are flagged ⚠ and must not be read as method failures.", "",
         "**Coverage.** This gallery extracts every unique-content MP4 under "
         "`renders/` and `datasets/`. Timestamped-subdir re-renders (identical "
         "seed sets) and the `official_demo_verify/` re-verification clips are "
         "skipped as duplicates of the parent-directory clips shown here.", ""]
    for title, rows in sections:
        L.append(f"## {title}")
        L.append("")
        for row in rows:
            if isinstance(row, TripletRow):
                if row.caveat:
                    rs = "⚠ wristcam (flag unreliable)"
                else:
                    rs = {True: "retry SUCCESS", False: "retry FAILURE"}.get(row.retry_success, "retry unread")
                L.append(f"### {row.task} · seed {row.seed:04d} — revised `{row.factor}` "
                         f"({row.group}) — **{rs}**")
                L.append("")
                L.append(f"- **status:** {row.success_label}")
                if row.notes:
                    L.append(f"- _{row.notes}_")
                if row.caveat:
                    L.append(f"- **{row.caveat}**")
                for nm, fs in (("demo", row.demo), ("attempt-1", row.attempt1), ("retry", row.retry)):
                    if not fs:
                        continue
                    play = ""
                    if fs.video_link:
                        vrel = os.path.relpath(REPO / fs.video_link, gallery_root)
                        play = f" — [▶ play mp4]({vrel})"
                    L.append(f"- **{nm}** (`{os.path.basename(fs.video)}`, n={fs.n_frames}){play}")
                    if fs.strip:
                        rel = os.path.relpath(REPO / fs.strip, gallery_root)
                        L.append(f"  ![{nm}]({rel})")
                L.append("")
            elif isinstance(row, CompositeRow):
                st = ("⚠ wristcam (flag unreliable)" if row.caveat
                      else {True: "SUCCESS", False: "FAILURE"}.get(row.success, "unread"))
                L.append(f"### seed {row.seed:04d} — {row.condition} — **{st}**")
                L.append("")
                L.append(f"- {row.meta.get('condition_desc','')}")
                L.append(f"- **status:** {row.success_label}")
                if row.caveat:
                    L.append(f"- **{row.caveat}**")
                if row.fs and row.fs.video_link:
                    vrel = os.path.relpath(REPO / row.fs.video_link, gallery_root)
                    L.append(f"- [▶ play full clip mp4]({vrel})")
                if row.fs and row.fs.strip:
                    rel = os.path.relpath(REPO / row.fs.strip, gallery_root)
                    L.append(f"  ![strip]({rel})")
                L.append("")
            elif isinstance(row, IntentTableRow):
                st = {True: "SUCCESS", False: "FAILURE"}.get(row.final_success, "unread")
                L.append(f"### seed {row.seed:04d} — decoded intent ({st}) — "
                         f"VLM factor `{row.vlm_factor}`, changed "
                         f"`{row.factors_changed}`, attribution_correct={row.attribution_correct}")
                L.append("")
                L.append("| factor | ground-truth | decoded (latent) | revised | flag |")
                L.append("|---|---|---|---|---|")
                for f, g, d, r, flag in row.diff:
                    L.append(f"| {f} | {g} | {d} | {r} | {flag} |")
                L.append("")
            elif isinstance(row, SingleRow):
                L.append(f"### {row.label}")
                L.append("")
                if row.caption:
                    L.append(f"- {row.caption}")
                if row.fs and row.fs.video_link:
                    vrel = os.path.relpath(REPO / row.fs.video_link, gallery_root)
                    L.append(f"- [▶ play mp4]({vrel})")
                if row.fs and row.fs.strip:
                    rel = os.path.relpath(REPO / row.fs.strip, gallery_root)
                    L.append(f"  ![strip]({rel})")
                L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def _row_to_dict(row) -> dict:
    if isinstance(row, TripletRow):
        d = asdict(row)
        d["kind"] = "triplet"
    elif isinstance(row, CompositeRow):
        d = asdict(row)
        d["kind"] = "composite"
    elif isinstance(row, IntentTableRow):
        d = asdict(row)
        d["kind"] = "intent_table"
    elif isinstance(row, SingleRow):
        d = asdict(row)
        d["kind"] = "single"
    else:
        d = {}
    return d


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build(out_dir: Path, *, width: int, jpeg_quality: int,
          groups: Optional[set], max_seeds: Optional[int]) -> dict:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "strips").mkdir(parents=True, exist_ok=True)

    iconic_meta = load_iconic_metadata()
    stage0_report = load_stage0_pushcube_report()

    sections: list[tuple] = []
    n_videos = 0

    # Triplet groups
    for spec in TRIPLET_SPECS:
        if groups and spec["name"] not in groups:
            continue
        rows = discover_triplet_rows(spec, frames_dir, width=width,
                                     jpeg_quality=jpeg_quality, max_seeds=max_seeds,
                                     stage0_report=stage0_report)
        if rows:
            sections.append((spec["section"], rows))
            for r in rows:
                n_videos += sum(1 for fs in (r.demo, r.attempt1, r.retry) if fs)

    # Iconic composites (outcome from burned-in caption)
    iconic_seeds: list[int] = []
    if not groups or ICONIC_SPEC["name"] in groups:
        rows = discover_iconic_rows(ICONIC_SPEC, frames_dir, width=width,
                                    jpeg_quality=jpeg_quality, max_seeds=max_seeds)
        if rows:
            sections.append((ICONIC_SPEC["section"], rows))
            n_videos += sum(1 for r in rows if r.fs)
            iconic_seeds = sorted({r.seed for r in rows})

    # Measured latent-intent decode table (current P1/P2 run; no video).
    # Kept separate so the June-3 eval is never conflated with the May-24
    # iconic videos above.
    if (not groups or "intent_table" in groups) and iconic_seeds:
        trows = build_intent_table_rows(iconic_meta, seeds=iconic_seeds)
        if trows:
            sections.append((
                "Measured latent-intent decode — current P1/P2 run "
                "(June-3 eval; tabular, newer than the iconic videos above)",
                trows,
            ))

    # Single groups
    for spec in SINGLE_SPECS:
        if groups and spec["name"] not in groups:
            continue
        rows = discover_single_rows(spec, frames_dir, width=width, jpeg_quality=jpeg_quality)
        if rows:
            sections.append((spec["section"], rows))
            n_videos += sum(1 for r in rows if r.fs)

    n_rows = sum(len(rows) for _, rows in sections)
    stats = {
        "n_sections": len(sections),
        "n_rows": n_rows,
        "n_videos_framed": n_videos,
        "summary": (f"{n_videos} videos · {n_rows} rows · {len(sections)} sections · "
                    f"key frames: first / mid / late(n-5) / final · "
                    f"every success/failure label is read back from the clip's own "
                    f"burned-in caption (the rendered outcome), not assumed."),
    }

    gallery_root = out_dir
    html_doc = render_html(sections, gallery_root, stats)
    md_doc = render_markdown(sections, gallery_root, stats)
    (out_dir / "index.html").write_text(html_doc)
    (out_dir / "index.md").write_text(md_doc)

    manifest = {
        "stats": stats,
        "sections": [
            {"title": title, "rows": [_row_to_dict(r) for r in rows]}
            for title, rows in sections
        ],
    }
    (out_dir / "gallery_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reports/stage5/demo_execution_gallery",
                    help="Output gallery directory (repo-relative ok).")
    ap.add_argument("--width", type=int, default=384, help="Max frame width (px).")
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--groups", default=None,
                    help="Comma-separated subset of group names to build (default: all).")
    ap.add_argument("--max-seeds", type=int, default=None,
                    help="Cap seeds per group (smoke test).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    groups = set(g.strip() for g in args.groups.split(",")) if args.groups else None

    stats = build(out_dir, width=args.width, jpeg_quality=args.jpeg_quality,
                  groups=groups, max_seeds=args.max_seeds)
    print(f"[gallery] {stats['summary']}")
    print(f"[gallery] wrote: {_rel(out_dir / 'index.html')}")
    print(f"[gallery] wrote: {_rel(out_dir / 'index.md')}")
    print(f"[gallery] wrote: {_rel(out_dir / 'gallery_manifest.json')}")


if __name__ == "__main__":
    main()
