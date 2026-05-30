"""Stitch three-phase MP4s into side-by-side comparison videos per task.

Produces one MP4 per task: [demo | attempt | retry] with phase labels.
"""
from pathlib import Path
import cv2
import numpy as np

RENDERS = Path("/home/wang4433/scratch/babysteps/renders")
OUT_DIR = Path("/home/wang4433/scratch/babysteps/renders/comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = {
    "PushCube": RENDERS / "pushcube_bigwall/videos_maniskill/pushcube_blocked_approach_seed_0000",
    "PickCube": RENDERS / "pickcube/videos_maniskill/pickcube_grasp_slip_seed_0000",
    "StackCube": RENDERS / "stackcube/videos_maniskill/stackcube_underspec_goal_seed_0000",
    "TurnFaucet": RENDERS / "turnfaucet/videos_maniskill/turnfaucet_wrong_contact_seed_0000",
}
PHASES = [
    ("1_demo", "DEMO (third-person)"),
    ("2_attempt_blocked", "ATTEMPT (fails)"),
    ("3_retry", "RETRY (revised)"),
]
LABEL_H = 40
FPS_OUT = 20


def read_all_frames(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def pad_to_length(frames: list[np.ndarray], target: int) -> list[np.ndarray]:
    if len(frames) >= target:
        return frames[:target]
    return frames + [frames[-1]] * (target - len(frames))


def add_label(frame: np.ndarray, text: str) -> np.ndarray:
    h, w = frame.shape[:2]
    bar = np.zeros((LABEL_H, w, 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, frame])


def make_task_video(task_name: str, prefix: Path):
    phase_frames = []
    for suffix, label in PHASES:
        path = f"{prefix}__{suffix}.mp4"
        frames = read_all_frames(path)
        if not frames:
            print(f"  SKIP {path} (no frames)")
            return
        phase_frames.append((frames, label))

    max_len = max(len(fs) for fs, _ in phase_frames)

    padded = []
    for frames, label in phase_frames:
        frames = pad_to_length(frames, max_len)
        frames = [add_label(f, label) for f in frames]
        padded.append(frames)

    h, w = padded[0][0].shape[:2]
    out_path = OUT_DIR / f"{task_name}_comparison.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS_OUT, (w * 3, h),
    )
    for i in range(max_len):
        row = np.hstack([padded[j][i] for j in range(3)])
        writer.write(row)
    writer.release()
    print(f"  wrote {out_path} ({max_len} frames, {w*3}x{h})")


for task, prefix in TASKS.items():
    print(f"\n{task}:")
    make_task_video(task, prefix)

print("\nDone.")
