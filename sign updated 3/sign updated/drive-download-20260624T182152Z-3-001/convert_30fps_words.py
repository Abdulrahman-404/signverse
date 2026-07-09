"""
convert_30fps_words.py
-----------------------
Converts the "sign and avatar" project's 30FPS word-motion folders (real,
temporal MediaPipe Holistic captures, external to this project) into this
project's (n_frames, 258) format and appends them to Dataset/final_manifest.csv.

Source frames are (1662,) = pose(132) + face(1404) + lh(63) + rh(63), the
same MediaPipe Holistic model this project uses but with face landmarks
included. This project's format drops face: pose(132) + lh(63) + rh(63) = 258.

Only ONE take per word (take 0) is used, not all 60 available takes.
pipeline.get_keypoints() pools every frame from every manifest path for a
word into a single sequence before interpolating down to SEQ_LEN — using
many takes would concatenate unrelated repetitions end-to-end and interpolate
across that concatenation, producing garbled cross-fades between separate
performances rather than one clean motion. A single real take avoids that.

Words without a verified Arabic gloss (checked against the eng_ara map in
the source project's Text To Sign/maps.py) are skipped rather than guessed:
'Behind' has no entry there (and its take-0 folder is empty anyway).

Usage:
    python convert_30fps_words.py
"""
import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.absolute()
DATASET_DIR = PROJECT_ROOT / "Dataset"
LABELS_DIR = DATASET_DIR / "Labels"
MANIFEST_PATH = DATASET_DIR / "final_manifest.csv"

SOURCE_ROOT = Path(r"C:\Users\w.i\OneDrive\Desktop\sign and avatar\Notebook\AllData\30FPS")
TAKE_INDEX = "0"

# Folder name -> Arabic gloss, sourced from the verified eng_ara map in
# "sign and avatar/Text To Sign/maps.py". 'Behind' has no entry there.
WORD_TO_ARABIC = {
    "Hello":        "مرحبا",
    "How are you":  "كيف حالك",
    "I":            "انا",
    "Egypt":        "في مصر",
    "Brother":      "اخ",
    "Sister":       "اخت",
    "Father":       "اب",
    "Mother":       "ام",
    "Front of":     "امام",
}

_POSE_DIM = 33 * 4   # 132
_FACE_DIM = 468 * 3  # 1404
_HAND_DIM = 21 * 3   # 63


def convert_frame_1662_to_258(arr: np.ndarray) -> np.ndarray:
    pose = arr[0:_POSE_DIM]
    lh = arr[_POSE_DIM + _FACE_DIM: _POSE_DIM + _FACE_DIM + _HAND_DIM]
    rh = arr[_POSE_DIM + _FACE_DIM + _HAND_DIM: _POSE_DIM + _FACE_DIM + 2 * _HAND_DIM]
    return np.concatenate([pose, lh, rh]).astype(np.float32)


def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def main():
    if not SOURCE_ROOT.exists():
        safe_print(f"ERROR: source not found at {SOURCE_ROOT}")
        return

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for folder_name, arabic_word in WORD_TO_ARABIC.items():
        take_dir = SOURCE_ROOT / folder_name / TAKE_INDEX
        frame_files = sorted(take_dir.glob("*.npy"), key=lambda p: int(p.stem))

        if not frame_files:
            safe_print(f"  [skip] '{folder_name}' -> no frames found in {take_dir}")
            continue

        frames = []
        for fp in frame_files:
            arr = np.load(fp)
            if arr.shape != (1662,):
                safe_print(f"    [warn] unexpected shape {arr.shape} in {fp.name}, skipping frame")
                continue
            frames.append(convert_frame_1662_to_258(arr))

        if not frames:
            safe_print(f"  [skip] '{folder_name}' -> no valid frames")
            continue

        npy_array = np.stack(frames, axis=0)
        safe_print(f"  '{folder_name}' -> {arabic_word}  shape={npy_array.shape}")

        out_dir = LABELS_DIR / arabic_word / "keypoints"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "motion_take0.npy"
        np.save(str(out_path), npy_array)

        rel_path = out_path.relative_to(PROJECT_ROOT).as_posix()
        manifest_rows.append({
            "Word": arabic_word,
            "Slide_Index": "0",
            "Frame_Count": str(npy_array.shape[0]),
            "Keypoints_Path": rel_path,
            "Raw_Image_Path": "",
        })

    # ── Append to existing manifest ──
    fieldnames = ["Word", "Slide_Index", "Frame_Count", "Keypoints_Path", "Raw_Image_Path"]
    existing_rows = []
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    all_rows = existing_rows + manifest_rows
    with open(MANIFEST_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    safe_print(f"\nWords added: {len(manifest_rows)}")
    safe_print(f"New manifest total: {len(all_rows)}")


if __name__ == "__main__":
    main()
