"""
extract_arasl.py
-----------------
Extracts MediaPipe keypoints from the ArASL_Database_54K_Final alphabet
image set (28 classes, ~54K images, external to this project) and APPENDS
the results to Dataset/final_manifest.csv, alongside whatever
extract_keypoints.py already produced from data/sign_data/images/.

Uses mp.solutions.holistic (back to the original approach). An mp.solutions
.hands variant was tried to fix Holistic's 88.6% zero-landmark rate, on the
theory that Hands doesn't need a pose to seed hand detection the way
Holistic does. That theory was wrong: every image in this ArASL copy is a
64x64 thumbnail (confirmed across 6 different classes) — too low-resolution
for either detector, and upscaling doesn't recover detail that was already
destroyed by the original downsize. Hands actually did WORSE on identical
images (123 frames vs Holistic's 192), so Holistic is kept as the better of
two limited options. Getting meaningfully better yield from this source
would require a higher-resolution release of ArASL, not a different
MediaPipe solution — this dataset's ceiling is resolution, not detector
choice.

Re-running this script first removes any previously-written arasl_aug.npy
files and their manifest rows, so it's safe to re-run after a method change
without accumulating stale/duplicate entries.

Unlike extract_keypoints.py, this script writes the real Arabic letter
directly as 'Word' (via the verified transliteration table also used by
fix_manifest_letters.py) — so no separate patch step is needed for this
source.

Subsamples SAMPLES_PER_CLASS images per class (ArASL has ~1,500-2,100
images per letter; using all of them is unnecessary and slow — 60 mirrors
the take-count convention already used by the "sign and avatar" project's
own 30FPS word data).

Must be run with the Python 3.11 venv (.venv311) — same mediapipe
requirement as extract_keypoints.py.

Usage:
    ./.venv311/Scripts/python.exe extract_arasl.py
"""
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

PROJECT_ROOT = Path(__file__).parent.absolute()
DATASET_DIR = PROJECT_ROOT / "Dataset"
LABELS_DIR = DATASET_DIR / "Labels"
MANIFEST_PATH = DATASET_DIR / "final_manifest.csv"

ARASL_ROOT = Path(r"C:\Users\w.i\OneDrive\Desktop\sign and avatar\Notebook\AllData\ArASL_Database_54K_Final")
ARASL_LABELS_CSV = Path(r"C:\Users\w.i\OneDrive\Desktop\sign and avatar\Notebook\AllData\ArSL_Data_Labels.csv")

SAMPLES_PER_CLASS = 60
RANDOM_SEED = 42
KEYPOINTS_FILENAME = "arasl_aug.npy"

TRANSLIT_TO_ARABIC = {
    'ain': 'ع', 'aleff': 'ا', 'bb': 'ب', 'dal': 'د', 'dha': 'ظ',
    'dhad': 'ض', 'fa': 'ف', 'gaaf': 'ق', 'ghain': 'غ', 'ha': 'ه',
    'haa': 'ح', 'jeem': 'ج', 'kaaf': 'ك', 'khaa': 'خ', 'laam': 'ل',
    'meem': 'م', 'nun': 'ن', 'ra': 'ر', 'saad': 'ص', 'seen': 'س',
    'sheen': 'ش', 'ta': 'ت', 'taa': 'ط', 'thaa': 'ث', 'thal': 'ذ',
    'waw': 'و', 'ya': 'ي', 'zay': 'ز',
}

mp_holistic = mp.solutions.holistic


def extract_keypoints_from_image(image_bgr, holistic) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic.process(image_rgb)

    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
            dtype=np.float32
        ).flatten()
    else:
        pose = np.zeros(132, dtype=np.float32)

    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32
        ).flatten()
    else:
        lh = np.zeros(63, dtype=np.float32)

    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32
        ).flatten()
    else:
        rh = np.zeros(63, dtype=np.float32)

    return np.concatenate([pose, lh, rh])


def remove_previous_arasl_output():
    """Delete stale arasl_aug.npy files and their manifest rows from a
    prior run (e.g. the earlier Holistic-based pass) so re-running this
    script doesn't accumulate duplicate/outdated entries."""
    removed_files = 0
    for npy_path in LABELS_DIR.glob(f"*/keypoints/{KEYPOINTS_FILENAME}"):
        npy_path.unlink()
        removed_files += 1

    if not MANIFEST_PATH.exists():
        return removed_files, 0

    with open(MANIFEST_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    kept = [r for r in rows if KEYPOINTS_FILENAME not in r["Keypoints_Path"]]
    removed_rows = len(rows) - len(kept)

    with open(MANIFEST_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    return removed_files, removed_rows


def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def main():
    t_start = time.time()
    random.seed(RANDOM_SEED)

    if not ARASL_ROOT.exists():
        safe_print(f"ERROR: ArASL dataset not found at {ARASL_ROOT}")
        sys.exit(1)
    if not ARASL_LABELS_CSV.exists():
        safe_print(f"ERROR: labels CSV not found at {ARASL_LABELS_CSV}")
        sys.exit(1)

    removed_files, removed_rows = remove_previous_arasl_output()
    if removed_files or removed_rows:
        safe_print(f"Removed {removed_files} stale .npy file(s) and {removed_rows} "
                    f"manifest row(s) from a previous run.")

    # ── Read labels CSV, group by class ──
    class_files = defaultdict(list)
    with open(ARASL_LABELS_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_files[row["Class"]].append(row["File_Name"])

    safe_print(f"Classes found in CSV: {len(class_files)}")

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    total_processed = 0
    total_zero = 0
    total_skipped_class = 0

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5
    ) as holistic:

        for idx, (cls, filenames) in enumerate(sorted(class_files.items())):
            arabic_word = TRANSLIT_TO_ARABIC.get(cls)
            if arabic_word is None:
                safe_print(f"  [skip class] '{cls}' has no verified Arabic mapping.")
                total_skipped_class += 1
                continue

            sample = filenames if len(filenames) <= SAMPLES_PER_CLASS else \
                random.sample(filenames, SAMPLES_PER_CLASS)

            safe_print(f"\n[{idx+1}/{len(class_files)}] '{cls}' -> {arabic_word}  "
                       f"({len(sample)}/{len(filenames)} images sampled)...")

            frames = []
            for fname in sample:
                img_path = ARASL_ROOT / cls / fname
                if not img_path.exists():
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                keypoints = extract_keypoints_from_image(img, holistic)
                if np.all(np.abs(keypoints) < 1e-6):
                    total_zero += 1
                    continue

                frames.append(keypoints)
                total_processed += 1

            if not frames:
                safe_print(f"    [WARN] No valid keypoints for '{cls}' — skipping.")
                continue

            npy_array = np.stack(frames, axis=0).astype(np.float32)
            safe_print(f"    Shape: {npy_array.shape}  |  non-zero frames: {len(frames)}")

            out_dir = LABELS_DIR / arabic_word / "keypoints"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / KEYPOINTS_FILENAME
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

    elapsed = time.time() - t_start
    safe_print(f"\n{'=' * 60}")
    safe_print(f"Done in {elapsed:.1f}s")
    safe_print(f"Classes processed : {len(manifest_rows)}")
    safe_print(f"Classes skipped   : {total_skipped_class}")
    safe_print(f"Frames extracted  : {total_processed}")
    safe_print(f"Frames all-zero   : {total_zero}")
    safe_print(f"Manifest rows added: {len(manifest_rows)}")
    safe_print(f"New manifest total: {len(all_rows)}")
    safe_print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
