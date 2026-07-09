"""
extract_keypoints.py
---------------------
Reads all images from data/sign_data/images/, extracts MediaPipe Holistic
keypoints (258-dim per frame), groups by sign class, saves each sign as a
(N, 258) .npy file under Dataset/Labels/{sign}/keypoints/data.npy, and
writes Dataset/final_manifest.csv.

Usage:
    python extract_keypoints.py

Expected image filename format:
    {person_id}_{age}_{gender}_{sign}_{frame}.jpg
    e.g. 1001_46_F_ain_0.jpg  →  sign = ain
"""

import os
import sys
import csv
import time
import re
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.absolute()
IMAGES_DIR   = PROJECT_ROOT / "data" / "sign_data" / "images"
DATASET_DIR  = PROJECT_ROOT / "Dataset"
LABELS_DIR   = DATASET_DIR / "Labels"
MANIFEST_PATH = DATASET_DIR / "final_manifest.csv"

# Mediapipe image dimensions (used for normalisation)
# Images are NOT resized — we pass them as-is, MP handles normalisation internally.

# ── MediaPipe setup ─────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic


def extract_keypoints_from_image(image_bgr, holistic) -> np.ndarray:
    """
    Run MediaPipe Holistic on a BGR image.
    Returns a (258,) float32 array:
        pose      : 33 landmarks × 4 (x, y, z, visibility) = 132
        left hand : 21 landmarks × 3 (x, y, z)             =  63
        right hand: 21 landmarks × 3 (x, y, z)             =  63
    Invisible/missing landmarks are returned as zeros.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic.process(image_rgb)

    # ── pose (33 × 4) ──
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
            dtype=np.float32
        ).flatten()  # (132,)
    else:
        pose = np.zeros(132, dtype=np.float32)

    # ── left hand (21 × 3) ──
    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32
        ).flatten()  # (63,)
    else:
        lh = np.zeros(63, dtype=np.float32)

    # ── right hand (21 × 3) ──
    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32
        ).flatten()  # (63,)
    else:
        rh = np.zeros(63, dtype=np.float32)

    return np.concatenate([pose, lh, rh])  # (258,)


def parse_sign_from_filename(filename: str):
    """
    Extract sign label from filename like:
        1001_46_F_ain_0.jpg  →  'ain'
        229_20_M_dhad_5.jpg  →  'dhad'

    The sign is the 4th underscore-separated token (0-indexed: token[3]),
    everything up to (but not including) the last numeric token.
    Handles multi-part sign names like 'aleff', 'ghain', etc.
    """
    stem = Path(filename).stem  # e.g. "1001_46_F_ain_0"
    parts = stem.split("_")
    # parts = [person_id, age, gender, sign..., frame_idx]
    # The last part is always an integer (frame index).
    # Everything from index 3 to -1 is the sign name.
    if len(parts) < 5:
        return None
    sign_parts = parts[3:-1]  # everything between gender and frame
    return "_".join(sign_parts)


def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def main():
    t_start = time.time()

    # ── gather all images and group by sign ──
    safe_print(f"Scanning images in: {IMAGES_DIR}")
    all_images = sorted(IMAGES_DIR.glob("*.jpg"))
    safe_print(f"Found {len(all_images)} images")

    if not all_images:
        safe_print("ERROR: No .jpg images found. Check the path.")
        sys.exit(1)

    # Group image paths by sign class
    sign_images: dict[str, list[Path]] = defaultdict(list)
    skipped_parse = 0
    for img_path in all_images:
        sign = parse_sign_from_filename(img_path.name)
        if sign is None:
            skipped_parse += 1
            continue
        sign_images[sign].append(img_path)

    safe_print(f"Unique signs found: {len(sign_images)}")
    safe_print(f"Images with unparseable names: {skipped_parse}")
    if sign_images:
        for s, imgs in sorted(sign_images.items()):
            safe_print(f"  {s:20s}: {len(imgs)} images")

    # ── process each sign ──
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    total_processed = 0
    total_skipped = 0
    total_zero = 0

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5
    ) as holistic:

        for sign_idx, (sign, img_paths) in enumerate(sorted(sign_images.items())):
            safe_print(f"\n[{sign_idx+1}/{len(sign_images)}] Processing '{sign}' ({len(img_paths)} images)...")

            frames = []
            img_paths_used = []

            for img_path in img_paths:
                img = cv2.imread(str(img_path))
                if img is None:
                    safe_print(f"    [skip] Could not read: {img_path.name}")
                    total_skipped += 1
                    continue

                keypoints = extract_keypoints_from_image(img, holistic)

                # Skip all-zero frames (no body/hand detected)
                if np.all(np.abs(keypoints) < 1e-6):
                    safe_print(f"    [zero] No landmarks: {img_path.name}")
                    total_zero += 1
                    continue

                frames.append(keypoints)
                img_paths_used.append(img_path)
                total_processed += 1

            if not frames:
                safe_print(f"    [WARN] No valid keypoints for '{sign}' — skipping.")
                continue

            # Stack all frames into (N, 258)
            npy_array = np.stack(frames, axis=0).astype(np.float32)
            safe_print(f"    Shape: {npy_array.shape}  |  non-zero frames: {len(frames)}")

            # Save .npy file
            out_dir = LABELS_DIR / sign / "keypoints"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "data.npy"
            np.save(str(out_path), npy_array)

            # Build relative path for manifest
            rel_path = out_path.relative_to(PROJECT_ROOT).as_posix()

            manifest_rows.append({
                "Word": sign,
                "Slide_Index": "0",
                "Frame_Count": str(npy_array.shape[0]),
                "Keypoints_Path": rel_path,
                "Raw_Image_Path": ""
            })

    # ── write manifest CSV ──
    fieldnames = ["Word", "Slide_Index", "Frame_Count", "Keypoints_Path", "Raw_Image_Path"]
    with open(MANIFEST_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    elapsed = time.time() - t_start
    safe_print(f"\n{'=' * 60}")
    safe_print(f"Done in {elapsed:.1f}s")
    safe_print(f"Signs processed  : {len(manifest_rows)}")
    safe_print(f"Frames extracted : {total_processed}")
    safe_print(f"Frames all-zero  : {total_zero}")
    safe_print(f"Images unreadable: {total_skipped}")
    safe_print(f"Manifest written : {MANIFEST_PATH}")
    safe_print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
