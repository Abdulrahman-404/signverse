"""Phase 2/3: Convert KArSL-502 .pose → (frames, 258) .npy for overlapping words."""
import json, csv, os, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
from pose_format import Pose

# ── paths ──────────────────────────────────────────────────────────────────
KARSL_ROOT = Path(r"C:\Users\hp\Downloads\KArSL folders\content\kars502-pose-75k")
POSE_DIR = KARSL_ROOT / "pose_dir"
TRAIN_SPLIT = KARSL_ROOT / "splits" / "train_70.txt"
OVERLAP_PATH = Path(r"C:\Users\hp\AppData\Local\Temp\opencode\overlap_mapping.json")

PROJECT_ROOT = Path(__file__).parent.absolute()
DATASET_DIR = PROJECT_ROOT / "Dataset"
MANIFEST_PATH = DATASET_DIR / "final_manifest.csv"
IMG_W = 1000  # from KArSL PoseHeader dimensions
IMG_H = 1000


def convert_one_pose(pose_path: Path) -> np.ndarray:
    """Load a .pose file and return (frames, 258) normalized array."""
    with open(pose_path, "rb") as f:
        buf = f.read()
    pose = Pose.read(buf)

    data = pose.body.data   # (F, 1, 543, 3) – MaskedArray
    conf = pose.body.confidence  # (F, 1, 543) – MaskedArray

    # Fill masked entries with 0
    data_filled = np.ma.filled(data, 0.0).astype(np.float32)
    conf_filled = np.ma.filled(conf, 0.0).astype(np.float32)

    # ── pose landmarks [0:33] → interleave xyz + conf ──
    p_data = data_filled[:, 0, 0:33, :]   # (F, 33, 3)
    p_conf = conf_filled[:, 0, 0:33, None]  # (F, 33, 1)

    p_data[:, :, 0] /= IMG_W  # x
    p_data[:, :, 1] /= IMG_H  # y

    pose_out = np.concatenate([p_data, p_conf], axis=-1)  # (F, 33, 4)
    pose_flat = pose_out.reshape(pose_out.shape[0], -1)   # (F, 132)

    # ── left hand [501:522] ──
    lh = data_filled[:, 0, 501:522, :]   # (F, 21, 3)
    lh[:, :, 0] /= IMG_W
    lh[:, :, 1] /= IMG_H
    lh_flat = lh.reshape(lh.shape[0], -1)  # (F, 63)

    # ── right hand [522:543] ──
    rh = data_filled[:, 0, 522:543, :]   # (F, 21, 3)
    rh[:, :, 0] /= IMG_W
    rh[:, :, 1] /= IMG_H
    rh_flat = rh.reshape(rh.shape[0], -1)  # (F, 63)

    result = np.concatenate([pose_flat, lh_flat, rh_flat], axis=1)  # (F, 258)
    return result.astype(np.float32)


def safe_print(msg: str):
    """Print with fallback for Unicode chars unsupported by the console."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def main():
    # ── 1. Load overlap mapping ──
    with open(OVERLAP_PATH, encoding="utf-8") as f:
        overlap = json.load(f)  # {current_word: karsl_sign_id_str}

    # Build reverse map: zero-padded sign_id → list of current words
    sign_to_words = defaultdict(list)
    for word, sid in overlap.items():
        sign_to_words[sid.zfill(4)].append(word)

    # ── 2. Parse training split → collect .pose paths per sign_id ──
    sign_train_poses = defaultdict(list)
    with open(TRAIN_SPLIT) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # path: pose_dir/sign_0195_s1_sess1_001.pose
            parts = line.split("/")
            fname = parts[-1]
            sign_id = fname.split("_")[1]  # e.g. "0195"
            sign_train_poses[sign_id].append(POSE_DIR / fname)

    # Keep only overlapping sign_ids
    overlap_sign_ids = set(sign_to_words.keys())
    available = overlap_sign_ids & set(sign_train_poses.keys())
    safe_print(f"Overlap sign IDs with training data: {len(available)} / {len(overlap_sign_ids)}")

    # ── 3. Process each word ──
    manifest_path = MANIFEST_PATH
    new_manifest_rows = []

    total_words = len(overlap)
    processed = 0
    skipped = 0
    total_frames = 0
    t_start = time.time()

    for word, sid in sorted(overlap.items(), key=lambda x: x[0]):
        sign_id_pad = sid.zfill(4)
        pose_files = sign_train_poses.get(sign_id_pad, [])

        if not pose_files:
            skipped += 1
            continue

        word_dir = DATASET_DIR / "Labels" / word / "keypoints"
        word_dir.mkdir(parents=True, exist_ok=True)
        out_path = word_dir / "karsl_aug.npy"

        # Convert all .pose files for this word
        frames_list = []
        for pf in pose_files:
            try:
                arr = convert_one_pose(pf)
                if arr.shape[0] > 0:
                    frames_list.append(arr)
            except Exception as e:
                safe_print(f"  ! {word}: error {pf.name}: {e}")

        if not frames_list:
            skipped += 1
            continue

        # Concatenate all samples into one big (N, 258) array
        big_arr = np.concatenate(frames_list, axis=0)
        np.save(out_path, big_arr)
        total_frames += big_arr.shape[0]

        # Build relative path for manifest
        rel_path = os.path.relpath(out_path, PROJECT_ROOT).replace("\\", "/")

        # Add row to manifest: word, slide_index=0, frame_count=num_frames (multi), path
        new_manifest_rows.append({
            "Word": word,
            "Slide_Index": "0",
            "Frame_Count": str(big_arr.shape[0]),
            "Keypoints_Path": rel_path,
            "Raw_Image_Path": ""
        })

        processed += 1
        if processed % 50 == 0:
            elapsed = time.time() - t_start
            safe_print(f"  [{processed}/{total_words}] ID={sid} frames={big_arr.shape[0]} "
                       f"{elapsed:.0f}s elapsed")

    # ── 4. Append new rows to manifest CSV ──
    if new_manifest_rows:
        fieldnames = ["Word", "Slide_Index", "Frame_Count", "Keypoints_Path", "Raw_Image_Path"]
        # Read existing rows
        existing_rows = []
        with open(manifest_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)

        # Append new rows
        all_rows = existing_rows + new_manifest_rows

        with open(manifest_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    # ── summary ──
    elapsed = time.time() - t_start
    safe_print(f"\n{'=' * 50}")
    safe_print(f"Done in {elapsed:.1f}s")
    safe_print(f"Words processed: {processed}")
    safe_print(f"Words skipped:   {skipped}")
    safe_print(f"Total frames:    {total_frames}")
    safe_print(f"Manifest rows added: {len(new_manifest_rows)}")
    safe_print(f"New manifest total: {len(existing_rows) + len(new_manifest_rows)}")


if __name__ == "__main__":
    main()
