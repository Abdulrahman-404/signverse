"""
convert_karsl_zip.py
---------------------
Extracts MediaPipe Holistic keypoints from the local KArSL-502 archive
(raw frame-image sequences, NOT the .pose format convert_karsl.py expects)
for the 141 sign IDs that have a verified Arabic word label
(karsl_labels.py), and appends the results to Dataset/final_manifest.csv.

Reads frames directly out of the zip via zipfile — the archive is 23.7GB,
so nothing is extracted to disk; each frame is decompressed to memory,
decoded, run through MediaPipe, and discarded.

Zip layout confirmed by inspection:
    {signer:01-03}/{signer}/{split:train|test}/{sign_id:0001-0502}/{session}/{session}_{frame:04d}.jpg
For each sign_id we use ONE session (prefer signer 01, split train, the
session with the most frames) — using multiple sessions would pool separate
real performances into one sequence and interpolate across the seam, the
same reasoning applied in convert_30fps_words.py.

Arabic words are normalized (normalize_arabic) before being written as
'Word', so spelling variants (e.g. 'أب' vs the existing 'اب' from
convert_30fps_words.py) merge into the same dictionary entry instead of
becoming an orphaned duplicate — DictionaryLoader groups by the literal
'Word' string, not its normalized form.

Must be run with the Python 3.11 venv (.venv311) — same mediapipe
requirement as extract_keypoints.py.

Usage:
    ./.venv311/Scripts/python.exe convert_karsl_zip.py
"""
import csv
import re
import sys
import time
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).parent))
from karsl_labels import KARSL_WORD_TO_SIGN_ID

# Inlined from sign_translator/text/normalizer.py's normalize_arabic() —
# importing that module in .venv311 would pull in the full sign_translator
# package __init__ chain (pandas, arabic_reshaper, cv2's Arabic text
# rendering deps, ...) that this extraction venv doesn't have and doesn't
# need. Keep this in sync with normalizer.py if that logic ever changes.
_TASHKEEL = r'[ً-ْ]'
_KASHIDA = r'[ـ]'
_NORMALIZATION_MAP = {
    ord('أ'): 'ا', ord('إ'): 'ا', ord('آ'): 'ا', ord('ٱ'): 'ا',
    ord('ؤ'): 'و', ord('ئ'): 'ي', ord('ة'): 'ه', ord('ى'): 'ي',
}


def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(_KASHIDA, '', text)
    text = re.sub(_TASHKEEL, '', text)
    text = text.translate(_NORMALIZATION_MAP)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

PROJECT_ROOT = Path(__file__).parent.absolute()
DATASET_DIR = PROJECT_ROOT / "Dataset"
LABELS_DIR = DATASET_DIR / "Labels"
MANIFEST_PATH = DATASET_DIR / "final_manifest.csv"

ZIP_PATH = Path(r"C:\Users\w.i\OneDrive\Desktop\sign zift\archive.zip")
PREFERRED_SIGNER = "01"
PREFERRED_SPLIT = "train"
MAX_FRAMES_PER_SIGN = 60
KEYPOINTS_FILENAME = "karsl_aug.npy"

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


def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def remove_previous_karsl_output():
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


def build_sign_session_index(zf: zipfile.ZipFile, needed_sign_ids: set):
    """Single pass over the zip's namelist -> {sign_id: {session_path: [frame_names...]}}
    restricted to the preferred signer/split, for only the sign IDs we need."""
    index = defaultdict(lambda: defaultdict(list))
    prefix = f"{PREFERRED_SIGNER}/{PREFERRED_SIGNER}/{PREFERRED_SPLIT}/"
    for name in zf.namelist():
        if not name.startswith(prefix) or not name.endswith(".jpg"):
            continue
        rest = name[len(prefix):]
        parts = rest.split("/")
        if len(parts) != 3:
            continue
        sign_id, session, _fname = parts
        if sign_id not in needed_sign_ids:
            continue
        index[sign_id][session].append(name)
    return index


def main():
    t_start = time.time()

    if not ZIP_PATH.exists():
        safe_print(f"ERROR: KArSL zip not found at {ZIP_PATH}")
        sys.exit(1)

    removed_files, removed_rows = remove_previous_karsl_output()
    if removed_files or removed_rows:
        safe_print(f"Removed {removed_files} stale .npy file(s) and {removed_rows} "
                    f"manifest row(s) from a previous run.")

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    needed_sign_ids = set(KARSL_WORD_TO_SIGN_ID.values())
    safe_print(f"Indexing zip for {len(needed_sign_ids)} needed sign IDs "
               f"(signer {PREFERRED_SIGNER}, split {PREFERRED_SPLIT})...")

    manifest_rows = []
    total_processed = 0
    total_zero = 0
    total_missing_sign = 0

    with zipfile.ZipFile(ZIP_PATH) as zf:
        sign_index = build_sign_session_index(zf, needed_sign_ids)
        safe_print(f"Found {len(sign_index)}/{len(needed_sign_ids)} sign IDs in the archive.\n")

        with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5
        ) as holistic:

            for idx, (word, sign_id) in enumerate(sorted(KARSL_WORD_TO_SIGN_ID.items(), key=lambda kv: kv[1])):
                # Normalize multi-character words so spelling variants merge
                # with existing entries (e.g. 'أب' -> 'اب' to match the
                # existing "father" entry from convert_30fps_words.py) — but
                # NOT single letters. normalize_arabic() maps 'ة' -> 'ه',
                # which is correct for fuzzy word matching but would merge
                # KArSL's distinct recorded sign for 'ة' into 'ه', silently
                # losing a real letter (and mis-fingerspelling every word
                # ending in 'ة', extremely common in Arabic).
                arabic_word = word if len(word) == 1 else normalize_arabic(word)
                sessions = sign_index.get(sign_id)
                if not sessions:
                    safe_print(f"[{idx+1}/{len(KARSL_WORD_TO_SIGN_ID)}] sign {sign_id} ('{word}') "
                               f"-> NOT FOUND in archive, skipping.")
                    total_missing_sign += 1
                    continue

                # pick the session with the most frames
                best_session, frame_names = max(sessions.items(), key=lambda kv: len(kv[1]))
                frame_names = sorted(frame_names)[:MAX_FRAMES_PER_SIGN]

                safe_print(f"[{idx+1}/{len(KARSL_WORD_TO_SIGN_ID)}] sign {sign_id} -> {word}  "
                           f"({len(frame_names)} frames from {best_session})...")

                frames = []
                for name in frame_names:
                    data = zf.read(name)
                    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    keypoints = extract_keypoints_from_image(img, holistic)
                    if np.all(np.abs(keypoints) < 1e-6):
                        total_zero += 1
                        continue
                    frames.append(keypoints)
                    total_processed += 1

                if not frames:
                    safe_print(f"    [WARN] No valid keypoints for sign {sign_id} — skipping.")
                    continue

                npy_array = np.stack(frames, axis=0).astype(np.float32)

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
    safe_print(f"Signs processed    : {len(manifest_rows)}")
    safe_print(f"Signs missing in zip: {total_missing_sign}")
    safe_print(f"Frames extracted   : {total_processed}")
    safe_print(f"Frames all-zero    : {total_zero}")
    safe_print(f"Manifest rows added: {len(manifest_rows)}")
    safe_print(f"New manifest total : {len(all_rows)}")
    safe_print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
