"""
fix_manifest_letters.py
------------------------
Patches Dataset/final_manifest.csv so the 'Word' column holds the actual
Arabic letter (e.g. "ط") instead of the internal transliterated class name
(e.g. "taa") that extract_keypoints.py writes.

Without this, DictionaryMatcher.find() can never match real Arabic text
from a transcript, because normalize_search_key() only normalises Arabic
Unicode ranges — Latin class names pass through unchanged and never equal
a real Arabic character.

Rows whose class name has no verified mapping (al, la, toot, yaa) are
dropped rather than guessed, to avoid silently mislabeling data.

Usage:
    python fix_manifest_letters.py
"""
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
MANIFEST_PATH = PROJECT_ROOT / "Dataset" / "final_manifest.csv"

# Verified transliteration -> Arabic letter map, sourced from the
# already-in-use ARABIC_LETTER_MAP in the "sign and avatar" project's
# Text To Sign/maps.py (28 entries).
TRANSLIT_TO_ARABIC = {
    'aleff': 'ا', 'bb': 'ب', 'ta': 'ت', 'thaa': 'ث',
    'jeem': 'ج', 'haa': 'ح', 'khaa': 'خ', 'dal': 'د',
    'thal': 'ذ', 'ra': 'ر', 'zay': 'ز', 'seen': 'س',
    'sheen': 'ش', 'saad': 'ص', 'dhad': 'ض', 'taa': 'ط',
    'dha': 'ظ', 'ain': 'ع', 'ghain': 'غ', 'fa': 'ف',
    'gaaf': 'ق', 'kaaf': 'ك', 'laam': 'ل', 'meem': 'م',
    'nun': 'ن', 'ha': 'ه', 'waw': 'و', 'ya': 'ي',
}

# Classes intentionally NOT mapped (unverified) — dropped, not guessed.
_UNMAPPED = {'al', 'la', 'toot', 'yaa'}


def main():
    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found at {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    kept, dropped, unknown = [], [], set()
    for row in rows:
        cls = row["Word"]
        if cls in TRANSLIT_TO_ARABIC:
            row["Word"] = TRANSLIT_TO_ARABIC[cls]
            kept.append(row)
        elif cls in _UNMAPPED:
            dropped.append(cls)
        else:
            unknown.add(cls)
            dropped.append(cls)

    with open(MANIFEST_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"Patched {len(kept)} rows -> real Arabic letters.")
    print(f"Dropped {len(dropped)} rows (unmapped classes): {sorted(set(dropped))}")
    if unknown:
        print(f"WARNING: classes seen but not in _UNMAPPED either (check manually): {sorted(unknown)}")


if __name__ == "__main__":
    main()
