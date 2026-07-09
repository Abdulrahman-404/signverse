#!/usr/bin/env python3
"""
verify_dataset.py
------------------
Sanity-checks that Dataset/final_manifest.csv and every .npy file it
references actually exist on disk, with the correct non-zero size.

Run this any time after copying/moving this project folder, or right
after someone else receives it, BEFORE trying to run app.py. It has no
dependencies beyond the standard library, so it works even without the
project's requirements.txt or .venv311 installed.

Usage:
    python verify_dataset.py

Exit code 0 = dataset OK. Exit code 1 = problems found (see output).
"""
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
MANIFEST_PATH = PROJECT_ROOT / "Dataset" / "final_manifest.csv"


def main() -> int:
    if not MANIFEST_PATH.exists():
        print(f"FAIL: manifest not found at {MANIFEST_PATH}")
        print(
            "  -> Dataset/final_manifest.csv is missing entirely. If this "
            "project was copied from OneDrive, the copy may have been made "
            "before OneDrive fully downloaded the Dataset folder (Files "
            "On-Demand placeholders). Right-click Dataset/ in OneDrive and "
            "choose 'Always keep on this device', wait for the sync icon to "
            "clear, then re-copy the folder."
        )
        return 1

    with open(MANIFEST_PATH, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    words = {r["Word"] for r in rows}
    missing = []
    zero_size = []
    for r in rows:
        p = PROJECT_ROOT / r["Keypoints_Path"]
        if not p.exists():
            missing.append(r["Keypoints_Path"])
        elif p.stat().st_size == 0:
            zero_size.append(r["Keypoints_Path"])

    print(f"Manifest rows     : {len(rows)}")
    print(f"Unique words      : {len(words)}")
    print(f"Missing files     : {len(missing)}")
    print(f"Zero-byte files   : {len(zero_size)}")

    if not missing and not zero_size:
        print("\nOK: dataset is complete and consistent.")
        return 0

    print(
        "\nFAIL: dataset is incomplete. This usually means the folder was "
        "copied/synced before all files finished transferring (OneDrive "
        "Files On-Demand is the most common cause). Re-copy the whole "
        "project folder after confirming Dataset/ is fully downloaded "
        "('Always keep on this device'), then run this script again."
    )
    for path in (missing + zero_size)[:20]:
        print(f"  missing/empty: {path}")
    if len(missing) + len(zero_size) > 20:
        print(f"  ... and {len(missing) + len(zero_size) - 20} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
