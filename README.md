# Arabic Sign Language Translator

Text/audio -> Whisper -> Arabic preprocessing -> dictionary lookup ->
motion rendering -> video + SiGML, served by `app.py` (Flask).

## Before running: verify the dataset

The app depends on `Dataset/final_manifest.csv` and the `.npy` keypoint files
it points to under `Dataset/Labels/`. **Whenever you copy this project
folder, move it, or receive it from someone else, run this first:**

```
python verify_dataset.py
```

It has no dependencies (stdlib only) and checks that every keypoint file the
manifest references actually exists and is non-empty. If it reports missing
files, do not run `app.py` yet — see below.

## Why this matters (OneDrive gotcha)

This project lives under a OneDrive-synced folder. OneDrive's "Files
On-Demand" feature can leave large folders as placeholders that look present
in File Explorer but aren't fully downloaded yet. If you copy the project
folder while `Dataset/` hasn't finished downloading, the copy will silently
be missing most `.npy` files — the app won't fail until someone actually
tries to translate a word, with a confusing `FileNotFoundError` pointing at
`Dataset/final_manifest.csv` or at individual keypoint paths.

**Before copying or sending this folder to anyone:**
1. Right-click the `Dataset` folder -> "Always keep on this device".
2. Wait for the OneDrive sync icon on it to clear (fully green check, not a
   cloud icon).
3. Then copy/zip/send the folder.

**After receiving or copying this folder:** run `python verify_dataset.py`
before anything else.

## Running the app

Requires the "main" Python environment with `requirements.txt` installed
(Flask, pandas, opencv, whisper, etc.) — **not** `.venv311`, which is a
separate environment used only by the one-off data-extraction scripts
(`extract_keypoints.py`, `convert_karsl.py`, etc.) because `mediapipe`'s
legacy Holistic API requires Python <=3.11.

```
python app.py
```

Then open http://127.0.0.1:5000.

## Project layout

- `app.py`, `config.py` — Flask web app and paths/constants
- `sign_translator/` — the actual pipeline (text, dictionary, motion,
  render, sigml, audio, utils submodules)
- `Dataset/` — `final_manifest.csv` + `Labels/{word}/keypoints/*.npy`
  (required at runtime; see verification steps above)
- `data/sign_data/` — source images/labels used only by `extract_keypoints.py`
  to build part of the dataset, not needed at app runtime
- `templates/`, `static/` — main app frontend
- `Avatar/` — separate, unrelated 3D avatar viewer (Three.js + glTF) driven
  by SiGML files, not part of the main translation pipeline
- `tests/` — unit tests for pipeline components (`python tests/run_all.py`)
- `convert_*.py`, `extract_*.py`, `fix_manifest_letters.py` — one-off
  scripts used to build the dataset from external sources; not needed to
  run the app, only to regenerate/extend the dataset
