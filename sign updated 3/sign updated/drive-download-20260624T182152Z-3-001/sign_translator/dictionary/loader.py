import os
import pandas as pd
import unicodedata
from pathlib import Path


class DictionaryLoader:
    def __init__(self, csv_path: str):
        self._csv_path = csv_path
        self._project_root = Path(csv_path).parent.parent

    def load(self) -> tuple[dict, set]:
        if not os.path.exists(self._csv_path):
            raise FileNotFoundError(f"Dictionary not found at {self._csv_path}")

        df = pd.read_csv(self._csv_path)
        df['Word'] = df['Word'].str.strip().apply(
            lambda x: unicodedata.normalize('NFC', str(x)))

        word_paths: dict = {}
        for word, group in df.groupby('Word'):
            word_paths[word] = group['Keypoints_Path'].tolist()

        word_set = set(word_paths.keys())

        missing = 0
        checked = 0
        for paths in word_paths.values():
            for rel_path in paths:
                abs_path = self._project_root / rel_path
                if not abs_path.exists():
                    missing += 1
                checked += 1
                if checked >= 500:
                    break
            if checked >= 500:
                break
        status = "ok" if missing == 0 else f"{missing}/{checked} missing"
        print(f"DictionaryLoader: {len(word_set)} signs loaded. ({status})")

        return word_paths, word_set

    def get_project_root(self) -> Path:
        return self._project_root
