import os
import glob
import yt_dlp
from pathlib import Path


class AudioDownloader:
    def __init__(self, temp_dir: str):
        self._temp_dir = Path(temp_dir)
        self._temp_dir.mkdir(exist_ok=True)

    def download(self, url_or_path: str) -> tuple[str, bool]:
        if os.path.exists(url_or_path):
            print(f"   Using local file: {url_or_path}")
            return url_or_path, False
        print(f"   Downloading audio from: {url_or_path}")
        out_pattern = str(self._temp_dir / 'temp_audio_raw.webm')
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': out_pattern,
            'overwrites': True,
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url_or_path])
        raw_pattern = str(self._temp_dir / 'temp_audio_raw.*')
        raw_files = glob.glob(raw_pattern)
        if not raw_files:
            raise FileNotFoundError("Failed to download audio from YouTube.")
        return raw_files[0], True
