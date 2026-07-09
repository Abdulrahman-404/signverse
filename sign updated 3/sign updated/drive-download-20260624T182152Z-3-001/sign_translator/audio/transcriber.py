import os
import glob
import shutil
import imageio_ffmpeg
from moviepy import AudioFileClip
import whisper
from pathlib import Path


class AudioTranscriber:
    def __init__(self, temp_dir: str, model_name: str = "base"):
        self._temp_dir = Path(temp_dir)
        self._temp_dir.mkdir(exist_ok=True)
        self._model_name = model_name
        self._model = None

    def _ensure_ffmpeg(self):
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = self._temp_dir / "ffmpeg"
        ffmpeg_dir.mkdir(exist_ok=True)
        ffmpeg_path = ffmpeg_dir / "ffmpeg.exe"
        if not ffmpeg_path.exists():
            shutil.copy(ffmpeg_exe, str(ffmpeg_path))
        os.environ["PATH"] = str(ffmpeg_dir.absolute()) + os.pathsep + os.environ.get("PATH", "")

    def _load_model(self):
        if self._model is None:
            print(f"   Loading Whisper model '{self._model_name}' (first time)...")
            self._model = whisper.load_model(self._model_name)
        return self._model

    def transcribe(self, audio_path: str) -> str:
        self._ensure_ffmpeg()
        clean_path = str(self._temp_dir / "clean_audio.wav")
        print("   Converting audio for Whisper...")
        audio = AudioFileClip(audio_path)
        audio.write_audiofile(
            clean_path,
            fps=16000, nbytes=2, codec='pcm_s16le',
            ffmpeg_params=["-ac", "1"],
            logger=None
        )
        audio.close()
        print(f"   Transcribing with Whisper ({self._model_name})...")
        model = self._load_model()
        result = model.transcribe(clean_path, language="ar")
        self._cleanup(clean_path, audio_path)
        return result["text"]

    def _cleanup(self, *paths):
        for f in paths:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except OSError:
                pass
