from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent.absolute()
DATASET_DIR = SCRIPT_DIR / "Dataset"
MANIFEST_PATH = str(DATASET_DIR / "final_manifest.csv")

WHISPER_MODEL = "medium"
SEQ_LEN = 30
FPS = 10  # lower = each sign holds on screen longer (was 30 -> ~1.3s/sign, now ~3.8s/sign)

_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_VIDEO = str(SCRIPT_DIR / f"translation_output_{_timestamp}.mp4")
OUTPUT_SIGML = str(SCRIPT_DIR / f"translation_output_{_timestamp}.sigml")
TRANSCRIPTION_FILE = str(SCRIPT_DIR / f"transcription_{_timestamp}.txt")
