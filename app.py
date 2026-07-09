# app.py
# ── Flask web API for the Arabic Sign Language pipeline ──────────────────────
#
# Endpoints:
#   GET  /                              → serves templates/index.html
#   POST /api/translate                 → full pipeline: text/audio -> video+SiGML+breakdown
#   GET  /outputs/<session_id>/<file>   → serves per-request generated files
#
# Run with the MAIN Python environment (not .venv311) — this app never needs
# mediapipe, only the already-extracted keypoints in Dataset/Labels/.
#
#   python app.py
#
import logging
import os
import threading
import uuid
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory

from config import FPS, MANIFEST_PATH, SCRIPT_DIR, SEQ_LEN
from sign_translator.dictionary.loader import DictionaryLoader
from sign_translator.dictionary.matcher import DictionaryMatcher
from sign_translator.dictionary.phrase_matcher import PhraseMatcher
from sign_translator.pipeline import Pipeline
from sign_translator.render.renderer import LandmarkRenderer
from sign_translator.sigml.encoder import SiGMLEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
WEB_OUTPUT_DIR = SCRIPT_DIR / "web_output"
WEB_OUTPUT_DIR.mkdir(exist_ok=True)

# Video containers work unchanged through the existing transcription code —
# AudioTranscriber.transcribe() uses moviepy's AudioFileClip, which pulls the
# audio track directly out of a video file via ffmpeg (verified: works on a
# real muxed .mp4 with no code changes needed).
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac", ".aac", ".opus"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
ALLOWED_MEDIA_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS
MAX_AUDIO_BYTES = 150 * 1024 * 1024  # 150 MB — raised from 50MB to accommodate video uploads
ALLOWED_OUTPUT_FILENAMES = {"output.mp4", "output.sigml"}

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_AUDIO_BYTES + 1024 * 1024  # small headroom for form fields

# ── Load dictionary once at startup (not per-request) ─────────────────────────
logger.info("Loading dictionary...")
_loader = DictionaryLoader(MANIFEST_PATH)
_word_paths, _word_set = _loader.load()
_matcher = DictionaryMatcher(_word_paths, _word_set, str(_loader.get_project_root()))
_phrase_matcher = PhraseMatcher(_word_paths, _word_set, _matcher)
logger.info("Dictionary ready: %d unique signs.", len(_word_set))

# ── One shared Pipeline instance (so the Whisper model loads once, not per-request) ──
pipeline = Pipeline()

# ── Warm up Whisper at startup so the first real request isn't slow ───────────
logger.info("Warming up Whisper model...")
try:
    pipeline._transcriber._load_model()
    logger.info("Whisper model ready.")
except Exception:
    logger.exception("Whisper warm-up failed — transcription requests will likely fail too.")

# AudioDownloader/AudioTranscriber write to FIXED filenames under SCRIPT_DIR
# regardless of caller (temp_audio_raw.*, clean_audio.wav) — concurrent
# transcribe() calls would race on those files. Serialize at this layer
# rather than touching sign_translator's working, tested code.
_transcribe_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _error(message: str, status: int):
    return jsonify({"status": "error", "error": message}), status


def _validate_audio_upload(file_storage) -> tuple[str | None, str | None]:
    """Validate an uploaded audio/video file's extension.
    Returns (extension, None) on success or (None, error_message) on failure."""
    original_name = file_storage.filename or ""
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_MEDIA_EXTENSIONS:
        return None, (
            f"نوع الملف '{ext}' غير مدعوم. الأنواع المدعومة: "
            f"{', '.join(sorted(ALLOWED_MEDIA_EXTENSIONS))}"
        )
    return ext, None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/translate", methods=["POST"])
def translate():
    text_input = (request.form.get("text") or "").strip()
    audio_file = request.files.get("audio")
    has_audio = audio_file is not None and audio_file.filename

    provided = sum([bool(text_input), bool(has_audio)])
    if provided == 0:
        return _error("يرجى إدخال نص أو رفع ملف صوتي.", 400)
    if provided > 1:
        return _error("يرجى استخدام مُدخَل واحد فقط (نص أو ملف صوتي).", 400)

    session_id = str(uuid.uuid4())
    session_dir = WEB_OUTPUT_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    transcript = None
    warnings: list[str] = []

    try:
        # ── 1. Resolve raw text ────────────────────────────────────────────
        if text_input:
            raw_text = text_input
        else:
            ext, err = _validate_audio_upload(audio_file)
            if err:
                return _error(err, 400)
            saved_path = str(session_dir / f"upload{ext}")
            audio_file.save(saved_path)
            if os.path.getsize(saved_path) > MAX_AUDIO_BYTES:
                return _error(
                    f"حجم الملف يتجاوز الحد الأقصى ({MAX_AUDIO_BYTES // (1024 * 1024)} ميجابايت).",
                    400,
                )
            try:
                with _transcribe_lock:
                    raw_text = pipeline.transcribe(saved_path)
            except Exception:
                logger.exception("Transcription failed for uploaded audio")
                return _error("تعذر تحويل الملف الصوتي إلى نص.", 500)
            transcript = raw_text

        # ── 2. Preprocess ────────────────────────────────────────────────────
        clean_text = pipeline.preprocess(raw_text)
        if not clean_text.strip():
            return _error("النص فارغ بعد المعالجة، لا توجد كلمات لترجمتها.", 400)

        # ── 3. Map to signs (both the flat sequence for rendering, and the
        #      grouped breakdown for the UI) ──────────────────────────────────
        word_sequence = pipeline.map_signs(clean_text, _matcher, _phrase_matcher)
        breakdown = pipeline.map_signs_grouped(clean_text, _matcher, _phrase_matcher)

        if not word_sequence:
            return _error("لم يتم التعرف على أي كلمات قابلة للترجمة.", 400)

        # ── 4. Render video + build SiGML ───────────────────────────────────
        renderer = LandmarkRenderer(fps=FPS)
        sigml_encoder = SiGMLEncoder(fps=FPS)

        video_path = str(session_dir / "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, FPS, (renderer.w, renderer.h))

        total_frames = 0
        dictionary_hits = 0

        for display_word, lookup_key in word_sequence:
            rendered_real_sign = False
            if lookup_key is not None:
                landmarks = pipeline.get_keypoints(lookup_key, _matcher, target_len=SEQ_LEN)
                if landmarks is not None:
                    frames = renderer.render_word_frames(landmarks, display_word, transition_frames=8)
                    sigml_encoder.add_sign(display_word, landmarks)
                    rendered_real_sign = True
                else:
                    frames = renderer.render_unknown_word(display_word, num_frames=SEQ_LEN)
                    sigml_encoder.add_unknown(display_word)
            else:
                frames = renderer.render_unknown_word(display_word, num_frames=SEQ_LEN)
                sigml_encoder.add_unknown(display_word)

            if rendered_real_sign:
                dictionary_hits += 1
            for frame in frames:
                writer.write(frame)
            total_frames += len(frames)

        writer.release()

        sigml_path = str(session_dir / "output.sigml")
        sigml_encoder.save(sigml_path)

        if dictionary_hits == 0:
            warnings.append("لم يتم العثور على أي كلمة في القاموس، تم عرض حروف/إشارات بديلة فقط.")

        return jsonify({
            "status": "ok",
            "session_id": session_id,
            "video_url": f"/outputs/{session_id}/output.mp4",
            "sigml_url": f"/outputs/{session_id}/output.sigml",
            "transcript": transcript,
            "clean_text": clean_text,
            "stats": {
                "total_units": len(word_sequence),
                "dictionary_hits": dictionary_hits,
                "video_frames": total_frames,
            },
            "breakdown": breakdown,
            "warnings": warnings,
        })

    except Exception:
        logger.exception("Unexpected error in /api/translate")
        return _error("حدث خطأ غير متوقع أثناء الترجمة.", 500)


@app.route("/outputs/<session_id>/<filename>")
def serve_output(session_id: str, filename: str):
    try:
        uuid.UUID(session_id)
    except ValueError:
        return jsonify({"error": "Invalid session ID"}), 400
    if filename not in ALLOWED_OUTPUT_FILENAMES:
        return jsonify({"error": "Invalid filename"}), 400
    return send_from_directory(str(WEB_OUTPUT_DIR / session_id), filename)


# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n>>> Open your browser at  http://127.0.0.1:5000\n")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
