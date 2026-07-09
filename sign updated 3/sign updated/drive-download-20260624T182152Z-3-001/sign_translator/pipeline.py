import numpy as np
import os
from pathlib import Path

from .text.preprocessor import ArabicPreProcessor
from .text.segmenter import SentenceSegmenter
from .text.normalizer import is_arabic_char
from .dictionary.loader import DictionaryLoader
from .dictionary.matcher import DictionaryMatcher
from .dictionary.phrase_matcher import PhraseMatcher
from .motion.interpolator import MotionInterpolator
from .motion.transitions import SmoothTransitions
from .motion.smoother import TemporalSmoother
from .render.renderer import LandmarkRenderer
from .sigml.encoder import SiGMLEncoder
from .audio.downloader import AudioDownloader
from .audio.transcriber import AudioTranscriber
from .utils.caching import NPYCache

from config import (MANIFEST_PATH, WHISPER_MODEL, SEQ_LEN, FPS,
                    OUTPUT_VIDEO, OUTPUT_SIGML, TRANSCRIPTION_FILE, SCRIPT_DIR)

# get_keypoints() picks exactly one of these per word rather than pooling
# frames across all of them — concatenating independent sources and cubic-
# spline interpolating across the whole thing produces overshoot artifacts
# at every boundary between unrelated frames (confirmed via audit: 83/144
# signs had frames where the hand disconnects from the body, concentrated
# almost entirely in the 28/36 letters that pool multiple sources).
# Real video (temporally coherent) ranks above static-photo stacks
# (independent captures from different people/sessions, no temporal
# relationship to each other).
_SOURCE_PRIORITY = ["motion_take0.npy", "karsl_aug.npy", "data.npy", "arasl_aug.npy"]
_PHOTO_STACK_FILENAMES = {"data.npy", "arasl_aug.npy"}

_LHAND_SLICE = slice(132, 195)
_RHAND_SLICE = slice(195, 258)


def _fill_missing_hand_frames(src: np.ndarray) -> np.ndarray:
    """Real recordings occasionally have a frame where MediaPipe briefly
    lost tracking of one hand (fast motion blur, momentary occlusion) —
    that frame's hand landmarks come back as (0,0,0), a real value, not a
    sentinel for "missing". TemporalSmoother's sliding window doesn't know
    that, so it blends a real neighbouring frame's hand position toward
    that zero, corrupting frames next to the dropout (most visible right at
    a word's first frame, where the smoothing window is already lopsided).
    Replace each dropout frame's hand data with the nearest frame (by
    index) where that hand WAS detected, before interpolation/smoothing
    ever sees it. A hand that's missing in every frame is left untouched —
    nothing valid to fill from, and is_valid_hand() already treats an
    all-zero hand as "don't draw" correctly."""
    src = src.copy()
    for hand_slice in (_LHAND_SLICE, _RHAND_SLICE):
        sub = src[:, hand_slice]
        valid = ~np.all(np.abs(sub) < 1e-6, axis=1)
        if not valid.any() or valid.all():
            continue
        valid_idx = np.where(valid)[0]
        for i in np.where(~valid)[0]:
            nearest = valid_idx[np.argmin(np.abs(valid_idx - i))]
            src[i, hand_slice] = src[nearest, hand_slice]
    return src


class Pipeline:
    """End-to-end pipeline: YouTube → Whisper → Arabic → Dictionary → Motion → Render → SiGML"""

    def __init__(self):
        self._npy_cache = NPYCache(maxsize=200)
        self._interpolator = MotionInterpolator(method='cubic')
        self._transitions = SmoothTransitions(transition_frames=10, hold_frames=5)
        self._smoother = TemporalSmoother(window=5, order=2)
        self._downloader = AudioDownloader(str(SCRIPT_DIR))
        self._transcriber = AudioTranscriber(str(SCRIPT_DIR), model_name=WHISPER_MODEL)
        self._segmenter = SentenceSegmenter()

    def transcribe(self, url_or_path: str) -> str:
        audio_path, was_downloaded = self._downloader.download(url_or_path)
        text = self._transcriber.transcribe(audio_path)
        return text

    def preprocess(self, raw_text: str) -> str:
        return ArabicPreProcessor.full_pipeline(raw_text)

    def map_signs(self, clean_text: str, mapper, phrase_matcher):
        tokens = clean_text.split()
        raw_phrases = phrase_matcher.match_phrases(tokens)
        sequence = []
        for display_word, lookup_key in raw_phrases:
            sequence.append((display_word, lookup_key))
        if not sequence:
            return []
        found = sum(1 for _, k in sequence if k is not None)
        if found >= len(sequence) / 2 or len(sequence) <= 1:
            return self._apply_fingerspelling_fallback(sequence, mapper)
        greedy_seq = self._greedy_keypoint_sequence(clean_text, mapper)
        greedy_found = sum(1 for _, k in greedy_seq if k is not None)
        chosen = greedy_seq if greedy_found > found else sequence
        return self._apply_fingerspelling_fallback(chosen, mapper)

    def map_signs_grouped(self, clean_text: str, mapper, phrase_matcher) -> list:
        """Same phrase-vs-greedy selection as map_signs, but keeps
        fingerspelled words grouped under their original word instead of
        flattening into individual letter units. Used for a per-word
        breakdown (e.g. a web UI) — map_signs itself is unchanged and stays
        the source of truth for the actual render/SiGML sequence."""
        tokens = clean_text.split()
        raw_phrases = phrase_matcher.match_phrases(tokens)
        sequence = [(w, k) for w, k in raw_phrases]
        if not sequence:
            return []
        found = sum(1 for _, k in sequence if k is not None)
        if found >= len(sequence) / 2 or len(sequence) <= 1:
            chosen = sequence
        else:
            greedy_seq = self._greedy_keypoint_sequence(clean_text, mapper)
            greedy_found = sum(1 for _, k in greedy_seq if k is not None)
            chosen = greedy_seq if greedy_found > found else sequence

        grouped = []
        for display_word, lookup_key in chosen:
            if lookup_key is not None:
                grouped.append({"word": display_word, "matched": True, "letters": None})
            else:
                letters = self._fingerspell_word(display_word, mapper)
                grouped.append({
                    "word": display_word,
                    "matched": False,
                    "letters": [{"letter": l, "matched": k is not None} for l, k in letters],
                })
        return grouped

    def _fingerspell_word(self, word: str, mapper) -> list:
        """Decompose an out-of-vocabulary word into individual Arabic
        letters and look each one up. A letter with no dictionary entry
        still gets a (letter, None) placeholder rather than being dropped,
        so at least the other letters in the word still render."""
        letters = [ch for ch in word if is_arabic_char(ch)]
        if not letters:
            return [(word, None)]
        return [(letter, mapper.find(letter)) for letter in letters]

    def _apply_fingerspelling_fallback(self, sequence, mapper) -> list:
        expanded = []
        for display_word, lookup_key in sequence:
            if lookup_key is not None:
                expanded.append((display_word, lookup_key))
            else:
                expanded.extend(self._fingerspell_word(display_word, mapper))
        return expanded

    def _greedy_keypoint_sequence(self, text, mapper):
        tokens = text.split()
        text_no_spaces = ''.join(tokens)
        return self._greedy_match(text_no_spaces, mapper)

    def _greedy_match(self, text, mapper):
        i = 0
        sequence = []
        while i < len(text):
            found = False
            for length in range(min(15, len(text) - i), 1, -1):
                sub = text[i:i + length]
                # find_exact only, not the full find() cascade: this loop
                # already tries every substring window of the run-on text,
                # so with 100+ dictionary entries, allowing fuzzy/substring
                # matching per window turns into a near-guaranteed source of
                # coincidental false positives (random letter-run overlaps
                # with unrelated words) rather than real matches.
                match = mapper.find_exact(sub)
                if match:
                    sequence.append((match, match))
                    i += length
                    found = True
                    break
            if not found:
                i += 1
        return sequence

    def get_keypoints(self, word: str, mapper, target_len=SEQ_LEN) -> np.ndarray:
        matched = mapper.find(word)
        if matched is None:
            return None

        paths = mapper.get_keypoints_paths(matched)
        if not paths:
            return None

        for rel_path in self._rank_sources(paths):
            frames = self._load_source_frames(rel_path, mapper)
            if frames:
                src = _fill_missing_hand_frames(np.stack(frames, axis=0))
                interpolated = self._interpolator.interpolate(src, target_len)
                smoothed = self._smoother.smooth_with_fallback(interpolated)
                return smoothed

        return None

    @staticmethod
    def _rank_sources(paths: list) -> list:
        """Best source first. Unrecognised filenames (e.g. future sources)
        sort last rather than erroring, so they're still tried as a
        fallback."""
        def rank(p):
            name = Path(p).name
            return _SOURCE_PRIORITY.index(name) if name in _SOURCE_PRIORITY else len(_SOURCE_PRIORITY)
        return sorted(paths, key=rank)

    def _load_source_frames(self, rel_path: str, mapper) -> list:
        """Load one source file as a list of frame vectors. A photo-stack
        source (independent single captures, no temporal relationship
        between entries) contributes only its first valid frame — held as
        a single static pose rather than spline-interpolated across
        unrelated people/sessions. A video source contributes all its
        frames for genuine motion interpolation."""
        abs_path = Path(mapper.get_project_root()) / rel_path
        if not abs_path.exists():
            return []
        try:
            arr = self._npy_cache.load(str(abs_path))
        except Exception:
            return []

        is_photo_stack = Path(rel_path).name in _PHOTO_STACK_FILENAMES
        frames = []
        if arr.ndim == 1 and arr.shape[0] == 258:
            if not np.all(np.abs(arr) < 1e-6):
                frames.append(arr)
        elif arr.ndim == 2 and arr.shape[1] == 258:
            for i in range(arr.shape[0]):
                row = arr[i]
                if np.all(np.abs(row) < 1e-6):
                    continue
                frames.append(row)
                if is_photo_stack:
                    break
        return frames

    def run(self, url_or_path: str):
        import cv2

        print("=" * 60)
        print("  Arabic Sign Language Translator")
        print("  URL -> Audio -> Text -> Sign Language -> Video")
        print("=" * 60)

        try:
            print("\n[Step 1/7] Transcribing audio...")
            raw_text = self.transcribe(url_or_path)
            print(f"  Raw text: {raw_text[:120]}...")

            with open(TRANSCRIPTION_FILE, "w", encoding="utf-8") as f:
                f.write(raw_text)
            print(f"  Saved to {TRANSCRIPTION_FILE}")

            print("\n[Step 2/7] Preprocessing Arabic text...")
            clean_text = self.preprocess(raw_text)
            print(f"  Cleaned: {clean_text[:120]}...")

            if not clean_text.strip():
                print("  No text to translate.")
                return

            print("\n[Step 3/7] Loading dictionary...")
            loader = DictionaryLoader(MANIFEST_PATH)
            word_paths, word_set = loader.load()
            matcher = DictionaryMatcher(word_paths, word_set, str(loader.get_project_root()))
            phrase_matcher = PhraseMatcher(word_paths, word_set, matcher)

            print("\n[Step 4/7] Mapping text to signs...")
            word_sequence = self.map_signs(clean_text, matcher, phrase_matcher)
            print(f"  {len(word_sequence)} sign units mapped")
            found_count = sum(1 for _, k in word_sequence if k is not None)
            print(f"  {found_count}/{len(word_sequence)} words in dictionary")

            print("\n[Step 5/7] Generating motion sequences...")
            renderer = LandmarkRenderer(fps=FPS)
            sigml_encoder = SiGMLEncoder(fps=FPS)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (renderer.w, renderer.h))
            total_frames_written = 0
            prev_frames = None

            for display_word, lookup_key in word_sequence:
                if lookup_key is not None:
                    landmarks = self.get_keypoints(lookup_key, matcher, target_len=SEQ_LEN)
                    if landmarks is not None:
                        frames = renderer.render_word_frames(landmarks, display_word, transition_frames=8)
                        sigml_encoder.add_sign(display_word, landmarks)
                        print(f"  + {display_word}")
                    else:
                        frames = renderer.render_unknown_word(display_word, num_frames=SEQ_LEN)
                        sigml_encoder.add_unknown(display_word)
                        print(f"  ! {display_word} (keypoints missing)")
                else:
                    frames = renderer.render_unknown_word(display_word, num_frames=SEQ_LEN)
                    sigml_encoder.add_unknown(display_word)
                    print(f"  ? {display_word} (not in dictionary)")

                for frame in frames:
                    writer.write(frame)
                total_frames_written += len(frames)

            writer.release()

            print("\n[Step 6/7] Saving SiGML...")
            sigml_encoder.save(OUTPUT_SIGML)

            print("\n[Step 7/7] Done!")
            if total_frames_written > 0:
                print(f"  Video : {OUTPUT_VIDEO} ({total_frames_written} frames)")
                print(f"  SiGML : {OUTPUT_SIGML} ({len(sigml_encoder._signs)} signs)")
            else:
                print("  No frames generated.")

        except Exception as e:
            print(f"\nPipeline error: {e}")
            raise


def run_translation_system():
    import cv2
    url = input("\nEnter video URL: ").strip()
    pipeline = Pipeline()
    pipeline.run(url)
