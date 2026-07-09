#!/usr/bin/env python3
"""
Terminal testing interface for the Arabic Sign Language pipeline.

Usage:
    python test_pipeline.py

Menu:
    1. Arabic Text   — enter text directly, skip Whisper
    2. YouTube URL   — download audio, transcribe, then full pipeline
    3. Audio File    — transcribe local file, then full pipeline

Reports every stage with exact error location on failure.
Does NOT modify the backend or project architecture.
"""
import sys, os, json, traceback
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

# ── ensure we can import project modules ──────────────────────────────────
_PROJECT = Path(__file__).parent.absolute()
sys.path.insert(0, str(_PROJECT))
os.chdir(str(_PROJECT))

from config import MANIFEST_PATH, WHISPER_MODEL, SEQ_LEN, FPS
from sign_translator.pipeline import Pipeline
from sign_translator.dictionary.loader import DictionaryLoader
from sign_translator.dictionary.matcher import DictionaryMatcher
from sign_translator.dictionary.phrase_matcher import PhraseMatcher
from sign_translator.render.renderer import LandmarkRenderer
from sign_translator.sigml.encoder import SiGMLEncoder
from sign_translator.text.preprocessor import ArabicPreProcessor


# ── safe console output for Arabic text on Windows ────────────────────────
def _print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        safe = msg.encode("ascii", "replace").decode("ascii")
        print(safe)


# ── stage tracker ─────────────────────────────────────────────────────────
class Stage:
    """Context that records the current stage name for error reporting."""
    _current: str = "startup"


def _set_stage(name: str):
    Stage._current = name


def _stage_header(label: str, detail: str = ""):
    bar = "-" * 50
    _print(f"\n{bar}")
    _print(f"  [{label}]  {detail}")
    _print(bar)


def _stage_ok(msg: str):
    _print(f"  [OK] {msg}")


def _stage_info(msg: str):
    _print(f"       {msg}")


# ── menu ──────────────────────────────────────────────────────────────────
def _show_menu():
    _print("\n" + "=" * 56)
    _print("  Arabic Sign Language Pipeline  -  Test Interface")
    _print("=" * 56)
    _print("  1  Arabic Text")
    _print("  2  YouTube URL")
    _print("  3  Audio File")
    _print("  q  Quit")
    _print("=" * 56)


def _choose(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        _print("")
        return "q"


# ── dictionary helpers (load once, reuse) ─────────────────────────────────
_dict_loader = None
_dict_matcher = None
_dict_phrase_matcher = None


def _ensure_dictionary():
    global _dict_loader, _dict_matcher, _dict_phrase_matcher
    if _dict_matcher is not None:
        return _dict_loader, _dict_matcher, _dict_phrase_matcher
    _set_stage("dictionary_load")
    _stage_header("STAGE 3", "Loading dictionary")
    _dict_loader = DictionaryLoader(MANIFEST_PATH)
    word_paths, word_set = _dict_loader.load()
    _dict_matcher = DictionaryMatcher(
        word_paths, word_set, str(_dict_loader.get_project_root())
    )
    _dict_phrase_matcher = PhraseMatcher(word_paths, word_set, _dict_matcher)
    _stage_ok(f"{len(word_set)} unique words, {sum(len(v) for v in word_paths.values())} keypoint files")
    return _dict_loader, _dict_matcher, _dict_phrase_matcher


# ── dataset source reporter ────────────────────────────────────────────────
def _report_dataset_source(word: str, matcher: DictionaryMatcher):
    """Report which .npy files exist for a word (original, karsl, or both)."""
    paths = matcher.get_keypoints_paths(word)
    flags = []
    for p in paths:
        abs_p = Path(str(matcher.get_project_root())) / p
        label = "karsl_aug" if "karsl_aug" in p else "original"
        exists = abs_p.exists()
        if exists:
            arr = np.load(abs_p)
            frames = arr.shape[0] if arr.ndim >= 1 else 1
            dim = arr.shape[1] if arr.ndim == 2 else 1
            flags.append(f"{label} ({frames} frames, {dim} dims)")
        else:
            flags.append(f"{label} (FILE MISSING)")
    if flags:
        _stage_info("sources: " + " + ".join(flags))
    else:
        _stage_info("sources: none found")


# ── full pipeline runner ──────────────────────────────────────────────────
def _run(text_input: str, source_label: str,
         output_dir: Path, skip_whisper: bool):
    """
    Execute the full pipeline:
        1.  Input
        2.  Whisper (optional)
        3.  Arabic preprocessing
        4.  Dictionary lookup
        5.  Dataset source inspection
        6.  Motion interpolation
        7.  SiGML generation
        8.  Avatar rendering
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_video = str(output_dir / f"test_output_{timestamp}.mp4")
    out_sigml = str(output_dir / f"test_output_{timestamp}.sigml")

    # ── STAGE 1: Input ───────────────────────────────────────────────────
    _set_stage("input")
    _stage_header("STAGE 1", "Input")
    _stage_ok(f"source: {source_label}")
    _stage_info(f"raw input length: {len(text_input)} characters")

    try:
        raw_text = text_input

        # ── STAGE 2: Whisper (if applicable) ─────────────────────────────
        if not skip_whisper:
            _set_stage("whisper")
            _stage_header("STAGE 2", "Whisper transcription")
            pipeline = Pipeline()
            _stage_info("transcribing audio ...")
            raw_text = pipeline.transcribe(raw_text)
            _stage_ok(f"transcription: \"{raw_text[:200]}\"")
        else:
            _stage_info("(skipped - direct Arabic text input)")

        # ── STAGE 3: Dictionary (loaded lazily above) ────────────────────
        loader, matcher, phrase_matcher = _ensure_dictionary()

        # ── STAGE 4: Arabic preprocessing ────────────────────────────────
        _set_stage("preprocessing")
        _stage_header("STAGE 4", "Arabic preprocessing")
        clean_text = ArabicPreProcessor.full_pipeline(raw_text)
        if not clean_text.strip():
            _stage_info("WARNING: empty after preprocessing, no signs to render")
        _stage_ok(f"cleaned text length: {len(clean_text)} characters")
        word_count = len(clean_text.split())
        _stage_info(f"word count: {word_count}")

        # ── STAGE 5: Dictionary lookup ───────────────────────────────────
        _set_stage("dictionary_lookup")
        _stage_header("STAGE 5", "Dictionary lookup")

        pipeline = Pipeline()
        word_sequence = pipeline.map_signs(clean_text, matcher, phrase_matcher)
        total = len(word_sequence)
        found = sum(1 for _, k in word_sequence if k is not None)
        _stage_ok(f"{total} sign units, {found} in dictionary")

        for display_word, lookup_key in word_sequence:
            status = "DICTIONARY HIT" if lookup_key else "NOT FOUND"
            _stage_info(f"  {repr(display_word)}  =>  {status}")
            if lookup_key:
                _report_dataset_source(lookup_key, matcher)

        if found == 0:
            _stage_info("WARNING: no dictionary matches, rendering placeholders")

        # ── STAGE 6: Motion interpolation ────────────────────────────────
        _set_stage("motion")
        _stage_header("STAGE 6", "Motion interpolation")
        all_landmarks: list[tuple[str, np.ndarray]] = []
        # per_unit holds the landmarks (or None) for each word_sequence
        # entry, in lockstep — not reconstructed afterward via membership-
        # testing display words against lookup keys. That comparison is
        # broken whenever a matched word's display text differs from its
        # canonical dictionary form (e.g. a fuzzy match), and silently
        # double-counts the word as both matched AND unknown in the SiGML.
        per_unit: list[np.ndarray | None] = []
        for display_word, lookup_key in word_sequence:
            lms = None
            if lookup_key is not None:
                lms = pipeline.get_keypoints(lookup_key, matcher,
                                              target_len=SEQ_LEN)
            per_unit.append(lms)
            if lms is not None:
                all_landmarks.append((display_word, lms))
                _stage_info(f"  {repr(display_word)}  "
                            f"(30 frames, {lms.shape[1]} dims)")
            else:
                reason = "keypoints missing" if lookup_key is not None else "no lookup key"
                _stage_info(f"  {repr(display_word)}  {reason}")

        _stage_ok(f"motion sequences generated for {len(all_landmarks)} words")

        # ── STAGE 7: SiGML generation ────────────────────────────────────
        _set_stage("sigml")
        _stage_header("STAGE 7", "SiGML generation")
        sigml_encoder = SiGMLEncoder(fps=FPS)
        for (display_word, _lookup_key), lms in zip(word_sequence, per_unit):
            if lms is not None:
                sigml_encoder.add_sign(display_word, lms)
            else:
                sigml_encoder.add_unknown(display_word)
        sigml_encoder.save(out_sigml)
        size_kb = Path(out_sigml).stat().st_size / 1024
        _stage_ok(f"SiGML file: {out_sigml}  ({size_kb:.0f} KB, "
                  f"{len(sigml_encoder._signs)} signs)")

        # ── STAGE 8: Avatar rendering ────────────────────────────────────
        _set_stage("render")
        _stage_header("STAGE 8", "Avatar rendering")
        import cv2
        renderer = LandmarkRenderer(fps=FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video, fourcc, FPS,
                                 (renderer.w, renderer.h))
        total_frames = 0
        for word, lms in all_landmarks:
            frames = renderer.render_word_frames(lms, word,
                                                  transition_frames=8)
            for f in frames:
                writer.write(f)
            total_frames += len(frames)
        writer.release()
        video_size_kb = Path(out_video).stat().st_size / 1024
        _stage_ok(f"video file: {out_video}  ({video_size_kb:.0f} KB, "
                  f"{total_frames} frames)")

        # ── SUMMARY ──────────────────────────────────────────────────────
        _set_stage("done")
        _print("\n" + "#" * 56)
        _print("  PIPELINE COMPLETED SUCCESSFULLY")
        _print("#" * 56)
        _print(f"  Input words:        {total}")
        _print(f"  Dictionary hits:    {found}")
        _print(f"  Motion sequences:   {len(all_landmarks)}")
        _print(f"  SiGML signs:        {len(sigml_encoder._signs)}")
        _print(f"  Video frames:       {total_frames}")
        _print(f"  SiGML path:         {out_sigml}")
        _print(f"  Video path:         {out_video}")
        _print("#" * 56)

    except Exception:
        _print("\n" + "!" * 56)
        _print(f"  PIPELINE FAILED at stage: [{Stage._current}]")
        _print("!" * 56)
        traceback.print_exc()
        _print("!" * 56)


# ── main ──────────────────────────────────────────────────────────────────
def main():
    output_dir = _PROJECT / "test_output"
    output_dir.mkdir(exist_ok=True)

    while True:
        _show_menu()
        choice = _choose("  Choose option [1-3/q]: ")

        if choice == "q":
            _print("\nDone.")
            break

        elif choice == "1":
            text = _choose("\n  Enter Arabic text: ")
            if not text:
                _print("  (empty, skipped)")
                continue
            _run(text, "direct text input", output_dir, skip_whisper=True)

        elif choice == "2":
            url = _choose("\n  Enter YouTube URL: ")
            if not url:
                _print("  (empty, skipped)")
                continue
            _run(url, f"YouTube URL: {url}", output_dir, skip_whisper=False)

        elif choice == "3":
            audio_path = _choose("\n  Enter audio file path: ")
            if not audio_path:
                _print("  (empty, skipped)")
                continue
            if not os.path.isfile(audio_path):
                _print(f"  ERROR: file not found: {audio_path}")
                continue
            _run(audio_path, f"audio file: {audio_path}",
                 output_dir, skip_whisper=False)

        else:
            _print(f"  Unknown option: {choice}")


if __name__ == "__main__":
    main()
