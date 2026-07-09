import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import numpy as np
from sign_translator import (
    ArabicPreProcessor, DictionaryMatcher, PhraseMatcher,
    MotionInterpolator, SmoothTransitions, TemporalSmoother,
    LandmarkRenderer, SiGMLEncoder
)


def test_preprocessing():
    text = "\u0645\u0631\u062d\u0628\u0627\n\u0643\u064a\u0641 \u062d\u0627\u0644\u0643\u061f \u0623\u0646\u0627 \u0628\u062e\u064a\u0631."
    clean = ArabicPreProcessor.full_pipeline(text)
    assert "\u0645\u0631\u062d\u0628\u0627" in clean  # مرحبا
    assert "\u0643\u064a\u0641" in clean  # كيف
    assert "\u0628\u062e\u064a\u0631" in clean  # بخير
    assert "\u061f" not in clean  # ? removed
    assert "." not in clean
    print("  Preprocessing: OK")


def test_interpolation():
    src = np.random.rand(5, 258).astype(np.float32)
    interp = MotionInterpolator(method='cubic')
    result = interp.interpolate(src, 30)
    assert result.shape == (30, 258)
    print(f"  Interpolation: {src.shape} -> {result.shape}")


def test_transitions():
    seq = np.random.rand(30, 258).astype(np.float32)
    t = SmoothTransitions(transition_frames=5, hold_frames=3)
    held = t.add_hold_frames(seq)
    assert held.shape == (33, 258)
    faded = t.crossfade(seq[:10], seq[10:20])
    assert faded.shape == (5, 258)
    eased = t.apply_easing(seq)
    assert eased.shape == seq.shape
    print("  Transitions: hold/crossfade/easing OK")


def test_smoothing():
    seq = np.random.rand(30, 258).astype(np.float32)
    s = TemporalSmoother(window=5, order=2)
    result = s.smooth_with_fallback(seq)
    assert result.shape == seq.shape
    print("  Smoothing: OK")


def test_rendering():
    renderer = LandmarkRenderer()
    word = "\u0645\u0631\u062d\u0628\u0627"  # مرحبا
    frame = renderer.render_frame(np.random.rand(258).astype(np.float32), word)
    assert frame.shape == (640, 640, 3)
    frames = renderer.render_word_frames(np.random.rand(30, 258).astype(np.float32), word)
    assert len(frames) == 38
    unknown = renderer.render_unknown_word(word)
    assert len(unknown) == 30
    print("  Rendering: OK")


def test_sigml():
    encoder = SiGMLEncoder(fps=30)
    encoder.add_sign("test", np.random.rand(30, 258).astype(np.float32))
    encoder.add_unknown("unknown_word")
    assert len(encoder._signs) == 2
    root = encoder.build_xml()
    signs = root.findall("hamgestural_sign")
    assert len(signs) == 2
    print("  SiGML: OK")


if __name__ == "__main__":
    test_preprocessing()
    test_interpolation()
    test_transitions()
    test_smoothing()
    test_rendering()
    test_sigml()
    print("\nPipeline logic: ALL OK")
