import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import numpy as np
from sign_translator.motion.transitions import SmoothTransitions, ease_in_out, ease_in, ease_out


def test_ease_in_out():
    assert ease_in_out(0.0) == 0.0
    assert abs(ease_in_out(0.5) - 0.5) < 0.01
    assert ease_in_out(1.0) == 1.0


def test_ease_in():
    assert ease_in(0.0) == 0.0
    assert ease_in(1.0) == 1.0


def test_ease_out():
    assert ease_out(0.0) == 0.0
    assert ease_out(1.0) == 1.0


def test_add_hold_frames():
    seq = np.random.rand(10, 258).astype(np.float32)
    t = SmoothTransitions()
    result = t.add_hold_frames(seq, num_hold=5)
    assert result.shape == (15, 258)
    assert np.allclose(result[-1], seq[-1])


def test_crossfade():
    from_seq = np.random.rand(10, 258).astype(np.float32)
    to_seq = np.random.rand(10, 258).astype(np.float32)
    t = SmoothTransitions(transition_frames=5)
    fade = t.crossfade(from_seq, to_seq)
    assert fade.shape == (5, 258)


def test_apply_easing():
    seq = np.random.rand(30, 258).astype(np.float32)
    t = SmoothTransitions()
    result = t.apply_easing(seq)
    assert result.shape == seq.shape


if __name__ == "__main__":
    test_ease_in_out()
    test_ease_in()
    test_ease_out()
    test_add_hold_frames()
    test_crossfade()
    test_apply_easing()
    print("All transitions tests passed ✓")
