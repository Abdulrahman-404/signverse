import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import numpy as np
from sign_translator.motion.smoother import TemporalSmoother


def test_smoother_basic():
    seq = np.random.rand(30, 258).astype(np.float32)
    s = TemporalSmoother(window=5, order=2)
    result = s.smooth(seq)
    assert result.shape == seq.shape


def test_smoother_short_sequence():
    seq = np.random.rand(3, 258).astype(np.float32)
    s = TemporalSmoother(window=5, order=2)
    result = s.smooth(seq)
    assert result.shape == seq.shape


def test_smoother_fallback():
    seq = np.random.rand(30, 258).astype(np.float32)
    s = TemporalSmoother(window=5, order=2)
    result = s.smooth_with_fallback(seq)
    assert result.shape == seq.shape


def test_smoother_constant():
    seq = np.ones((30, 258), dtype=np.float32)
    s = TemporalSmoother(window=5, order=2)
    result = s.smooth_with_fallback(seq)
    assert np.allclose(result, 1.0)


if __name__ == "__main__":
    test_smoother_basic()
    test_smoother_short_sequence()
    test_smoother_fallback()
    test_smoother_constant()
    print("All smoother tests passed ✓")
