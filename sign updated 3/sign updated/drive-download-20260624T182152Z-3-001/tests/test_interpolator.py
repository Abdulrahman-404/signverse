import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import numpy as np
from sign_translator.motion.interpolator import MotionInterpolator


def test_interpolate_same_length():
    src = np.random.rand(30, 258).astype(np.float32)
    interp = MotionInterpolator(method='linear')
    result = interp.interpolate(src, 30)
    assert result.shape == (30, 258)


def test_interpolate_double():
    src = np.random.rand(10, 258).astype(np.float32)
    interp = MotionInterpolator(method='cubic')
    result = interp.interpolate(src, 20)
    assert result.shape == (20, 258)


def test_interpolate_single_frame():
    src = np.random.rand(1, 258).astype(np.float32)
    interp = MotionInterpolator(method='cubic')
    result = interp.interpolate(src, 30)
    assert result.shape == (30, 258)
    assert np.allclose(result[0], src[0])


def test_interpolate_empty():
    src = np.zeros((0, 258), dtype=np.float32)
    interp = MotionInterpolator()
    result = interp.interpolate(src, 30)
    assert result.shape == (30, 258)


if __name__ == "__main__":
    test_interpolate_same_length()
    test_interpolate_double()
    test_interpolate_single_frame()
    test_interpolate_empty()
    print("All interpolator tests passed ✓")
