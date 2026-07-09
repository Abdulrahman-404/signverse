import numpy as np


def _hermite_cubic(p0, p1, t0, t1, t):
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1


def _cubic_spline_interp(x_src, y_src, x_tgt):
    n_src = len(x_src)
    if n_src < 2:
        return np.full_like(x_tgt, y_src[0] if n_src == 1 else 0)
    h = np.diff(x_src)
    if np.any(h <= 0):
        return np.interp(x_tgt, x_src, y_src)
    alpha = np.zeros(n_src)
    beta = np.zeros(n_src)
    for i in range(1, n_src - 1):
        hi = h[i - 1]
        hip1 = h[i]
        a = hi / (hi + hip1)
        b = 1 - a
        c = 3 * (a * (y_src[i] - y_src[i - 1]) / hi +
                 b * (y_src[i + 1] - y_src[i]) / hip1)
        denom = 2 + a * beta[i - 1]
        alpha[i] = (c - a * alpha[i - 1]) / denom
        beta[i] = -b / denom
    tangents = np.zeros(n_src)
    for i in range(n_src - 2, 0, -1):
        tangents[i] = alpha[i] + beta[i] * tangents[i + 1]
    tangents[0] = 3 * (y_src[1] - y_src[0]) / (2 * h[0]) - 0.5 * tangents[1]
    tangents[-1] = 3 * (y_src[-1] - y_src[-2]) / (2 * h[-1]) - 0.5 * tangents[-2]
    result = np.zeros_like(x_tgt)
    for i, t in enumerate(x_tgt):
        idx = np.searchsorted(x_src, t) - 1
        idx = max(0, min(idx, n_src - 2))
        local_t = (t - x_src[idx]) / h[idx] if h[idx] > 0 else 0
        result[i] = _hermite_cubic(
            y_src[idx], y_src[idx + 1],
            tangents[idx] * h[idx], tangents[idx + 1] * h[idx],
            local_t
        )
    return result


class MotionInterpolator:
    def __init__(self, method: str = 'cubic'):
        self._method = method

    def interpolate(self, src_frames: np.ndarray, target_len: int) -> np.ndarray:
        if src_frames.ndim != 2:
            raise ValueError(f"src_frames must be 2D, got shape {src_frames.shape}")
        n_src = src_frames.shape[0]
        n_dims = src_frames.shape[1]
        if n_src == target_len:
            return src_frames.copy()
        if n_src == 0:
            return np.zeros((target_len, n_dims), dtype=np.float32)
        if n_src == 1:
            return np.tile(src_frames[0], (target_len, 1))
        x_src = np.linspace(0.0, 1.0, n_src)
        x_tgt = np.linspace(0.0, 1.0, target_len)
        if self._method == 'cubic':
            out = np.zeros((target_len, n_dims), dtype=np.float32)
            for dim in range(n_dims):
                out[:, dim] = _cubic_spline_interp(x_src, src_frames[:, dim], x_tgt)
            return out
        out = np.zeros((target_len, n_dims), dtype=np.float32)
        for dim in range(n_dims):
            out[:, dim] = np.interp(x_tgt, x_src, src_frames[:, dim])
        return out
