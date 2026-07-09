import numpy as np


def _savitzky_golay_1d(y: np.ndarray, window: int, order: int) -> np.ndarray:
    if window % 2 == 0 or window < 3:
        return y
    if order >= window:
        return y
    half = window // 2
    A = np.zeros((window, order + 1))
    for i in range(window):
        for j in range(order + 1):
            A[i, j] = (i - half) ** j
    coeffs = np.linalg.lstsq(A, np.eye(window), rcond=None)[0][0]
    padded = np.pad(y, (half, half), mode='edge')
    return np.convolve(padded, coeffs[::-1], mode='valid')


class TemporalSmoother:
    def __init__(self, window: int = 5, order: int = 2):
        self._window = window
        self._order = order

    def smooth(self, seq: np.ndarray) -> np.ndarray:
        if len(seq) < self._window:
            return seq
        out = np.zeros_like(seq)
        for dim in range(seq.shape[1]):
            out[:, dim] = _savitzky_golay_1d(seq[:, dim], self._window, self._order)
        return out

    def smooth_with_fallback(self, seq: np.ndarray) -> np.ndarray:
        try:
            return self.smooth(seq)
        except np.linalg.LinAlgError:
            window = 3
            out = np.zeros_like(seq)
            for dim in range(seq.shape[1]):
                kernel = np.ones(window) / window
                padded = np.pad(seq[:, dim], (window // 2, window // 2), mode='edge')
                out[:, dim] = np.convolve(padded, kernel, mode='valid')
            return out
