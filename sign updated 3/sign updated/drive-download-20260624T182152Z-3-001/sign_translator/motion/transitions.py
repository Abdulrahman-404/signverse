import numpy as np


def ease_in_out(t: float) -> float:
    if t < 0.5:
        return 2 * t * t
    return -1 + (4 - 2 * t) * t


def ease_in(t: float) -> float:
    return t * t


def ease_out(t: float) -> float:
    return t * (2 - t)


class SmoothTransitions:
    def __init__(self, transition_frames: int = 10, hold_frames: int = 5):
        self._transition_frames = transition_frames
        self._hold_frames = hold_frames

    def add_hold_frames(self, seq: np.ndarray, num_hold: int = None) -> np.ndarray:
        if num_hold is None:
            num_hold = self._hold_frames
        if num_hold <= 0 or len(seq) == 0:
            return seq
        last_frame = seq[-1:]
        hold = np.repeat(last_frame, num_hold, axis=0)
        return np.concatenate([seq, hold], axis=0)

    def crossfade(self, from_seq: np.ndarray, to_seq: np.ndarray) -> np.ndarray:
        n = self._transition_frames
        if n <= 0 or len(from_seq) == 0:
            return to_seq
        fade = np.zeros((n, from_seq.shape[1]), dtype=np.float32)
        for i in range(n):
            alpha = ease_in_out(i / (n - 1)) if n > 1 else 1.0
            from_idx = min(i, len(from_seq) - 1)
            to_idx = min(i, len(to_seq) - 1)
            fade[i] = (1 - alpha) * from_seq[from_idx] + alpha * to_seq[to_idx]
        return fade

    def apply_easing(self, seq: np.ndarray) -> np.ndarray:
        if len(seq) < 2:
            return seq
        n = len(seq)
        weights = np.array([ease_in_out(i / (n - 1)) for i in range(n)])
        centroid = np.mean(seq, axis=0)
        delta = seq - centroid
        return centroid + delta * weights[:, np.newaxis]
