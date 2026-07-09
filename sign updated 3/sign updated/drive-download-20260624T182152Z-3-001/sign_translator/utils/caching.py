from collections import OrderedDict
import numpy as np


class NPYCache:
    def __init__(self, maxsize: int = 200):
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._maxsize = maxsize

    def get(self, path: str) -> np.ndarray:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        return None

    def put(self, path: str, arr: np.ndarray):
        if len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)
        self._cache[path] = arr

    def load(self, path: str) -> np.ndarray:
        cached = self.get(path)
        if cached is not None:
            return cached
        arr = np.load(path).astype(np.float32)
        self.put(path, arr)
        return arr

    def clear(self):
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def maxsize(self) -> int:
        return self._maxsize
