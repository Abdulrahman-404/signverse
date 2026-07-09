import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import numpy as np
import tempfile
import os
from sign_translator.utils.caching import NPYCache


def test_cache_basic():
    cache = NPYCache(maxsize=10)
    assert cache.size == 0

    arr = np.random.rand(258).astype(np.float32)
    cache.put("test_key", arr)
    assert cache.size == 1

    retrieved = cache.get("test_key")
    assert retrieved is not None
    assert np.allclose(retrieved, arr)


def test_cache_eviction():
    cache = NPYCache(maxsize=3)
    for i in range(5):
        cache.put(f"key_{i}", np.random.rand(258).astype(np.float32))
    assert cache.size == 3
    assert cache.get("key_0") is None
    assert cache.get("key_4") is not None


def test_cache_file_load():
    cache = NPYCache(maxsize=10)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        path = f.name
        arr = np.random.rand(258).astype(np.float32)
        np.save(path, arr)

    try:
        loaded = cache.load(path)
        assert loaded.shape == (258,)
        assert np.allclose(loaded, arr)
        assert cache.size == 1
        loaded_again = cache.load(path)
        assert cache.size == 1
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_cache_basic()
    test_cache_eviction()
    test_cache_file_load()
    print("All caching tests passed ✓")
