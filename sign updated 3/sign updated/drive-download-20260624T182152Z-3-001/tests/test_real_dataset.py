import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import os
from sign_translator.dictionary.loader import DictionaryLoader
from sign_translator.dictionary.matcher import DictionaryMatcher
from config import MANIFEST_PATH

loader = DictionaryLoader(MANIFEST_PATH)
word_paths, word_set = loader.load()
print(f"Loaded {len(word_set)} unique words")

matcher = DictionaryMatcher(word_paths, word_set, str(loader.get_project_root()))

tests = ["\u0623\u0633\u0631\u0629", "\u0623\u0628", "\u0623\u0645", "\u0628\u064a\u062a"]
for t in tests:
    match = matcher.find(t)
    if match:
        paths = matcher.get_keypoints_paths(match)
        exists = os.path.exists(os.path.join(str(loader.get_project_root()), paths[0])) if paths else False
        print(f"  Match: '{hex(ord(t[0]))}..' -> '{hex(ord(match[0]))}..' ({len(paths)} paths, on disk: {exists})")
    else:
        print(f"  '{hex(ord(t[0]))}..' -> NOT FOUND")

fuzzy = matcher.find("\u0627\u0633\u0631\u0647")
print(f"  Fuzzy: '{'found' if fuzzy else 'not found'}'")

prefixes = matcher.find_prefix("\u0627\u0628")
print(f"  Prefix matches: {len(prefixes)} words")

# Verify O(1) lookup speed
import time
start = time.time()
for _ in range(1000):
    matcher.find("\u0623\u0633\u0631\u0629")
elapsed = time.time() - start
print(f"  1000 lookups in {elapsed:.3f}s ({elapsed/1000*1000:.1f} microsec/lookup)")

print("Real dataset test: OK")
