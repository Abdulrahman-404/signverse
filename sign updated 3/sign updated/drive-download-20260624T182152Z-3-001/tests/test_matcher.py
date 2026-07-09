import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from sign_translator.dictionary.matcher import DictionaryMatcher, TrieNode


def test_trie_basic():
    root = TrieNode()
    for word in ["hello", "help", "world"]:
        node = root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.words.append(word)
    collected = []
    DictionaryMatcher._collect_words(root, collected)
    assert len(collected) == 3


def test_levenshtein():
    assert DictionaryMatcher._levenshtein("kitab", "kitab") == 0
    assert DictionaryMatcher._levenshtein("kitab", "kita") == 1
    assert DictionaryMatcher._levenshtein("", "abc") == 3


def test_norm_map_building():
    word_paths = {}
    word_set = {"الكتاب", "مدرسة", "بيت"}
    project_root = "/tmp"
    matcher = DictionaryMatcher(word_paths, word_set, project_root)
    match = matcher.find_exact("الكتاب")
    assert match is not None


if __name__ == "__main__":
    test_trie_basic()
    test_levenshtein()
    test_norm_map_building()
    print("All matcher tests passed ✓")
