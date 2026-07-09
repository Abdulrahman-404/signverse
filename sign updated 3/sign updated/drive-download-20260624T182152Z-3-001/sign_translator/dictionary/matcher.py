from typing import Optional
from ..text.normalizer import normalize_search_key
from ..utils.caching import NPYCache


class TrieNode:
    __slots__ = ('children', 'words')
    def __init__(self):
        self.children = {}
        self.words = []


class DictionaryMatcher:
    def __init__(self, word_paths: dict, word_set: set, project_root: str):
        self._word_paths = word_paths
        self._word_set = word_set
        self._project_root = project_root
        self._norm_map: dict[str, str] = {}
        self._trie_root = TrieNode()
        self._npy_cache = NPYCache(maxsize=200)

        self._build_norm_map()
        self._build_trie()

    def _build_norm_map(self):
        for word in self._word_set:
            norm = normalize_search_key(word)
            self._norm_map.setdefault(norm, word)

    def _build_trie(self):
        for word in self._word_set:
            norm = normalize_search_key(word)
            node = self._trie_root
            for ch in norm:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
            node.words.append(word)

    def _norm(self, word: str) -> str:
        return normalize_search_key(word)

    def find_exact(self, word: str) -> Optional[str]:
        # A literal match always wins over the normalized lookup. Without
        # this, two distinct dictionary entries that normalize to the same
        # search key (e.g. 'ة' and 'ه' both normalize to 'ه') leave one of
        # them permanently unreachable — norm_map can only point to one
        # canonical word per key, so the other has real keypoint data on
        # disk that find() can never return.
        if word in self._word_set:
            return word
        norm = self._norm(word)
        return self._norm_map.get(norm)

    def find_substring(self, word: str) -> Optional[str]:
        norm = self._norm(word)
        # A short search token (e.g. 'في' "in") coincidentally appears as a
        # contiguous letter run inside many unrelated longer phrases (e.g.
        # inside 'زفير' "exhale") purely by chance — Arabic's limited letter
        # set makes 2-3 letter overlaps common even between semantically
        # unrelated words. Below this length, a "substring" match is noise,
        # not signal, so require find_exact/find_fuzzy instead. Mirrors the
        # same reasoning find_fuzzy already applies to short words.
        if len(norm) <= 3:
            return None
        for canonical_norm, canonical_word in self._norm_map.items():
            # A single-letter dictionary entry (alphabet sign) is a substring
            # of almost every multi-letter word — skip it here so it can't
            # hijack whole-word matching. Single letters still match via
            # find_exact when the input itself is a single letter.
            if len(canonical_norm) <= 1:
                continue
            if norm in canonical_norm or canonical_norm in norm:
                return canonical_word
        return None

    def find_prefix(self, prefix: str) -> list[str]:
        norm = self._norm(prefix)
        node = self._trie_root
        for ch in norm:
            if ch not in node.children:
                return []
            node = node.children[ch]
        results = []
        self._collect_words(node, results)
        results.sort(key=len, reverse=True)
        return results

    def find_fuzzy(self, word: str, max_dist: int = 2) -> Optional[str]:
        norm = self._norm(word)
        if not norm:
            return None
        best = None
        best_dist = max_dist + 1
        for canonical_norm, canonical_word in self._norm_map.items():
            # A fixed absolute edit-distance budget is too loose for short
            # strings: 2 edits is half of a 4-letter word, so unrelated
            # short words collide easily (a name like 'انجي' matched
            # 'انسجه' "tissue" at distance 2 with no length-awareness).
            # Scale the allowed distance down for short candidates —
            # longer words keep the full budget, since a couple of edits
            # there is more plausibly a real typo/ASR slip than a
            # coincidental match to an unrelated word.
            # Words of 3 characters or fewer get no fuzzy tolerance at all —
            # at that length even a 1-edit "close" match (e.g. 'بيت' house
            # vs 'بنت' girl) is just as likely to be a different real word
            # as a typo, and Arabic's limited consonant set means short
            # words collide constantly. find_exact already handles a true
            # exact match for these; anything else should fall through to
            # fingerspelling instead of guessing.
            shortest = min(len(norm), len(canonical_norm))
            allowed = 0 if shortest <= 3 else min(max_dist, shortest // 3)
            dist = self._levenshtein(norm, canonical_norm)
            if dist <= allowed and dist < best_dist:
                best_dist = dist
                best = canonical_word
        return best

    def find(self, word: str) -> Optional[str]:
        result = self.find_exact(word)
        if result:
            return result
        result = self.find_substring(word)
        if result:
            return result
        result = self.find_fuzzy(word, max_dist=2)
        if result:
            return result
        return None

    def get_keypoints_paths(self, matched_word: str) -> list[str]:
        return self._word_paths.get(matched_word, [])

    def get_project_root(self):
        return self._project_root

    @staticmethod
    def _collect_words(node: TrieNode, out: list):
        if node.words:
            out.extend(node.words)
        for child in node.children.values():
            DictionaryMatcher._collect_words(child, out)

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        if len(a) < len(b):
            a, b = b, a
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i]
            for j, cb in enumerate(b, 1):
                cost = 0 if ca == cb else 1
                curr.append(min(
                    curr[-1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + cost
                ))
            prev = curr
        return prev[-1]
