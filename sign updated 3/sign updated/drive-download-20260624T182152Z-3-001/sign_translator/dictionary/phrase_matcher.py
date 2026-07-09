from typing import Optional
from ..text.normalizer import normalize_search_key


class PhraseMatcher:
    def __init__(self, word_paths: dict, word_set: set, matcher):
        self._word_paths = word_paths
        self._word_set = word_set
        self._matcher = matcher

        self._multi_word: dict[str, str] = {}
        self._max_phrase_len = 1
        self._build_multi_word_index()

    def _build_multi_word_index(self):
        for word in self._word_set:
            tokens = word.split()
            if len(tokens) >= 2:
                norm = normalize_search_key(word)
                self._multi_word[norm] = word
                if len(tokens) > self._max_phrase_len:
                    self._max_phrase_len = len(tokens)

    def match_phrases(self, tokens: list[str]) -> list[tuple[str, Optional[str]]]:
        i = 0
        result = []
        while i < len(tokens):
            matched = False
            for n in range(min(self._max_phrase_len, len(tokens) - i), 1, -1):
                phrase = ' '.join(tokens[i:i + n])
                norm = normalize_search_key(phrase)
                if norm in self._multi_word:
                    result.append((phrase, self._multi_word[norm]))
                    i += n
                    matched = True
                    break
            if not matched:
                single = self._matcher.find(tokens[i])
                result.append((tokens[i], single if single else None))
                i += 1
        return result
