import re
from .normalizer import (
    normalize_arabic, normalize_search_key,
    remove_punctuation, is_arabic_word
)

_CONNECTOR_PREFIXES = re.compile(
    r'\b([ووفببلل])'
)

_KNOWN_CONNECTORS = {
    'و', 'ف', 'ب', 'ل', 'ك', 'س', 'أ', 'إ', 'آ',
}

_ARABIC_DIGITS = '0123456789٠١٢٣٤٥٦٧٨٩'


def _normalize_digits(text: str) -> str:
    result = []
    for ch in text:
        if '0' <= ch <= '9':
            result.append(ch)
        elif '٠' <= ch <= '٩':
            result.append(chr(ord('0') + ord(ch) - ord('٠')))
        else:
            result.append(ch)
    return ''.join(result)


class ArabicPreProcessor:
    @staticmethod
    def fix_text(text: str) -> str:
        if not text:
            return ""
        has_isolated = any('\uFE70' <= ch <= '\uFEFF' for ch in text)
        is_spaced = (len(text.split()) > 2 and
                     sum(1 for t in text.split() if len(t) == 1) / len(text.split()) > 0.4)
        if is_spaced or has_isolated:
            text = re.sub(r'\s+', '', text)
        text = unicodedata.normalize('NFKC', text)
        return unicodedata.normalize('NFC', text)

    @staticmethod
    def normalize(word: str) -> str:
        return normalize_arabic(word)

    @staticmethod
    def full_pipeline(text: str) -> str:
        if not text:
            return ""
        text = _normalize_digits(text)
        text = remove_punctuation(text)
        text = ArabicPreProcessor.fix_text(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def tokenize(text: str):
        tokens = text.split()
        result = []
        for token in tokens:
            if len(token) == 1 and token in _KNOWN_CONNECTORS:
                result.append(token)
                continue
            splits = _CONNECTOR_PREFIXES.split(token)
            for part in splits:
                if part:
                    result.append(part)
        return result

    @staticmethod
    def normalize_token(token: str) -> str:
        return normalize_search_key(token)


import unicodedata
