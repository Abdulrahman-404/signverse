import re

_ARABIC_SENTENCE_BREAKS = re.compile(
    r'[؟.!?]\s*'
)

_MIN_SEGMENT_LENGTH = 3


class SentenceSegmenter:
    @staticmethod
    def split_sentences(text: str) -> list[str]:
        if not text.strip():
            return []

        raw_segments = _ARABIC_SENTENCE_BREAKS.split(text)
        merged = []
        buffer = []
        for seg in raw_segments:
            seg = seg.strip()
            if not seg:
                continue
            tokens = seg.split()
            if len(tokens) < _MIN_SEGMENT_LENGTH and buffer:
                buffer.append(seg)
                merged.append(' '.join(buffer))
                buffer = []
            else:
                if buffer:
                    merged.append(' '.join(buffer))
                    buffer = []
                if len(tokens) > 10:
                    sub_parts = SentenceSegmenter._split_long_segment(seg)
                    merged.extend(sub_parts)
                else:
                    merged.append(seg)
        if buffer:
            merged.append(' '.join(buffer))
        return merged if merged else [text]

    @staticmethod
    def _split_long_segment(text: str, max_tokens: int = 8) -> list[str]:
        tokens = text.split()
        parts = []
        for i in range(0, len(tokens), max_tokens):
            parts.append(' '.join(tokens[i:i + max_tokens]))
        return parts

    @staticmethod
    def segment_to_utterances(tokens: list[str], max_words: int = 5) -> list[list[str]]:
        utterances = []
        current = []
        for token in tokens:
            if token in {'و', 'ف', 'ثم', 'لكن', 'بل', 'أو', 'لأن'}:
                if current:
                    utterances.append(current)
                current = [token]
            else:
                current.append(token)
                if len(current) >= max_words:
                    utterances.append(current)
                    current = []
        if current:
            utterances.append(current)
        if not utterances:
            utterances = [tokens]
        return utterances
