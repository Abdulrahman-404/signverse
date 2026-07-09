import unicodedata
import re

_ARABIC_ALEF_VARIANTS = r'[أإآٱ]'
_ARABIC_YEH_VARIANTS = r'[ىي]'
_ARABIC_WAW_VARIANTS = r'[ؤ]'
_TASHKEEL = r'[ً-ْ]'
_KASHIDA = r'[ـ]'
_LAM_ALEF_LIGATURES = r'[ﻻﻼﻵﻶ]'
_TEH_MARBUTA = r'[ة]'
_HEH = r'[ه]'
_ALEF_MAKSURA = r'[ى]'
_YEH = r'[ي]'

_NORMALIZATION_MAP = {
    ord('أ'): 'ا', ord('إ'): 'ا', ord('آ'): 'ا', ord('ٱ'): 'ا',
    ord('ؤ'): 'و',
    ord('ئ'): 'ي',
    ord('ة'): 'ه',
    ord('ى'): 'ي',
    ord('ﯨ'): 'ي',
    ord('ﯩ'): 'ي',
    ord('ك'): 'ك',
    ord('ﻚ'): 'ك', ord('ﻜ'): 'ك', ord('ﻛ'): 'ك',
    ord('ﻪ'): 'ه', ord('ﻬ'): 'ه', ord('ﻫ'): 'ه',
    ord('ﻲ'): 'ي', ord('ﻴ'): 'ي', ord('ﻳ'): 'ي',
    ord('ﺎ'): 'ا', ord('ﺍ'): 'ا',
    ord('ﺐ'): 'ب', ord('ﺒ'): 'ب', ord('ﺑ'): 'ب',
    ord('ﺔ'): 'ه', ord('ﺘ'): 'ت', ord('ﺗ'): 'ت',
    ord('ﺚ'): 'ث', ord('ﺜ'): 'ث', ord('ﺛ'): 'ث',
    ord('ﺞ'): 'ج', ord('ﺠ'): 'ج', ord('ﺟ'): 'ج',
    ord('ﺢ'): 'ح', ord('ﺤ'): 'ح', ord('ﺣ'): 'ح',
    ord('ﺦ'): 'خ', ord('ﺨ'): 'خ', ord('ﺧ'): 'خ',
    ord('ﺪ'): 'د', ord('ﺩ'): 'د',
    ord('ﺬ'): 'ذ', ord('ﺫ'): 'ذ',
    ord('ﺮ'): 'ر', ord('ﺭ'): 'ر',
    ord('ﺰ'): 'ز', ord('ﺯ'): 'ز',
    ord('ﺲ'): 'س', ord('ﺴ'): 'س', ord('ﺳ'): 'س',
    ord('ﺶ'): 'ش', ord('ﺸ'): 'ش', ord('ﺷ'): 'ش',
    ord('ﺺ'): 'ص', ord('ﺼ'): 'ص', ord('ﺻ'): 'ص',
    ord('ﺾ'): 'ض', ord('ﻀ'): 'ض', ord('ﺿ'): 'ض',
    ord('ﻂ'): 'ط', ord('ﻄ'): 'ط', ord('ﻃ'): 'ط',
    ord('ﻆ'): 'ظ', ord('ﻈ'): 'ظ', ord('ﻇ'): 'ظ',
    ord('ﻊ'): 'ع', ord('ﻌ'): 'ع', ord('ﻋ'): 'ع',
    ord('ﻎ'): 'غ', ord('ﻐ'): 'غ', ord('ﻏ'): 'غ',
    ord('ﻒ'): 'ف', ord('ﻔ'): 'ف', ord('ﻓ'): 'ف',
    ord('ﻖ'): 'ق', ord('ﻘ'): 'ق', ord('ﻗ'): 'ق',
    ord('ﻚ'): 'ك', ord('ﻜ'): 'ك', ord('ﻛ'): 'ك',
    ord('ﻞ'): 'ل', ord('ﻠ'): 'ل', ord('ﻟ'): 'ل',
    ord('ﻢ'): 'م', ord('ﻤ'): 'م', ord('ﻣ'): 'م',
    ord('ﻦ'): 'ن', ord('ﻨ'): 'ن', ord('ﻧ'): 'ن',
    ord('ﻪ'): 'ه', ord('ﻬ'): 'ه', ord('ﻫ'): 'ه',
    ord('ﻮ'): 'و', ord('ﻭ'): 'و',
    ord('ﻲ'): 'ي', ord('ﻴ'): 'ي', ord('ﻳ'): 'ي',
}

def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(_KASHIDA, '', text)
    text = re.sub(_TASHKEEL, '', text)
    text = text.translate(_NORMALIZATION_MAP)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_search_key(text: str) -> str:
    return normalize_arabic(text).replace(' ', '')


def remove_punctuation(text: str) -> str:
    arabic_punct = r'[؟،؛!\.\?,:;\-\(\)\[\]{}""'+"'"+'»«]'
    return re.sub(arabic_punct, ' ', text)


def is_arabic_char(ch: str) -> bool:
    return '\u0600' <= ch <= '\u06FF' or '\u0750' <= ch <= '\u077F' or '\u08A0' <= ch <= '\u08FF'


def is_arabic_word(word: str) -> bool:
    if not word:
        return False
    return sum(1 for ch in word if is_arabic_char(ch)) / max(len(word), 1) > 0.5
