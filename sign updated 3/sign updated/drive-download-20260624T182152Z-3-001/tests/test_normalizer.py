import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from sign_translator.text.normalizer import (
    normalize_arabic, normalize_search_key,
    remove_punctuation, is_arabic_word
)


def test_alef_normalization():
    assert normalize_arabic("أحمد") == "احمد"
    assert normalize_arabic("إبراهيم") == "ابراهيم"
    assert normalize_arabic("آدم") == "ادم"


def test_yeh_normalization():
    assert normalize_arabic("مصطفى") == "مصطفي"  # alif maksura (U+0649) → yeh (U+064A)


def test_teh_marbuta():
    assert normalize_arabic("مدرسة") == "مدرسه"


def test_kashida_removal():
    assert normalize_arabic("جمـــيل") == "جميل"


def test_tashkeel_removal():
    result = normalize_arabic("\u0645\u064f\u062d\u064e\u0645\u064e\u0651\u062f")
    expected = "\u0645\u062d\u0645\u062f"  # محمدهاش after removing tashkeel
    assert result == expected, f"Got {[hex(ord(c)) for c in result]}, expected {[hex(ord(c)) for c in expected]}"
    assert normalize_arabic("كِتابٌ") == "كتاب"


def test_normalize_search_key():
    assert normalize_search_key("الكتاب المدرسي") == "الكتابالمدرسي"


def test_remove_punctuation():
    assert remove_punctuation("مرحبا! كيف حالك؟").strip() == "مرحبا  كيف حالك"


def test_is_arabic_word():
    assert is_arabic_word("مرحبا") is True
    assert is_arabic_word("hello") is False


if __name__ == "__main__":
    test_alef_normalization()
    test_yeh_normalization()
    test_teh_marbuta()
    test_kashida_removal()
    test_tashkeel_removal()
    test_normalize_search_key()
    test_remove_punctuation()
    test_is_arabic_word()
    print("All normalizer tests passed ✓")
