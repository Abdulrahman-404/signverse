import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

_FONT_CANDIDATES = [
    "arial.ttf",
    "DejaVuSans.ttf",
    "NotoSansArabic-Regular.ttf",
    "tahoma.ttf",
    "times.ttf",
]


def _find_font(size: int = 40):
    import os
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            pass
    font_dirs = [
        r"C:\Windows\Fonts",
        r"C:\Windows\WinSxS\amd64_microsoft-windows-f..truetype-segoeui_31bf3856ad364e35_10.0.22621.1_none_97b6d41c3b3b7e51",
        "/usr/share/fonts",
        "/usr/local/share/fonts",
    ]
    for d in font_dirs:
        for name in _FONT_CANDIDATES:
            path = os.path.join(d, name)
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except (OSError, IOError):
                    pass
    return ImageFont.load_default()


class TextRenderer:
    def __init__(self):
        self._font_cache = {}

    def _get_font(self, size: int = 40):
        if size not in self._font_cache:
            self._font_cache[size] = _find_font(size)
        return self._font_cache[size]

    def draw_arabic_text(self, img_bgr, text, position, font_size=40,
                         fill=(255, 255, 255), shadow_offset=(0, 0)):
        font = self._get_font(font_size)
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            reshaped = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped)
        except Exception:
            bidi_text = text
        if shadow_offset != (0, 0):
            draw.text(
                (position[0] + shadow_offset[0], position[1] + shadow_offset[1]),
                bidi_text, font=font, fill=(0, 0, 0)
            )
        draw.text(position, bidi_text, font=font, fill=fill)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def text_size(self, text, font_size=40):
        font = self._get_font(font_size)
        try:
            reshaped = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped)
        except Exception:
            bidi_text = text
        left, top, right, bottom = font.getbbox(bidi_text)
        return right - left, bottom - top
