import cv2
import numpy as np
from .skeleton import (
    parse_landmarks, is_valid_hand,
    draw_skeleton,
    POSE_CONNECTIONS, HAND_CONNECTIONS,
    BG_COLOR, POSE_LINE_COLOR, POSE_JOINT_COLOR,
    LHAND_LINE_COLOR, LHAND_JOINT_COLOR,
    RHAND_LINE_COLOR, RHAND_JOINT_COLOR,
    TEXT_COLOR,
)
from .text_renderer import TextRenderer


def _wrist_hand_aligned(pose_wrist, hand_landmark0, max_dist: float = 0.2) -> bool:
    """The pose model's wrist and the hand model's own landmark-0 (its
    wrist/palm-base) are two independent estimates of the same physical
    point, and should nearly coincide. Cubic-spline interpolation can
    transiently make them diverge sharply for a single frame around a
    hand-visibility transition, which — since the connector line is drawn
    from one to the other — shows up as a long stray line shooting across
    the canvas. Skip the line when the two disagree implausibly."""
    dx = pose_wrist[0] - hand_landmark0[0]
    dy = pose_wrist[1] - hand_landmark0[1]
    return (dx * dx + dy * dy) ** 0.5 <= max_dist


class LandmarkRenderer:
    def __init__(self, width=640, height=640, fps=30):
        self.w = width
        self.h = height
        self.fps = fps
        self.margin = 40
        self._text_renderer = TextRenderer()
        self._blank_frame = None

    def _scale(self, x, y, z=0):
        px = int(np.clip(x, 0, 1) * (self.w - 2 * self.margin) + self.margin)
        py = int(np.clip(y, 0, 1) * (self.h - 2 * self.margin) + self.margin)
        return px, py

    def render_frame(self, frame_258, word_text="", overlay_frame=None, alpha=1.0):
        img = np.full((self.h, self.w, 3), BG_COLOR, dtype=np.uint8)
        pose, lhand, rhand = parse_landmarks(frame_258)

        draw_skeleton(img, pose, POSE_CONNECTIONS,
                      POSE_LINE_COLOR, POSE_JOINT_COLOR,
                      vis_check=True, scale_fn=self._scale)

        if is_valid_hand(lhand):
            draw_skeleton(img, lhand, HAND_CONNECTIONS,
                          LHAND_LINE_COLOR, LHAND_JOINT_COLOR,
                          line_thick=2, joint_radius=3, scale_fn=self._scale)
            wrist_idx = 15
            if pose[wrist_idx, 3] > 0.1 and _wrist_hand_aligned(pose[wrist_idx], lhand[0]):
                wx, wy = self._scale(pose[wrist_idx, 0], pose[wrist_idx, 1])
                hx, hy = self._scale(lhand[0, 0], lhand[0, 1])
                cv2.line(img, (wx, wy), (hx, hy), LHAND_LINE_COLOR, 2, cv2.LINE_AA)

        if is_valid_hand(rhand):
            draw_skeleton(img, rhand, HAND_CONNECTIONS,
                          RHAND_LINE_COLOR, RHAND_JOINT_COLOR,
                          line_thick=2, joint_radius=3, scale_fn=self._scale)
            wrist_idx = 16
            if pose[wrist_idx, 3] > 0.1 and _wrist_hand_aligned(pose[wrist_idx], rhand[0]):
                wx, wy = self._scale(pose[wrist_idx, 0], pose[wrist_idx, 1])
                hx, hy = self._scale(rhand[0, 0], rhand[0, 1])
                cv2.line(img, (wx, wy), (hx, hy), RHAND_LINE_COLOR, 2, cv2.LINE_AA)

        if overlay_frame is not None and alpha < 1.0:
            img = cv2.addWeighted(img, alpha, overlay_frame, 1.0 - alpha, 0)

        if word_text:
            img = self._text_renderer.draw_arabic_text(
                img, word_text, (50, self.h - 80), font_size=40,
                fill=TEXT_COLOR, shadow_offset=(2, 2)
            )
        return img

    def render_word_frames(self, landmark_sequence, word_text, transition_frames=8):
        frames = []
        for i in range(len(landmark_sequence)):
            frame = self.render_frame(landmark_sequence[i], word_text)
            frames.append(frame)
        if transition_frames > 0 and frames:
            blank = np.full((self.h, self.w, 3), BG_COLOR, dtype=np.uint8)
            if word_text:
                blank = self._text_renderer.draw_arabic_text(
                    blank, word_text, (50, self.h - 80), font_size=40,
                    fill=TEXT_COLOR, shadow_offset=(2, 2)
                )
            for _ in range(transition_frames):
                frames.append(blank.copy())
        return frames

    def render_unknown_word(self, word_text, num_frames=30):
        blank = np.full((self.h, self.w, 3), BG_COLOR, dtype=np.uint8)
        text_x = self.w // 2 - 100
        text_y = self.h // 2 - 40
        blank = self._text_renderer.draw_arabic_text(
            blank, word_text, (text_x, text_y), font_size=70,
            fill=TEXT_COLOR, shadow_offset=(3, 3)
        )
        blank = self._text_renderer.draw_arabic_text(
            blank, "(not in dictionary)", (text_x, text_y + 80), font_size=30,
            fill=(150, 150, 150), shadow_offset=(2, 2)
        )
        return [blank.copy() for _ in range(num_frames)]

    def create_blank(self, word_text=""):
        blank = np.full((self.h, self.w, 3), BG_COLOR, dtype=np.uint8)
        if word_text:
            blank = self._text_renderer.draw_arabic_text(
                blank, word_text, (50, self.h - 80), font_size=40,
                fill=TEXT_COLOR, shadow_offset=(2, 2)
            )
        return blank
