import cv2
import numpy as np

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

BG_COLOR = (20, 20, 40)
POSE_LINE_COLOR = (200, 180, 50)
POSE_JOINT_COLOR = (255, 255, 255)
LHAND_LINE_COLOR = (100, 255, 100)
LHAND_JOINT_COLOR = (150, 255, 150)
RHAND_LINE_COLOR = (100, 165, 255)
RHAND_JOINT_COLOR = (150, 200, 255)
TEXT_COLOR = (255, 255, 255)
TEXT_SHADOW = (0, 0, 0)


def parse_landmarks(frame_258):
    pose = frame_258[:132].reshape(33, 4)
    lhand = frame_258[132:195].reshape(21, 3)
    rhand = frame_258[195:258].reshape(21, 3)
    return pose, lhand, rhand


def is_valid_hand(hand):
    return np.any(hand[:, :2] > 0.01)


def draw_skeleton(img, points, connections, line_color, joint_color,
                  line_thick=2, joint_radius=4, vis_check=False,
                  scale_fn=None):
    if scale_fn is None:
        return
    for i, j in connections:
        if vis_check and (points[i, 3] < 0.1 or points[j, 3] < 0.1):
            continue
        x1, y1 = scale_fn(points[i, 0], points[i, 1], points[i, 2] if points.shape[1] > 2 else 0)
        x2, y2 = scale_fn(points[j, 0], points[j, 1], points[j, 2] if points.shape[1] > 2 else 0)
        cv2.line(img, (x1, y1), (x2, y2), line_color, line_thick, cv2.LINE_AA)
    for idx in range(len(points)):
        if vis_check and points[idx, 3] < 0.1:
            continue
        z = points[idx, 2] if points.shape[1] > 2 else 0
        x, y = scale_fn(points[idx, 0], points[idx, 1], z)
        radius = max(2, int(joint_radius * (1.0 - z * 0.3)))
        cv2.circle(img, (x, y), radius, joint_color, -1, cv2.LINE_AA)
