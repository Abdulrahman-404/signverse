import os
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime, timezone

_N_POSE  = 33
_N_HAND  = 21
_POSE_DIM = _N_POSE * 4
_HAND_DIM = _N_HAND * 3


def _parse_frame(frame_258: np.ndarray):
    pose  = frame_258[:_POSE_DIM].reshape(_N_POSE, 4)
    lhand = frame_258[_POSE_DIM : _POSE_DIM + _HAND_DIM].reshape(_N_HAND, 3)
    rhand = frame_258[_POSE_DIM + _HAND_DIM :].reshape(_N_HAND, 3)
    return pose, lhand, rhand


def _fmt(value: float) -> str:
    return f"{value:.4f}"


class SiGMLEncoder:
    def __init__(self, fps: int = 30):
        self.fps = fps
        self._signs: list[dict] = []

    def add_sign(self, gloss: str, landmark_sequence: np.ndarray):
        if landmark_sequence.ndim != 2 or landmark_sequence.shape[1] != 258:
            raise ValueError(
                f"landmark_sequence must have shape (n_frames, 258), "
                f"got {landmark_sequence.shape}"
            )
        self._signs.append({"gloss": gloss, "seq": landmark_sequence})

    def add_unknown(self, word: str):
        placeholder = np.zeros((1, 258), dtype=np.float32)
        self._signs.append({"gloss": word, "seq": placeholder, "oov": True})

    def build_xml(self) -> ET.Element:
        root = ET.Element("sigml")
        root.set("version", "3.0")
        root.set("format", "ArSL-SiGML-Keyframe")
        root.set("reference", "https://www.sign-lang.uni-hamburg.de/sigml")
        root.set("generator", "ArabicSignLanguageTranslator/SiGMLEncoder")
        root.set("created", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        root.set("fps", str(self.fps))
        root.set("total_signs", str(len(self._signs)))

        for entry in self._signs:
            sign_elem = self._build_sign_element(entry)
            root.append(sign_elem)

        return root

    def save(self, output_path: str) -> str:
        root = self.build_xml()
        xml_str = self._pretty_print(root)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(xml_str)
        abs_path = os.path.abspath(output_path)
        n_signs = len(self._signs)
        print(f"SiGML saved: {abs_path}  ({n_signs} signs)")
        return abs_path

    def clear(self):
        self._signs.clear()

    def _build_sign_element(self, entry: dict) -> ET.Element:
        gloss   = entry["gloss"]
        seq     = entry["seq"]
        is_oov  = entry.get("oov", False)
        n_frames = seq.shape[0]

        sign_el = ET.Element("hamgestural_sign")
        sign_el.set("gloss",        gloss)
        sign_el.set("arabic_gloss", gloss)
        sign_el.set("frame_count",  str(n_frames))
        sign_el.set("fps",          str(self.fps))
        sign_el.set("duration_ms",  str(round(n_frames / self.fps * 1000)))
        if is_oov:
            sign_el.set("status", "out_of_vocabulary")

        manual_el = ET.SubElement(sign_el, "sign_manual")

        ref_el = ET.SubElement(manual_el, "reference_frame")
        ref_el.set("landmark", "shoulder_midpoint")
        ref_el.set("description",
            "Coordinates are MediaPipe Holistic normalised (0–1 range). "
            "Pose landmark 11 = left shoulder, 12 = right shoulder."
        )

        kf_seq_el = ET.SubElement(manual_el, "keyframe_sequence")

        for fi in range(n_frames):
            pose, lhand, rhand = _parse_frame(seq[fi])
            frame_el = ET.SubElement(kf_seq_el, "frame")
            frame_el.set("index", str(fi))
            frame_el.set("time_ms", str(round(fi / self.fps * 1000)))

            pose_el = ET.SubElement(frame_el, "pose")
            for pi in range(_N_POSE):
                pt = ET.SubElement(pose_el, "point")
                pt.set("id",  str(pi))
                pt.set("x",   _fmt(float(pose[pi, 0])))
                pt.set("y",   _fmt(float(pose[pi, 1])))
                pt.set("z",   _fmt(float(pose[pi, 2])))
                pt.set("vis", _fmt(float(pose[pi, 3])))

            lhand_el = ET.SubElement(frame_el, "left_hand")
            for hi in range(_N_HAND):
                pt = ET.SubElement(lhand_el, "point")
                pt.set("id", str(hi))
                pt.set("x",  _fmt(float(lhand[hi, 0])))
                pt.set("y",  _fmt(float(lhand[hi, 1])))
                pt.set("z",  _fmt(float(lhand[hi, 2])))

            rhand_el = ET.SubElement(frame_el, "right_hand")
            for hi in range(_N_HAND):
                pt = ET.SubElement(rhand_el, "point")
                pt.set("id", str(hi))
                pt.set("x",  _fmt(float(rhand[hi, 0])))
                pt.set("y",  _fmt(float(rhand[hi, 1])))
                pt.set("z",  _fmt(float(rhand[hi, 2])))

        return sign_el

    @staticmethod
    def _pretty_print(root: ET.Element) -> str:
        raw = ET.tostring(root, encoding="unicode")
        parsed = minidom.parseString(raw)
        pretty = parsed.toprettyxml(indent="  ", encoding=None)
        if pretty.startswith("<?xml"):
            lines = pretty.split("\n", 1)
            pretty = '<?xml version="1.0" encoding="UTF-8"?>\n' + lines[1]
        return pretty
