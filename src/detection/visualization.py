"""Visualization of detection results.

The detector produces structured data; the visualizer decides how those
results are drawn. Only detection information is drawn (no depth, risk,
navigation, or haptic content at this stage).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.detection.detection_result import Detection, DetectionResult

#: BGR colors used to distinguish a few common class groups.
COLORS = {
    "person": (60, 180, 75),        # green
    "bicycle": (0, 120, 255),       # orange-ish
    "car": (0, 69, 255),            # red-ish
    "motorcycle": (0, 140, 255),
    "bus": (255, 120, 0),
    "truck": (255, 80, 0),
    "traffic light": (255, 255, 0),
    "stop sign": (255, 0, 255),
}
DEFAULT_COLOR = (200, 200, 40)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _color_for(class_name: str) -> tuple:
    return COLORS.get(class_name, DEFAULT_COLOR)


def annotate(frame: np.ndarray, result: DetectionResult, thickness: int = 2) -> np.ndarray:
    """Return an annotated copy of the frame with boxes and labels.

    The input frame is never modified: drawing happens on a copy, and
    the copy is BGR (ready for OpenCV or conversion by the caller).
    """
    annotated = frame.copy()
    for detection in result.detections:
        _draw_detection(annotated, detection, thickness)
    return annotated


def _draw_detection(image: np.ndarray, detection: Detection, thickness: int = 2) -> None:
    """Draw one bounding box plus its class/confidence label."""
    color = _color_for(detection.class_name)
    x1, y1, x2, y2 = (int(round(v)) for v in detection.bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    label = f"{detection.class_name} {detection.confidence:.2f}"
    label_size, baseline = cv2.getTextSize(label, FONT, 0.5, 1)
    top = y1 - label_size[1] - 6
    if top < 0:  # place the label inside the box when it would clip
        top = y1 + 2
    left = max(x1, 0)
    cv2.rectangle(
        image, (left, top), (left + label_size[0] + 4, top + label_size[1] + 6),
        color, thickness=-1,
    )
    cv2.putText(
        image, label, (left + 2, top + label_size[1] + 2),
        FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )


class DetectionVisualizer:
    """Stateless helper that renders :class:`DetectionResult` onto frames."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def render(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Return an annotated frame, or the original when disabled."""
        if not self.enabled:
            return frame
        return annotate(frame, result)

    @staticmethod
    def position_summary(result: DetectionResult) -> str:
        """One-line text summary of detections and image positions."""
        if not result.detections:
            return "No objects detected"
        parts = [
            f"{d.class_name} {d.confidence:.2f} ({d.position.value})"
            for d in result.detections
        ]
        return "; ".join(parts)
