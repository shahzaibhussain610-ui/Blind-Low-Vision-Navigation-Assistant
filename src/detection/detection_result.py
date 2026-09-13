"""Project-specific detection result structures.

Future modules (tracking, spatial analysis, risk) receive these
dataclasses and never Ultralytics result objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np


class ImagePosition(str, Enum):
    """Image-space horizontal localization of an object.

    This describes WHERE the object appears in the image only.
    It is NOT a navigation instruction (navigation belongs to Part 8).
    """

    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"


def classify_position(
    center_x: float, frame_width: int, tolerance: float = 0.15
) -> ImagePosition:
    """Classify horizontal image position from a center-x coordinate.

    Args:
        center_x: Object center x in pixels.
        frame_width: Total frame width in pixels.
        tolerance: Fraction of frame width treated as the center zone.
    """
    if frame_width <= 0:
        return ImagePosition.CENTER
    ratio = center_x / frame_width
    if ratio < 0.5 - tolerance:
        return ImagePosition.LEFT
    if ratio > 0.5 + tolerance:
        return ImagePosition.RIGHT
    return ImagePosition.CENTER


@dataclass
class Detection:
    """One detected object in one frame (image-space only).

    Attributes:
        class_id: Model class index.
        class_name: Human-readable class label from the model.
        confidence: Detection confidence in [0, 1].
        x1, y1, x2, y2: Bounding box in image coordinates
            (x1=left, y1=top, x2=right, y2=bottom).
        frame_number: Sequential frame counter from the input source.
        timestamp: Frame timestamp (seconds, from Part 2).
    """

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    frame_number: int
    timestamp: float
    frame_width: int = 0
    frame_height: int = 0
    track_id: int = -1  # Part 5: persistent tracking id (-1 = untracked)

    # ---- derived geometry ------------------------------------------------ #
    @property
    def center_x(self) -> float:
        """Horizontal center of the bounding box."""
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        """Vertical center of the bounding box."""
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        """Bounding-box width in pixels (NOT physical distance)."""
        return max(self.x2 - self.x1, 0.0)

    @property
    def height(self) -> float:
        """Bounding-box height in pixels (NOT physical distance)."""
        return max(self.y2 - self.y1, 0.0)

    @property
    def bbox(self) -> tuple:
        """(x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def position(self) -> ImagePosition:
        """Image-space LEFT/CENTER/RIGHT classification."""
        return classify_position(self.center_x, self.frame_width)

    def clamp_to_frame(self, width: int, height: int) -> "Detection":
        """Return a copy with coordinates clamped to the frame bounds."""
        return Detection(
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
            x1=min(max(self.x1, 0.0), width),
            y1=min(max(self.y1, 0.0), height),
            x2=min(max(self.x2, 0.0), width),
            y2=min(max(self.y2, 0.0), height),
            frame_number=self.frame_number,
            timestamp=self.timestamp,
            frame_width=width,
            frame_height=height,
            track_id=self.track_id,
        )

    def is_valid(self, width: int = 0, height: int = 0) -> bool:
        """Validate box geometry (and frame bounds when provided)."""
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            return False
        if not (0.0 <= self.confidence <= 1.0):
            return False
        if width > 0:
            if self.x1 < -1 or self.x2 > width + 1:
                return False
        if height > 0:
            if self.y1 < -1 or self.y2 > height + 1:
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 1) for v in self.bbox],
            "center": [round(self.center_x, 1), round(self.center_y, 1)],
            "size": [round(self.width, 1), round(self.height, 1)],
            "position": self.position.value,
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 4),
            "track_id": self.track_id,
        }


@dataclass
class DetectionResult:
    """All detections for one frame plus performance measurements.

    Attributes:
        frame_number: Sequential frame counter from the input source.
        timestamp: Frame timestamp (seconds, from Part 2).
        detections: List of :class:`Detection` objects (may be empty).
        inference_time_ms: Measured YOLO inference duration (ms).
        inference_fps: 1000 / inference_time_ms (0.0 when unavailable).
        model_name: Name of the model that produced these results.
        device: Device used for inference ("cpu" or "cuda").
        frame_width, frame_height: Dimensions of the source frame.
    """

    frame_number: int
    timestamp: float
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    inference_fps: float = 0.0
    model_name: str = ""
    device: str = "cpu"
    frame_width: int = 0
    frame_height: int = 0

    @property
    def count(self) -> int:
        """Number of valid detections in this frame."""
        return len(self.detections)

    def by_class(self, class_name: str) -> List[Detection]:
        """Return detections matching a class name."""
        return [d for d in self.detections if d.class_name == class_name]

    def to_dicts(self) -> list:
        """JSON-friendly list of all detections."""
        return [d.to_dict() for d in self.detections]
