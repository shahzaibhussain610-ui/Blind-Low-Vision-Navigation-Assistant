"""Object-detection package: YOLO11 detector, results, class filter, visuals.

Public API::

    from src.detection import YOLODetector, ClassFilter, DetectionResult

The detector consumes Part 2 :class:`FramePacket` objects and produces
project-specific :class:`DetectionResult` objects.
"""

from src.detection.class_filter import NAVIGATION_CLASSES, ClassFilter
from src.detection.detection_result import (
    Detection,
    DetectionResult,
    ImagePosition,
    classify_position,
)
from src.detection.detector import InferenceError, ModelLoadError, YOLODetector
from src.detection.visualization import DetectionVisualizer, annotate

__all__ = [
    "NAVIGATION_CLASSES",
    "ClassFilter",
    "Detection",
    "DetectionResult",
    "ImagePosition",
    "classify_position",
    "InferenceError",
    "ModelLoadError",
    "YOLODetector",
    "DetectionVisualizer",
    "annotate",
]
