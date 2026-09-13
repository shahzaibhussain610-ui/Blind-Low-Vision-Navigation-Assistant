"""Relative depth estimation (Part 4).

Two backends:
  * ``heuristic`` — fast, CPU-friendly proxy from bounding-box geometry
    (box area + vertical position, ground-plane assumption). Produces a
    RELATIVE depth score in [0, 1] (1 = nearest). This is NOT a metric
    distance in meters.
  * ``midas`` — real monocular depth (MiDaS small via torch.hub) when
    available and explicitly selected in the configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from src.detection.detection_result import Detection
from src.logging.logger import get_logger

logger = get_logger("depth")


class DepthError(Exception):
    """Raised when the depth backend fails."""


@dataclass
class ObjectDepth:
    """Relative depth information for one detection.

    ``relative_depth`` is a score in [0, 1] (higher = closer). It is a
    RELATIVE estimate, never a metric distance in meters.
    """

    relative_depth: float
    depth_label: str
    backend: str = "heuristic"

    def to_dict(self) -> dict:
        return {
            "relative_depth": round(self.relative_depth, 3),
            "depth_label": self.depth_label,
            "backend": self.backend,
        }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class RelativeDepthEstimator:
    """Estimates relative depth for detected objects in a frame."""

    def __init__(
        self,
        mode: str = "heuristic",
        near_threshold: float = 0.62,
        far_threshold: float = 0.35,
    ) -> None:
        self.mode = mode
        self.near_threshold = near_threshold
        self.far_threshold = far_threshold
        self._midas = None
        self._midas_transform = None
        self._last_frame: Optional[np.ndarray] = None
        if mode not in ("heuristic", "midas"):
            logger.warning("Unknown depth mode '%s'; using 'heuristic'", mode)
            self.mode = "heuristic"

    # ------------------------------------------------------------------ #
    def label_for(self, relative_depth: float) -> str:
        """NEAR / MIDDLE / FAR label for a relative depth score."""
        if relative_depth >= self.near_threshold:
            return "NEAR"
        if relative_depth <= self.far_threshold:
            return "FAR"
        return "MIDDLE"

    def set_frame(self, frame: np.ndarray) -> None:
        """Provide the current frame (required by the MiDaS backend)."""
        self._last_frame = frame

    def estimate_one(self, detection: Detection) -> ObjectDepth:
        """Estimate relative depth for a single detection."""
        if self.mode == "midas":
            depth = self._estimate_midas(detection)
            if depth is not None:
                return depth
        return self._estimate_heuristic(detection)

    def estimate_for_detections(self, frame: np.ndarray, detections: list) -> dict:
        """Estimate depth for every detection; keyed by identity (index)."""
        self.set_frame(frame)
        return {index: self.estimate_one(d) for index, d in enumerate(detections)}

    # ------------------------------------------------------------------ #
    # Heuristic backend (default, CPU-friendly)
    # ------------------------------------------------------------------ #
    def _estimate_heuristic(self, detection: Detection) -> ObjectDepth:
        """Relative depth proxy from bounding-box geometry.

        Documented, approximate assumptions:
          * Larger boxes usually mean closer objects (area cue).
          * A box bottom lower in the image usually means closer
            (ground-plane cue).
        """
        height, width = detection.frame_height, detection.frame_width
        if height <= 0 or width <= 0:
            return ObjectDepth(0.5, self.label_for(0.5), "heuristic")
        area_ratio = (detection.width * detection.height) / float(width * height)
        bottom_ratio = detection.y2 / float(height)
        proximity = _clamp01(0.65 * _clamp01(area_ratio * 4.0) + 0.35 * bottom_ratio)
        return ObjectDepth(proximity, self.label_for(proximity), "heuristic")

    # ------------------------------------------------------------------ #
    # MiDaS backend (optional; used only when mode == "midas")
    # ------------------------------------------------------------------ #
    def _load_midas(self) -> bool:
        """Load MiDaS small lazily. Returns False when unavailable."""
        if self._midas is not None:
            return True
        try:
            import torch

            self._midas = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
            )
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self._midas_transform = transforms.small_transform
            self._midas.eval()
            logger.info("MiDaS small loaded for monocular depth")
            return True
        except Exception as exc:
            logger.error("MiDaS could not be loaded: %s", exc)
            self._midas = None
            return False

    def _estimate_midas(self, detection: Detection) -> Optional[ObjectDepth]:
        """MiDaS-based relative depth, or None when unavailable."""
        if self._last_frame is None or not self._load_midas():
            return None
        try:
            import cv2
            import torch

            rgb = cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2RGB)
            batch = self._midas_transform(rgb)
            with torch.no_grad():
                prediction = self._midas(batch)
                depth_map = (
                    torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=self._last_frame.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze()
                    .numpy()
                )
        except Exception as exc:
            logger.error("MiDaS inference failure: %s", exc)
            return None
        x1, y1, x2, y2 = (int(v) for v in detection.bbox)
        region = depth_map[max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)]
        if region.size == 0:
            return None
        dmin, dmax = float(depth_map.min()), float(depth_map.max())
        if dmax - dmin <= 0:
            return None
        # MiDaS output: larger values mean closer to the camera.
        proximity = _clamp01((float(region.mean()) - dmin) / (dmax - dmin))
        return ObjectDepth(proximity, self.label_for(proximity), "midas")
