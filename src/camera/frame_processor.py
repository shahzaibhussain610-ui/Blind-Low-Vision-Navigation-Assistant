"""Basic frame preprocessing: validation, resizing, skipping.

No AI processing happens here. Frames stay in OpenCV BGR format;
RGB conversion is available on demand for display only.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from src.logging.logger import get_logger

logger = get_logger("frame_processor")


class FrameProcessor:
    """Reusable preprocessing pipeline for incoming frames."""

    def __init__(
        self,
        max_width: int = 640,
        target_size: Optional[Tuple[int, int]] = None,
        frame_skip: int = 0,
        preserve_aspect_ratio: bool = True,
        allow_upscale: bool = False,
    ) -> None:
        """Configure preprocessing.

        Args:
            max_width: Frames wider than this are downscaled
                (aspect-ratio preserving) unless ``target_size`` is set.
            target_size: Optional exact ``(width, height)`` output.
            frame_skip: 0 = process every frame, 1 = process every
                second frame, 2 = process every third frame, ...
            preserve_aspect_ratio: Keep aspect ratio when resizing.
            allow_upscale: When False (default), frames smaller than the
                target are left untouched.
        """
        self.max_width = int(max_width)
        self.target_size = target_size
        self.frame_skip = max(int(frame_skip), 0)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.allow_upscale = allow_upscale

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def validate(frame) -> bool:
        """True when the frame is a usable image (enters AI pipeline)."""
        return (
            isinstance(frame, np.ndarray)
            and frame.ndim >= 2
            and frame.size > 0
            and frame.shape[0] > 0
            and frame.shape[1] > 0
        )

    # ------------------------------------------------------------------ #
    # Resizing
    # ------------------------------------------------------------------ #
    def compute_target(self, width: int, height: int) -> Tuple[int, int]:
        """Compute the output size for a frame of the given dimensions."""
        if self.target_size:
            return self.target_size
        if width <= self.max_width:
            return width, height  # never upscale low-resolution frames
        scale = self.max_width / width
        return self.max_width, max(int(round(height * scale)), 1)

    def resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize a frame according to the configured policy.

        Returns the original array (no copy) when no resize is needed.
        """
        if not self.validate(frame):
            return frame
        height, width = frame.shape[:2]
        target_w, target_h = self.compute_target(width, height)
        if (target_w, target_h) == (width, height):
            return frame
        if self.preserve_aspect_ratio and self.target_size is None:
            return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return cv2.resize(
            frame, (target_w, target_h),
            interpolation=cv2.INTER_AREA if target_w < width else cv2.INTER_LINEAR,
        )

    # ------------------------------------------------------------------ #
    # Frame skipping
    # ------------------------------------------------------------------ #
    def should_process(self, frame_number: int) -> bool:
        """Decide whether a frame should pass to the pipeline.

        With ``frame_skip = N``, every (N+1)-th frame is processed:
        ``0`` → every frame, ``1`` → every 2nd frame, ``2`` → every 3rd.
        """
        if self.frame_skip <= 0:
            return True
        return frame_number % (self.frame_skip + 1) == 0

    def process(self, frame: np.ndarray, frame_number: int) -> Optional[np.ndarray]:
        """Full preprocessing step: validate → skip → resize.

        Returns:
            The (possibly resized) BGR frame, or ``None`` when the frame
            is invalid or skipped.
        """
        if not self.validate(frame):
            logger.warning("Invalid frame rejected (number=%d)", frame_number)
            return None
        if not self.should_process(frame_number):
            return None
        return self.resize(frame)

    # ------------------------------------------------------------------ #
    # Color handling
    # ------------------------------------------------------------------ #
    @staticmethod
    def to_rgb(frame: np.ndarray) -> np.ndarray:
        """Convert an OpenCV BGR frame to RGB for display purposes."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
