"""Common frame representation shared by all input sources.

Every module downstream (detection, depth, tracking, ...) receives a
:class:`FramePacket`, regardless of whether the frame came from a webcam
or a prerecorded video file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SourceType(str, Enum):
    """Origin of a frame."""

    WEBCAM = "webcam"
    VIDEO_FILE = "video_file"


class InputState(str, Enum):
    """State machine for the active input source."""

    NO_INPUT = "NO_INPUT"
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    END_OF_VIDEO = "END_OF_VIDEO"
    ERROR = "ERROR"


@dataclass
class FramePacket:
    """A single frame plus the metadata needed by future pipeline stages.

    Attributes:
        frame: The raw image in **OpenCV BGR** channel order.
        frame_number: Sequential counter maintained by the source
            (0, 1, 2, ...). Never derived from OpenCV's internal video
            position, because webcams do not provide reliable numbering.
        timestamp: Seconds. For prerecorded video this is the video
            timestamp (``CAP_PROP_POS_MSEC``) when reliably available;
            for webcam input it is the acquisition time
            (``time.perf_counter()``).
        source_type: :class:`SourceType` of the originating input.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    frame: np.ndarray
    frame_number: int
    timestamp: float
    source_type: SourceType
    width: int = 0
    height: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame is not None and self.frame.ndim >= 2:
            self.height, self.width = self.frame.shape[:2]

    @property
    def is_valid(self) -> bool:
        """True when the frame is a non-empty 2-D (or 3-D) uint8 image."""
        return FramePacket.validate(self.frame)

    @staticmethod
    def validate(frame) -> bool:
        """Return True when ``frame`` can enter the AI pipeline."""
        return (
            isinstance(frame, np.ndarray)
            and frame.ndim >= 2
            and frame.size > 0
            and frame.shape[0] > 0
            and frame.shape[1] > 0
        )

    def to_rgb(self) -> np.ndarray:
        """Return an RGB copy for display (Streamlit/matplotlib).

        OpenCV works in BGR; conversion happens only here, on demand,
        so internal processing can stay in the native BGR format.
        """
        import cv2  # local import keeps module import cost low

        return cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

    def to_dict(self) -> dict:
        """JSON-friendly summary without the raw image data."""
        return {
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 4),
            "source_type": self.source_type.value,
            "width": self.width,
            "height": self.height,
        }
