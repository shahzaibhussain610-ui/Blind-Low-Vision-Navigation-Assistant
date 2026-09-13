"""Camera/input package: webcam and prerecorded video frame sources.

Public API for Part 3+::

    from src.camera import CameraManager, ManagerConfig, FramePacket

Future AI modules consume :class:`FramePacket` and never need to know
whether the frame originated from a webcam or a video file.
"""

from src.camera.camera_manager import (
    CameraManager,
    FPSMeter,
    LatencyTracker,
    ManagerConfig,
)
from src.camera.frame import FramePacket, InputState, SourceType
from src.camera.frame_processor import FrameProcessor
from src.camera.video_source import (
    SourceInfo,
    VideoFileSource,
    VideoInputError,
    VideoSource,
    WebcamSource,
)

__all__ = [
    "CameraManager",
    "FPSMeter",
    "LatencyTracker",
    "ManagerConfig",
    "FramePacket",
    "InputState",
    "SourceType",
    "FrameProcessor",
    "SourceInfo",
    "VideoFileSource",
    "VideoInputError",
    "VideoSource",
    "WebcamSource",
]
