"""Input source implementations: webcam and prerecorded video files.

The rest of the application never touches ``cv2.VideoCapture`` directly;
it goes through :class:`VideoSource` subclasses via the
:class:`~src.camera.camera_manager.CameraManager`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.camera.frame import SourceType
from src.logging.logger import get_logger

logger = get_logger("camera")


class VideoInputError(Exception):
    """Raised when an input source cannot be opened or read."""


@dataclass
class SourceInfo:
    """Metadata describing an opened (or failed) input source."""

    source_type: SourceType
    name: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    duration_seconds: float = 0.0
    opened: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = {
            "source_type": self.source_type.value,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "frame_count": self.frame_count,
            "duration_seconds": round(self.duration_seconds, 2),
            "opened": self.opened,
        }
        data.update(self.extra)
        return data


class VideoSource(ABC):
    """Common interface for all frame input sources."""

    source_type: SourceType

    @abstractmethod
    def open(self) -> bool:
        """Open the source. Returns True on success."""

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Read one BGR frame, or None when no frame is available."""

    @abstractmethod
    def release(self) -> None:
        """Release the underlying capture resource."""

    @abstractmethod
    def is_open(self) -> bool:
        """True while the capture handle is open."""

    @abstractmethod
    def get_info(self) -> SourceInfo:
        """Return current metadata for this source."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable source identifier for logs and UI."""


class WebcamSource(VideoSource):
    """Laptop/USB webcam source.

    Args:
        index: Camera index (0, 1, 2, ...). Camera 0 is NOT assumed to exist.
        width: Preferred capture width (0 = driver default).
        height: Preferred capture height (0 = driver default).
        fps: Preferred capture FPS (0 = driver default).
        backend: OpenCV capture backend name (``"auto"``, ``"dshow"``,
            ``"msmf"``, ``"any"``). ``"auto"`` tries DirectShow first on
            Windows (faster startup), then falls back to the default.
    """

    source_type = SourceType.WEBCAM

    #: Map of friendly backend names to OpenCV constants.
    BACKENDS = {
        "auto": None,
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: float = 15.0,
        backend: str = "auto",
    ) -> None:
        self.index = int(index)
        self.requested_width = int(width)
        self.requested_height = int(height)
        self.requested_fps = float(fps)
        self.backend_name = backend
        self._capture: Optional[cv2.VideoCapture] = None

    @property
    def name(self) -> str:
        return f"Webcam {self.index}"

    def _ordered_backends(self):
        backend = self.BACKENDS.get(self.backend_name, None)
        if backend is not None:
            return [backend]
        # "auto": prefer DirectShow on Windows (quick start), then default.
        if hasattr(cv2, "CAP_DSHOW"):
            return [cv2.CAP_DSHOW, cv2.CAP_ANY]
        return [cv2.CAP_ANY]

    def open(self) -> bool:
        """Open the webcam with the requested properties.

        Returns:
            True when the camera opened successfully.

        Raises:
            VideoInputError: When the camera cannot be opened.
        """
        if self.is_open():
            return True
        last_error: str = "no backend attempted"
        for backend in self._ordered_backends():
            try:
                capture = (
                    cv2.VideoCapture(self.index, backend)
                    if backend is not None
                    else cv2.VideoCapture(self.index)
                )
            except Exception as exc:  # OpenCV internal failure
                last_error = f"{exc.__class__.__name__}: {exc}"
                logger.debug("Webcam %s open failed (%s)", self.index, last_error)
                continue
            if capture is not None and capture.isOpened():
                self._capture = capture
                self._apply_properties()
                logger.info(
                    "Webcam initialized: index=%d backend=%s requested=%dx%d@%.1ffps",
                    self.index, int(backend or 0),
                    self.requested_width, self.requested_height, self.requested_fps,
                )
                return True
            capture.release()
            last_error = "capture could not be opened"
        raise VideoInputError(
            f"Unable to open webcam {self.index}. Last error: {last_error}"
        )

    def _apply_properties(self) -> None:
        """Best-effort application of requested capture properties."""
        if self._capture is None:
            return
        if self.requested_width > 0:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        if self.requested_height > 0:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        if self.requested_fps > 0:
            self._capture.set(cv2.CAP_PROP_FPS, self.requested_fps)

    def read(self) -> Optional[np.ndarray]:
        """Read one BGR frame. Returns None when read fails."""
        if not self.is_open():
            return None
        try:
            ok, frame = self._capture.read()
        except Exception as exc:
            logger.error("Webcam %s read failure: %s", self.index, exc)
            return None
        if not ok or frame is None:
            return None
        return frame

    def release(self) -> None:
        """Release the camera handle so it is available to other apps."""
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception as exc:
                logger.warning("Error while releasing webcam %s: %s", self.index, exc)
            self._capture = None
            logger.info("Webcam stopped and released: index=%d", self.index)

    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def get_info(self) -> SourceInfo:
        info = SourceInfo(
            source_type=self.source_type,
            name=self.name,
            opened=self.is_open(),
            extra={"index": self.index, "backend": self.backend_name},
        )
        if self.is_open():
            info.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            info.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info.fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        else:
            info.width = self.requested_width
            info.height = self.requested_height
            info.fps = self.requested_fps
        return info


class VideoFileSource(VideoSource):
    """Prerecorded local video-file source.

    Processing stays fully local; files are never uploaded anywhere.
    """

    source_type = SourceType.VIDEO_FILE

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._capture: Optional[cv2.VideoCapture] = None
        self._info: Optional[SourceInfo] = None

    @property
    def name(self) -> str:
        return self.path.name

    def open(self) -> bool:
        """Open and validate the video file.

        Raises:
            VideoInputError: When the file is missing, unreadable, or
                cannot be decoded by OpenCV.
        """
        if self.is_open():
            return True
        if not self.path.is_file():
            raise VideoInputError(f"Video file does not exist: {self.path}")
        try:
            capture = cv2.VideoCapture(str(self.path))
        except Exception as exc:
            raise VideoInputError(
                f"OpenCV failed while opening '{self.path.name}': {exc}"
            ) from exc
        if not capture.isOpened():
            capture.release()
            raise VideoInputError(
                f"Unable to read the selected video: {self.path.name}. "
                "It may be corrupted or in an unsupported format."
            )
        self._capture = capture
        self._info = self._collect_metadata(capture)
        logger.info(
            "Video loaded: %s (%dx%d @ %.2ffps, %d frames, %.1fs)",
            self.path.name, self._info.width, self._info.height,
            self._info.fps, self._info.frame_count, self._info.duration_seconds,
        )
        return True

    def _collect_metadata(self, capture: cv2.VideoCapture) -> SourceInfo:
        """Read resolution / FPS / frame count / duration from the file."""
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not (fps > 0):  # some containers report 0/invalid FPS
            fps = 0.0
        duration = (frame_count / fps) if (fps > 0 and frame_count > 0) else 0.0
        return SourceInfo(
            source_type=self.source_type,
            name=self.path.name,
            width=width,
            height=height,
            fps=fps,
            frame_count=max(frame_count, 0),
            duration_seconds=duration,
            opened=True,
            extra={"path": str(self.path)},
        )

    @property
    def metadata(self) -> SourceInfo:
        """Metadata collected at open time (empty info before open)."""
        return self._info or SourceInfo(source_type=self.source_type, name=self.name)

    def read(self) -> Optional[np.ndarray]:
        """Read the next BGR frame. Returns None at end-of-video or on error."""
        if not self.is_open():
            return None
        try:
            ok, frame = self._capture.read()
        except Exception as exc:
            logger.error("Video decoding failure in '%s': %s", self.path.name, exc)
            return None
        if not ok or frame is None:
            return None
        return frame

    def get_timestamp(self) -> float:
        """Video position in seconds (``CAP_PROP_POS_MSEC`` / 1000)."""
        if not self.is_open():
            return 0.0
        try:
            msec = self._capture.get(cv2.CAP_PROP_POS_MSEC)
            if msec and msec > 0:  # unreliable for some containers
                return float(msec) / 1000.0
        except Exception:
            pass
        return 0.0

    def restart(self) -> None:
        """Rewind to the first frame."""
        if self.is_open():
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            logger.info("Video restarted: %s", self.path.name)

    def release(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception as exc:
                logger.warning("Error releasing video '%s': %s", self.path.name, exc)
            self._capture = None
            logger.info("Video stopped and released: %s", self.path.name)

    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def get_info(self) -> SourceInfo:
        return self.metadata
