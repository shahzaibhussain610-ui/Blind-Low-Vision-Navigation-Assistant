"""Central input management: owns the active source and its state.

The manager provides a single facade over webcam / video-file sources,
keeps an accurate :class:`InputState`, measures real input FPS and
per-stage latency, and guarantees that OpenCV resources are released.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.camera.frame import FramePacket, InputState, SourceType
from src.camera.video_source import (
    SourceInfo,
    VideoFileSource,
    VideoInputError,
    VideoSource,
    WebcamSource,
)
from src.logging.logger import get_logger

logger = get_logger("camera_manager")


class FPSMeter:
    """Rolling FPS measurement from recent frame-read intervals."""

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque = deque(maxlen=window)

    def tick(self, now: Optional[float] = None) -> None:
        """Register that a frame was just read."""
        self._timestamps.append(now if now is not None else time.perf_counter())

    @property
    def fps(self) -> float:
        """Measured FPS over the rolling window (0.0 until stable)."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        self._timestamps.clear()


class LatencyTracker:
    """Extensible per-stage latency measurement (milliseconds).

    Part 2 measures the ``input`` stage automatically. Later parts can
    call :meth:`measure` for ``detection``, ``depth``, ``tracking``,
    ``navigation`` and ``total_pipeline`` without any changes here.
    """

    #: Stage names reserved for the full pipeline (documented for Part 3+).
    FUTURE_STAGES = ("detection", "depth", "tracking", "navigation", "total_pipeline")

    def __init__(self, window: int = 30) -> None:
        self._durations: dict = {}
        self._window = window

    def measure(self, stage: str):
        """Context manager recording duration of ``stage`` in ms.

        Usage::

            with tracker.measure("input"):
                frame = capture.read()
        """
        return _StageTimer(self, stage, self._window)

    def record(self, stage: str, duration_ms: float) -> None:
        """Manually record a duration (ms) for a stage."""
        history = self._durations.setdefault(stage, deque(maxlen=self._window))
        history.append(duration_ms)

    def last_ms(self, stage: str) -> float:
        history = self._durations.get(stage)
        return history[-1] if history else 0.0

    def average_ms(self, stage: str) -> float:
        history = self._durations.get(stage)
        return sum(history) / len(history) if history else 0.0

    def to_dict(self) -> dict:
        return {
            stage: {
                "last_ms": round(self.last_ms(stage), 2),
                "avg_ms": round(self.average_ms(stage), 2),
            }
            for stage in self._durations
        }


class _StageTimer:
    """Internal context manager used by :class:`LatencyTracker`."""

    def __init__(self, tracker: LatencyTracker, stage: str, window: int) -> None:
        self._tracker = tracker
        self._stage = stage
        self._window = window
        self._start = 0.0

    def __enter__(self) -> "_StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        history = self._tracker._durations.setdefault(
            self._stage, deque(maxlen=self._window)
        )
        history.append(elapsed_ms)


@dataclass
class ManagerConfig:
    """Parameters used to construct sources through the manager."""

    webcam_index: int = 0
    webcam_width: int = 640
    webcam_height: int = 480
    webcam_fps: float = 15.0
    webcam_backend: str = "auto"
    frame_skip: int = 0


class CameraManager:
    """Facade that owns the active input source and its lifecycle.

    Typical usage::

        manager = CameraManager(ManagerConfig())
        manager.use_webcam(index=0)
        manager.start()
        packet = manager.read_packet()   # FramePacket or None
        manager.stop()                   # releases OpenCV resources
    """

    def __init__(self, config: Optional[ManagerConfig] = None) -> None:
        self._config = config or ManagerConfig()
        self._source: Optional[VideoSource] = None
        self._state = InputState.NO_INPUT
        self._frame_counter = 0
        self._error_message = ""
        self.fps_meter = FPSMeter()
        self.latency = LatencyTracker()

    # ------------------------------------------------------------------ #
    # Source selection (previous source is always released first)
    # ------------------------------------------------------------------ #
    def _release_source(self) -> None:
        if self._source is not None and self._source.is_open():
            self._source.release()
        self._source = None
        self._frame_counter = 0
        self.fps_meter.reset()

    def use_webcam(
        self, index: Optional[int] = None, backend: Optional[str] = None
    ) -> None:
        """Switch to a webcam source, releasing any previous source."""
        self._release_source()
        self._source = WebcamSource(
            index=self._config.webcam_index if index is None else index,
            width=self._config.webcam_width,
            height=self._config.webcam_height,
            fps=self._config.webcam_fps,
            backend=backend or self._config.webcam_backend,
        )
        self._state = InputState.INITIALIZING
        logger.info("Input source changed: webcam index=%s", self._source.name)

    def use_video_file(self, path) -> None:
        """Switch to a prerecorded video source, releasing any previous one."""
        self._release_source()
        self._source = VideoFileSource(path)
        self._state = InputState.INITIALIZING
        logger.info("Input source changed: video file '%s'", self._source.name)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> SourceInfo:
        """Open the selected source.

        Raises:
            VideoInputError: When the source cannot be opened.
        """
        if self._source is None:
            raise VideoInputError("No input source selected.")
        if self._source.is_open():
            return self._source.get_info()
        self._state = InputState.INITIALIZING
        self._source.open()
        self._frame_counter = 0
        self.fps_meter.reset()
        self._state = InputState.RUNNING
        logger.info("Input started: %s", self._source.name)
        return self._source.get_info()

    def read_packet(self) -> Optional[FramePacket]:
        """Read one frame wrapped in a :class:`FramePacket`.

        ``None`` means "no frame this time" (end of video, transient
        webcam failure). Invalid frames never leave the manager: they
        are dropped and logged.
        """
        if self._source is None or not self._source.is_open():
            return None
        if self._state not in (InputState.RUNNING, InputState.END_OF_VIDEO):
            return None
        try:
            with self.latency.measure("input"):
                raw = self._source.read()
        except Exception as exc:  # unexpected OpenCV failure
            self._error_message = f"{exc.__class__.__name__}: {exc}"
            self._state = InputState.ERROR
            logger.error("Unexpected read failure (%s): %s",
                         self._source.name, self._error_message)
            return None
        if raw is None:
            if self._source.source_type is SourceType.VIDEO_FILE:
                if self._state != InputState.END_OF_VIDEO:
                    self._state = InputState.END_OF_VIDEO
                    logger.info("Video ended: %s", self._source.name)
            return None
        if not FramePacket.validate(raw):
            logger.warning("Invalid frame dropped (number=%d)", self._frame_counter)
            return None
        self.fps_meter.tick()
        packet = FramePacket(
            frame=raw,
            frame_number=self._frame_counter,
            timestamp=self._current_timestamp(),
            source_type=self._source.source_type,
        )
        self._frame_counter += 1
        return packet

    def _current_timestamp(self) -> float:
        """Video position for files; acquisition time for webcams."""
        if self._source.source_type is SourceType.VIDEO_FILE:
            getter = getattr(self._source, "get_timestamp", None)
            if callable(getter):
                return getter()
        return time.perf_counter()

    def pause(self) -> None:
        if self._state is InputState.RUNNING:
            self._state = InputState.PAUSED
            logger.info("Input paused: %s", self._source.name if self._source else "")

    def resume(self) -> None:
        if self._state is InputState.PAUSED:
            self._state = InputState.RUNNING
            self.fps_meter.reset()
            logger.info("Input resumed: %s", self._source.name if self._source else "")

    def stop(self) -> None:
        """Stop the input and release all OpenCV resources."""
        self._release_source()
        self._state = InputState.STOPPED
        logger.info("Input stopped")

    def restart(self) -> None:
        """Rewind a video source to its first frame (webcam: no-op)."""
        if self._source is None:
            return
        if hasattr(self._source, "restart"):
            self._source.restart()  # type: ignore[attr-defined]
        self._frame_counter = 0
        self.fps_meter.reset()
        if self._state is InputState.END_OF_VIDEO:
            self._state = InputState.RUNNING
            logger.info("Input restarted: %s", self._source.name)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> InputState:
        return self._state

    @property
    def error_message(self) -> str:
        return self._error_message

    @property
    def source_info(self) -> Optional[SourceInfo]:
        return self._source.get_info() if self._source else None

    @property
    def source_type(self) -> Optional[SourceType]:
        return self._source.source_type if self._source else None

    @property
    def measured_fps(self) -> float:
        """Actual measured input FPS (never a configured target)."""
        return self.fps_meter.fps

    def is_active(self) -> bool:
        return self._source is not None and self._source.is_open()
