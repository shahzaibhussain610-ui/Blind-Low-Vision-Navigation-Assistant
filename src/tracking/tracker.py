"""Object tracking with ByteTrack (Part 5).

Uses Ultralytics' ByteTrack implementation through the detector's own
model (``model.track(persist=True)``) so the model is still loaded only
once. Maintains per-track history to classify movement by combining
position changes with relative-depth changes.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

import numpy as np

from src.detection.detector import YOLODetector
from src.detection.detection_result import Detection
from src.logging.logger import get_logger

logger = get_logger("tracking")


class MovementStatus(str, Enum):
    """Movement classification for a tracked object (Part 5)."""

    STATIC = "STATIC"
    MOVING = "MOVING"
    APPROACHING = "APPROACHING"
    MOVING_AWAY = "MOVING_AWAY"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class TrackSnapshot:
    """One historical observation of a track."""

    frame_number: int
    center_x: float
    center_y: float
    relative_depth: float


@dataclass
class TrackedObject:
    """A detection enriched with tracking + movement information."""

    detection: Detection
    relative_depth: float = 0.5
    depth_label: str = "MIDDLE"
    movement: MovementStatus = MovementStatus.UNCERTAIN
    displacement_px: float = 0.0
    depth_trend: float = 0.0
    frames_tracked: int = 0

    @property
    def track_id(self) -> int:
        return self.detection.track_id

    @property
    def class_name(self) -> str:
        return self.detection.class_name


@dataclass
class TrackingResult:
    """All tracked objects for one frame."""

    frame_number: int
    timestamp: float
    tracked: list = field(default_factory=list)
    processing_time_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.tracked)

    def by_movement(self, status: MovementStatus) -> list:
        return [t for t in self.tracked if t.movement is status]

    @property
    def any_approaching(self) -> bool:
        return any(t.movement is MovementStatus.APPROACHING for t in self.tracked)


class ObjectTracker:
    """ByteTrack wrapper with movement classification."""

    def __init__(
        self,
        detector: YOLODetector,
        tracker_config: str = "bytetrack.yaml",
        history_length: int = 12,
        static_threshold_px: float = 6.0,
        approaching_depth_delta: float = 0.03,
        min_history: int = 3,
    ) -> None:
        self.detector = detector
        self.tracker_config = tracker_config
        self.history_length = int(history_length)
        self.static_threshold_px = float(static_threshold_px)
        self.approaching_depth_delta = float(approaching_depth_delta)
        self.min_history = int(min_history)
        self._history: Dict[int, deque] = {}
        self._enabled = False
        self._tracking_fallback_warned = False

    def enable(self) -> None:
        """Switch the underlying model into tracking mode (one-time)."""
        if not self._enabled:
            if not self.detector.is_loaded:
                self.detector.load()
            logger.info(
                "Tracking enabled: %s (history=%d, static<=%.1fpx)",
                self.tracker_config, self.history_length, self.static_threshold_px,
            )
            self._enabled = True

    def reset(self) -> None:
        """Forget all track history (e.g. on source change/restart)."""
        self._history.clear()
        logger.info("Tracking history reset")

    # ------------------------------------------------------------------ #
    def update(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        depth_lookup=None,
    ) -> TrackingResult:
        """Track objects in one frame.

        Args:
            frame: BGR frame.
            frame_number: Sequential frame counter.
            timestamp: Frame timestamp.
            depth_lookup: Optional callable ``Detection -> float`` returning
                relative depth; when omitted, position-only movement is used.
        """
        start = time.perf_counter()
        detections = self._run_track(frame, frame_number, timestamp)
        tracked = []
        for detection in detections:
            depth = float(depth_lookup(detection)) if depth_lookup else 0.5
            tracked.append(self._classify(detection, depth, frame_number))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result = TrackingResult(
            frame_number=frame_number,
            timestamp=timestamp,
            tracked=tracked,
            processing_time_ms=elapsed_ms,
        )
        self._prune_history(frame_number)
        return result

    def _run_track(self, frame, frame_number, timestamp) -> list:
        """Run ByteTrack through the detector's model; return Detections."""
        if not self.detector.is_loaded:
            self.detector.load()
        try:
            results = self.detector._model.track(
                source=frame,
                conf=self.detector.confidence_threshold,
                iou=self.detector.iou_threshold,
                imgsz=self.detector.image_size,
                max_det=self.detector.max_detections,
                device=self.detector.device,
                tracker=self.tracker_config,
                persist=True,
                verbose=False,
            )
        except Exception as exc:
            # Ultralytics ByteTrack needs the optional ``lap`` package. If it
            # is unavailable offline, preserve detections through plain YOLO
            # inference rather than turning the entire scene into zero objects.
            if "No module named 'lap'" in str(exc) or "lap" in str(exc).lower():
                if not self._tracking_fallback_warned:
                    logger.warning(
                        "ByteTrack is unavailable (%s); falling back to YOLO "
                        "detections without persistent tracking IDs.", exc
                    )
                    self._tracking_fallback_warned = True
                detection_result = self.detector.detect_frame(
                    frame, frame_number, timestamp
                )
                for index, detection in enumerate(detection_result.detections):
                    detection.track_id = (
                        frame_number * self.detector.max_detections + index
                    )
                return detection_result.detections
            logger.error("Tracking failure on frame %s: %s", frame_number, exc)
            return []
        detections = self.detector._extract_detections(
            results, frame.shape[1], frame.shape[0], frame_number, timestamp
        )
        boxes = getattr(results[0], "boxes", None) if results else None
        ids = None
        if boxes is not None and getattr(boxes, "id", None) is not None:
            try:
                ids = boxes.id.tolist()
            except (AttributeError, TypeError, ValueError):
                ids = None

        # ByteTrack may withhold an ID for a new or briefly occluded object.
        # Keep that detection visible instead of dropping its box and status.
        for index, detection in enumerate(detections):
            if ids is not None and index < len(ids) and ids[index] is not None:
                detection.track_id = int(ids[index])
            elif detection.track_id < 0:
                detection.track_id = frame_number * self.detector.max_detections + index
        return detections

    def _classify(self, detection, depth, frame_number) -> TrackedObject:
        """Update history and classify movement for one track."""
        history = self._history.setdefault(
            detection.track_id, deque(maxlen=self.history_length)
        )
        history.append(
            TrackSnapshot(
                frame_number=frame_number,
                center_x=detection.center_x,
                center_y=detection.center_y,
                relative_depth=depth,
            )
        )
        tracked = TrackedObject(
            detection=detection, relative_depth=depth, frames_tracked=len(history)
        )
        if len(history) < self.min_history:
            tracked.movement = MovementStatus.UNCERTAIN
            return tracked
        first, last = history[0], history[-1]
        displacement = float(
            np.hypot(last.center_x - first.center_x, last.center_y - first.center_y)
        )
        tracked.displacement_px = displacement
        depth_trend = last.relative_depth - first.relative_depth
        tracked.depth_trend = depth_trend
        if displacement < self.static_threshold_px:
            if depth_trend >= self.approaching_depth_delta:
                tracked.movement = MovementStatus.APPROACHING
            elif depth_trend <= -self.approaching_depth_delta:
                tracked.movement = MovementStatus.MOVING_AWAY
            else:
                tracked.movement = MovementStatus.STATIC
        else:
            tracked.movement = MovementStatus.MOVING
        return tracked

    def _prune_history(self, frame_number: int) -> None:
        """Drop histories for tracks not seen recently."""
        stale = [
            track_id
            for track_id, history in self._history.items()
            if frame_number - history[-1].frame_number > self.history_length
        ]
        for track_id in stale:
            del self._history[track_id]

    def reclassify(self, tracking_result: TrackingResult) -> None:
        """Re-run movement classification with updated relative depths.

        Called by the pipeline after depth estimation mutates each
        TrackedObject's ``relative_depth``; history snapshots for the
        current frame are updated in place (no duplicates appended).
        """
        for tracked in tracking_result.tracked:
            history = self._history.get(tracked.track_id)
            if not history:
                continue
            history[-1].relative_depth = tracked.relative_depth
            if len(history) < self.min_history:
                tracked.movement = MovementStatus.UNCERTAIN
                continue
            first, last = history[0], history[-1]
            displacement = float(
                np.hypot(last.center_x - first.center_x,
                         last.center_y - first.center_y)
            )
            tracked.displacement_px = displacement
            depth_trend = last.relative_depth - first.relative_depth
            tracked.depth_trend = depth_trend
            if displacement < self.static_threshold_px:
                if depth_trend >= self.approaching_depth_delta:
                    tracked.movement = MovementStatus.APPROACHING
                elif depth_trend <= -self.approaching_depth_delta:
                    tracked.movement = MovementStatus.MOVING_AWAY
                else:
                    tracked.movement = MovementStatus.STATIC
            else:
                tracked.movement = MovementStatus.MOVING
