"""Tests for tracking & movement classification (Part 5, mocked model)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.detection import YOLODetector
from src.tracking import (
    MovementStatus,
    ObjectTracker,
    TrackingResult,
    TrackedObject,
)
from src.detection.detection_result import Detection


def make_detection(track_id, cx, cy, frame_number=0, confidence=0.9,
                   frame_width=640, frame_height=480):
    return Detection(
        class_id=0, class_name="person", confidence=confidence,
        x1=cx - 50, y1=cy - 80, x2=cx + 50, y2=cy + 80,
        frame_number=frame_number, timestamp=frame_number / 15.0,
        frame_width=frame_width, frame_height=frame_height,
        track_id=track_id,
    )


@pytest.fixture
def detector():
    mock_detector = YOLODetector(device="cpu")
    model = MagicMock()
    model.names = {0: "person"}
    mock_detector._model = model
    mock_detector._device = "cpu"
    mock_detector._warm = True
    return mock_detector


@pytest.fixture
def tracker(detector):
    return ObjectTracker(
        detector,
        history_length=10,
        static_threshold_px=6.0,
        approaching_depth_delta=0.03,
        min_history=3,
    )


class TestMovementClassification:
    def test_initially_uncertain(self, tracker):
        tracker2 = ObjectTracker(tracker.detector, min_history=3)
        det = make_detection(1, 320, 240, frame_number=0)
        tracked = tracker2._classify(det, 0.5, 0)
        assert tracked.movement is MovementStatus.UNCERTAIN

    def test_static_when_no_displacement_and_flat_depth(self, tracker):
        for n in range(4):
            det = make_detection(1, 320, 240, frame_number=n)
            tracked = tracker._classify(det, 0.50, n)  # same center, same depth
        assert tracked.movement is MovementStatus.STATIC

    def test_approaching_via_depth_trend(self, tracker):
        depths = [0.30, 0.34, 0.40, 0.48]
        tracked = None
        for n, d in enumerate(depths):
            det = make_detection(1, 320, 240, frame_number=n)  # fixed position
            tracked = tracker._classify(det, d, n)
        assert tracked.movement is MovementStatus.APPROACHING

    def test_moving_away_via_depth_trend(self, tracker):
        depths = [0.60, 0.55, 0.48, 0.40]
        tracked = None
        for n, d in enumerate(depths):
            det = make_detection(1, 320, 240, frame_number=n)
            tracked = tracker._classify(det, d, n)
        assert tracked.movement is MovementStatus.MOVING_AWAY

    def test_moving_via_displacement(self, tracker):
        xs = [100, 160, 220, 280]
        tracked = None
        for n, x in enumerate(xs):
            det = make_detection(1, x, 240, frame_number=n)
            tracked = tracker._classify(det, 0.5, n)
        assert tracked.movement is MovementStatus.MOVING

    def test_reclassify_after_depth_update(self, tracker):
        for n in range(3):
            det = make_detection(1, 320, 240, frame_number=n)
            tracker._classify(det, 0.30, n)
        result = TrackingResult(frame_number=3, timestamp=0.2)
        tracked = TrackedObject(detection=make_detection(1, 320, 240, 3),
                                relative_depth=0.50)
        result.tracked = [tracked]
        tracker.reclassify(result)
        assert tracked.movement is MovementStatus.APPROACHING


class TestHistoryManagement:
    def test_reset_clears_history(self, tracker):
        tracker._classify(make_detection(1, 320, 240, 0), 0.5, 0)
        tracker.reset()
        assert tracker._history == {}

    def test_history_pruned_when_track_disappears(self, tracker):
        for n in range(3):
            tracker._classify(make_detection(1, 320, 240, n), 0.5, n)
        tracker._prune_history(frame_number=100)  # far in the future
        assert 1 not in tracker._history

    def test_history_bounded(self, tracker):
        tracker = ObjectTracker(detector, history_length=5, min_history=1)
        for n in range(20):
            tracker._classify(make_detection(1, 320 + n, 240, n), 0.5, n)
        assert len(tracker._history[1]) <= 5

    def test_tracked_object_properties(self):
        det = make_detection(7, 320, 240)
        tracked = TrackedObject(detection=det, relative_depth=0.7)
        assert tracked.track_id == 7
        assert tracked.class_name == "person"

    def test_unmatched_detection_is_kept_for_annotation(self, tracker):
        """Objects without an immediate ByteTrack ID must remain visible."""
        box = MagicMock()
        box.cls.item.return_value = 0
        box.conf.item.return_value = 0.9
        coordinates = MagicMock()
        coordinates.tolist.return_value = [100.0, 100.0, 220.0, 320.0]
        box.xyxy = [coordinates]

        result = MagicMock()
        result.boxes = MagicMock()
        result.boxes.__iter__.return_value = iter([box])
        result.boxes.id = None
        tracker.detector._model.track.return_value = [result]

        tracking = tracker.update(
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_number=4,
            timestamp=0.25,
        )

        assert tracking.count == 1
        assert tracking.tracked[0].track_id >= 0
