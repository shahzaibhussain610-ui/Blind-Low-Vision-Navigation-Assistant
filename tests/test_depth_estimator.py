"""Tests for relative depth estimation (Part 4)."""

import numpy as np
import pytest

from src.depth import ObjectDepth, RelativeDepthEstimator
from src.detection.detection_result import Detection


def make_detection(frame_width=640, frame_height=480, **overrides):
    values = dict(
        class_id=0, class_name="person", confidence=0.9,
        x1=200.0, y1=100.0, x2=400.0, y2=400.0,   # big, low box -> NEAR
        frame_number=0, timestamp=0.0,
        frame_width=frame_width, frame_height=frame_height,
    )
    values.update(overrides)
    return Detection(**values)


class TestHeuristicBackend:
    def test_big_low_box_is_near(self):
        estimator = RelativeDepthEstimator()
        depth = estimator.estimate_one(make_detection())
        assert depth.backend == "heuristic"
        assert depth.depth_label == "NEAR"
        assert depth.relative_depth > 0.6

    def test_small_high_box_is_far(self):
        estimator = RelativeDepthEstimator()
        depth = estimator.estimate_one(
            make_detection(x1=300.0, y1=20.0, x2=340.0, y2=60.0)  # tiny, high
        )
        assert depth.depth_label in ("FAR", "MIDDLE")
        assert depth.relative_depth < 0.5

    def test_relative_depth_in_unit_range(self):
        estimator = RelativeDepthEstimator()
        for box in [(0, 0, 640, 480), (0, 0, 10, 10), (100, 100, 200, 200)]:
            depth = estimator.estimate_one(
                make_detection(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
            )
            assert 0.0 <= depth.relative_depth <= 1.0

    def test_degenerate_frame_dimensions(self):
        estimator = RelativeDepthEstimator()
        depth = estimator.estimate_one(make_detection(frame_width=0, frame_height=0))
        assert depth.relative_depth == pytest.approx(0.5)
        assert depth.depth_label == "MIDDLE"

    def test_labels_respect_configured_thresholds(self):
        estimator = RelativeDepthEstimator(near_threshold=0.9, far_threshold=0.1)
        assert estimator.label_for(0.95) == "NEAR"
        assert estimator.label_for(0.05) == "FAR"
        assert estimator.label_for(0.5) == "MIDDLE"

    def test_estimate_for_detections(self, frame=None):
        estimator = RelativeDepthEstimator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [make_detection(), make_detection(class_name="car", class_id=2)]
        mapping = estimator.estimate_for_detections(frame, detections)
        assert set(mapping.keys()) == {0, 1}
        assert all(isinstance(v, ObjectDepth) for v in mapping.values())

    def test_unknown_mode_falls_back(self):
        estimator = RelativeDepthEstimator(mode="nonsense")
        assert estimator.mode == "heuristic"

    def test_depth_is_relative_not_metric(self):
        """Documentation test: depth is a relative score, never meters."""
        estimator = RelativeDepthEstimator()
        depth = estimator.estimate_one(make_detection())
        assert 0.0 <= depth.relative_depth <= 1.0
        assert depth.depth_label in ("NEAR", "MIDDLE", "FAR")

    def test_midas_fallback_when_unavailable(self):
        """MiDaS mode must gracefully fall back to the heuristic backend."""
        estimator = RelativeDepthEstimator(mode="midas")
        estimator._last_frame = None  # no frame context -> MiDaS returns None
        depth = estimator.estimate_one(make_detection())
        assert depth.backend == "heuristic"
