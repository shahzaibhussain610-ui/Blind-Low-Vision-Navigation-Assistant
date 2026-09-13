"""Tests for detection result structures."""

import numpy as np
import pytest

from src.detection import Detection, DetectionResult, ImagePosition, classify_position


@pytest.fixture
def frame_640():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def make_detection(frame_width=640, **overrides):
    values = dict(
        class_id=0,
        class_name="person",
        confidence=0.9,
        x1=100.0, y1=50.0, x2=200.0, y2=300.0,
        frame_number=5,
        timestamp=1.25,
        frame_width=frame_width,
        frame_height=480,
    )
    values.update(overrides)
    return Detection(**values)


class TestDetectionGeometry:
    def test_center_calculation(self):
        d = make_detection()
        assert d.center_x == pytest.approx(150.0)
        assert d.center_y == pytest.approx(175.0)

    def test_size_calculation(self):
        d = make_detection()
        assert d.width == pytest.approx(100.0)
        assert d.height == pytest.approx(250.0)

    def test_bbox_tuple(self):
        d = make_detection()
        assert d.bbox == (100.0, 50.0, 200.0, 300.0)

    def test_center_of_person_is_image_center(self):
        d = make_detection(x1=250.0, x2=390.0)  # center x = 320 of 640
        assert d.position is ImagePosition.CENTER

    def test_position_left(self):
        d = make_detection(x1=0.0, x2=80.0)
        assert d.position is ImagePosition.LEFT

    def test_position_right(self):
        d = make_detection(x1=560.0, x2=640.0)
        assert d.position is ImagePosition.RIGHT

    def test_classify_position_function(self):
        assert classify_position(320, 640) is ImagePosition.CENTER
        assert classify_position(50, 640) is ImagePosition.LEFT
        assert classify_position(600, 640) is ImagePosition.RIGHT

    def test_clamp_to_frame(self):
        d = make_detection(x1=-30.0, x2=700.0)
        clamped = d.clamp_to_frame(640, 480)
        assert clamped.x1 == 0.0
        assert clamped.x2 == 640.0

    def test_invalid_box_geometry(self):
        assert make_detection(x1=200.0, x2=100.0).is_valid() is False
        assert make_detection(confidence=1.5).is_valid() is False

    def test_to_dict_fields(self):
        data = make_detection().to_dict()
        for key in ("class_id", "class_name", "confidence", "bbox",
                    "center", "size", "position", "frame_number", "timestamp"):
            assert key in data
        assert data["class_name"] == "person"


class TestDetectionResult:
    def test_empty_result(self):
        result = DetectionResult(frame_number=0, timestamp=0.0)
        assert result.count == 0
        assert result.detections == []
        assert result.to_dicts() == []

    def test_result_with_detections(self):
        result = DetectionResult(
            frame_number=3, timestamp=0.5,
            detections=[make_detection(), make_detection(class_name="car", class_id=2)],
            inference_time_ms=40.0,
            inference_fps=25.0,
            model_name="yolo11n.pt",
            device="cpu",
            frame_width=640,
            frame_height=480,
        )
        assert result.count == 2
        assert result.inference_time_ms == 40.0
        assert len(result.by_class("person")) == 1
        assert result.by_class("cat") == []
        assert result.to_dicts()[0]["class_name"] == "person"
