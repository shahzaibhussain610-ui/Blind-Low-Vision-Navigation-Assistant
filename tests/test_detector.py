"""Tests for the YOLO detector (model mocked; one optional real test)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detection import (
    ClassFilter,
    DetectionResult,
    InferenceError,
    ModelLoadError,
    YOLODetector,
)
from src.camera import FramePacket, SourceType


@pytest.fixture
def frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def make_box(cls_id=0, conf=0.9, xyxy=(100.0, 50.0, 200.0, 300.0)):
    """Build a mock Ultralytics box object."""
    box = MagicMock()
    box.cls.item.return_value = cls_id
    box.conf.item.return_value = conf
    tensor = MagicMock()
    tensor.tolist.return_value = list(xyxy)
    box.xyxy = [tensor]
    return box


def mock_results(boxes):
    """Build a mock Ultralytics results list."""
    result = MagicMock()
    result_boxes = MagicMock()
    result_boxes.__iter__.return_value = iter(boxes)
    result.boxes = result_boxes
    return [result]


def build_detector_with_mock_model(boxes=None, coco_names=None, **kwargs):
    """Create a YOLODetector whose internal model is a mock."""
    detector = YOLODetector(device="cpu", **kwargs)
    model = MagicMock()
    names = coco_names if coco_names is not None else {
        0: "person", 2: "car", 9: "traffic light", 58: "dog", 62: "chair-like"
    }
    model.names = names
    model.predict.return_value = mock_results(boxes if boxes is not None else [])
    detector._model = model
    detector._device = "cpu"
    detector._warm = True
    detector._unavailable_classes = detector.class_filter.set_model_classes(names)
    return detector


class TestConfiguration:
    def test_invalid_confidence_rejected(self):
        with pytest.raises(ValueError):
            YOLODetector(confidence=0.0)
        with pytest.raises(ValueError):
            YOLODetector(confidence=1.0)

    def test_defaults_suitable_for_cpu(self):
        detector = YOLODetector()
        assert detector.model_name == "yolo11n.pt"
        assert detector.confidence_threshold == pytest.approx(0.40)
        assert detector.image_size == 640

    def test_config_loads_detection_section(self):
        from src.config.config_manager import ConfigManager
        from src.utils.paths import APP_CONFIG_FILE

        config = ConfigManager(APP_CONFIG_FILE)
        assert config.has("detection.model")
        assert config.has("detection.confidence_threshold")
        assert config.get("detection.model") == "yolo11n.pt"
        assert config.get("detection.confidence_threshold") == 0.40


class TestDeviceSelection:
    def test_auto_resolves_to_valid_device(self):
        detector = YOLODetector(device="auto")
        detector.load()
        assert detector.device in ("cpu", "cuda")

    def test_forced_cpu(self):
        detector = YOLODetector(device="cpu")
        detector.load()
        assert detector.device == "cpu"


class TestModelLoading:
    def test_load_failure_raises_model_load_error(self):
        detector = YOLODetector(device="cpu", model_name="definitely-not-real.pt")
        with patch("ultralytics.YOLO", side_effect=RuntimeError("bad model")):
            with pytest.raises(ModelLoadError, match="Unable to load YOLO11 model"):
                detector.load()

    def test_model_loaded_once(self):
        detector = build_detector_with_mock_model()
        model = detector._model
        detector.load()
        assert detector._model is model  # not reloaded

    def test_unavailable_classes_documented(self):
        detector = build_detector_with_mock_model()
        assert detector.unavailable_classes  # e.g. stairs not in COCO


class TestInference:
    def test_empty_detections(self, frame):
        detector = build_detector_with_mock_model(boxes=[])
        result = detector.detect_frame(frame, frame_number=1, timestamp=0.1)
        assert isinstance(result, DetectionResult)
        assert result.count == 0
        assert result.inference_time_ms > 0
        assert result.device == "cpu"

    def test_bounding_box_conversion(self, frame):
        detector = build_detector_with_mock_model(
            boxes=[make_box(cls_id=0, conf=0.91)]
        )
        result = detector.detect_frame(frame, 2, 0.2)
        d = result.detections[0]
        assert d.class_name == "person"
        assert d.class_id == 0
        assert d.confidence == pytest.approx(0.91)
        assert d.bbox == (100.0, 50.0, 200.0, 300.0)
        assert d.center_x == pytest.approx(150.0)
        assert d.width == pytest.approx(100.0)

    def test_confidence_filtering_via_predict_params(self, frame):
        """The configured threshold must reach the model invocation."""
        detector = build_detector_with_mock_model(confidence=0.55)
        detector.detect_frame(frame, 1, 0.0)
        _, kwargs = detector._model.predict.call_args
        assert kwargs["conf"] == pytest.approx(0.55)

    def test_class_filter_applied(self, frame):
        detector = build_detector_with_mock_model(
            boxes=[make_box(cls_id=0), make_box(cls_id=62)]  # person + non-relevant
        )
        result = detector.detect_frame(frame, 1, 0.0)
        assert [d.class_id for d in result.detections] == [0]

    def test_invalid_frame_skips_inference(self):
        detector = build_detector_with_mock_model()
        result = detector.detect_frame(None, 1, 0.0)
        assert result.count == 0
        assert not detector._model.predict.called

    def test_inference_failure_raises(self, frame):
        detector = build_detector_with_mock_model()
        detector._model.predict.side_effect = RuntimeError("boom")
        with pytest.raises(InferenceError):
            detector.detect_frame(frame, 1, 0.0)

    def test_frame_packet_wrapper(self, frame):
        detector = build_detector_with_mock_model(boxes=[make_box(cls_id=2, conf=0.8)])
        packet = FramePacket(
            frame=frame, frame_number=9, timestamp=0.7, source_type=SourceType.WEBCAM
        )
        result = detector.detect(packet)
        assert result.frame_number == 9
        assert result.detections[0].class_name == "car"


@pytest.fixture(scope="module")
def real_detector():
    """Real YOLO11 model — skipped automatically if unavailable."""
    pytest.importorskip("ultralytics")
    detector = YOLODetector(device="cpu", model_name="yolo11n.pt", confidence=0.25)
    try:
        detector.load()
    except ModelLoadError as exc:
        pytest.skip(f"YOLO model unavailable in this environment: {exc}")
    return detector


class TestRealInference:
    """Real model inference — runs when the model can be downloaded/loaded."""

    def test_real_model_loads_once(self, real_detector):
        assert real_detector.is_loaded
        model = real_detector._model
        real_detector.load()
        assert real_detector._model is model

    def test_real_inference_valid_results(self, real_detector):
        import cv2

        # Synthetic scene (dark rectangle over noise): the model is NOT
        # required to find anything specific here.
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 150), (440, 330), (30, 30, 30), -1)
        result = real_detector.detect_frame(frame, 0, 0.0)
        assert result.inference_time_ms > 0
        for d in result.detections:
            assert 0.0 <= d.confidence <= 1.0
            assert d.class_name
            assert d.x2 > d.x1 and d.y2 > d.y1
            assert 0 <= d.class_id < len(real_detector.class_names)
