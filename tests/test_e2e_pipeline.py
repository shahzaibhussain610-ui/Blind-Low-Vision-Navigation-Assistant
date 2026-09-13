"""End-to-end pipeline test (Part 10): input → … → haptic, detector mocked."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.camera import CameraManager, InputState
from src.detection import ClassFilter
from src.detection.detection_result import Detection, DetectionResult
from src.evaluation import EvaluationMetrics
from src.pipeline import build_pipeline
from src.config.config_manager import ConfigManager
from src.utils.paths import APP_CONFIG_FILE
from src.navigation import NavigationCommand


class _FakeIds:
    """Mimics Ultralytics boxes.id tensor (only .tolist is needed)."""

    def __init__(self, ids):
        self._ids = list(ids)

    def tolist(self):
        return list(self._ids)


class _FakeBoxes:
    """Mimics Ultralytics result.boxes (iterable of boxes + .id)."""

    def __init__(self, boxes, ids):
        self._boxes = boxes
        self.id = _FakeIds(ids)

    def __iter__(self):
        return iter(self._boxes)


class MockDetector:
    """Deterministic fake YOLO detector producing a moving 'person' box."""

    model_name = "mock11n.pt"
    device = "cpu"
    confidence_threshold = 0.40
    iou_threshold = 0.45
    image_size = 640
    max_detections = 100
    is_loaded = True
    names = {0: "person"}

    def __init__(self):
        self._model = MagicMock()
        self._model.names = self.names
        self._model.track.side_effect = self._fake_track

    def load(self):
        pass

    def _fake_track(self, **kwargs):
        """Return a fake Ultralytics result: one person, stable id 1."""
        frame = kwargs.get("source")
        height, width = frame.shape[:2]
        cx = width - 80 - self._calls * 14
        self._calls += 1
        box = MagicMock()
        box.cls.item.return_value = 0
        box.conf.item.return_value = 0.88
        tensor = MagicMock()
        tensor.tolist.return_value = [cx - 60, 120, cx + 60, height - 60]
        box.xyxy = [tensor]
        return [MagicMock(boxes=_FakeBoxes([box], [1]))]

    _calls = 0

    def _extract_detections(self, results, width, height, frame_number, timestamp):
        return self.__class__._static_extract(self, results, width, height,
                                              frame_number, timestamp)

    @staticmethod
    def _static_extract(detector, results, width, height, frame_number, timestamp):
        # Reuse the real YOLODetector conversion logic via a lightweight shim.
        from src.detection.detector import YOLODetector

        shim = YOLODetector.__new__(YOLODetector)
        shim.class_filter = ClassFilter()
        shim._model = detector._model
        return shim._extract_detections(results, width, height,
                                        frame_number, timestamp)


@pytest.fixture
def pipeline():
    config = ConfigManager(APP_CONFIG_FILE)
    config._config["tracking"]["min_history"] = 2
    config._config["tracking"]["approaching_depth_delta"] = 0.01
    config._config["navigation"]["min_command_interval"] = 0
    config._config["navigation"]["hysteresis_frames"] = 1
    pipeline = build_pipeline(config)
    pipeline.detector = MockDetector()
    pipeline.tracker.detector = pipeline.detector
    return pipeline


class TestEndToEnd:
    def test_full_pipeline_on_synthetic_video(self, sample_video_path, pipeline):
        manager = CameraManager()
        manager.use_video_file(sample_video_path)
        manager.start()
        frames = []
        while True:
            packet = manager.read_packet()
            if packet is None:
                break
            frames.append(packet.frame)
        manager.stop()
        assert frames

        results = []
        for index, frame in enumerate(frames):
            frame = np.full_like(frame, 60)  # blank background, mock draws box
            results.append(pipeline.process_frame(frame))

        assert all(r.tracking is not None for r in results)
        assert all(r.spatial is not None for r in results)
        assert all(r.risk is not None for r in results)
        assert all(r.navigation is not None for r in results)

    def test_tracking_ids_persist(self, pipeline):
        for n in range(6):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = pipeline.process_frame(frame)
        ids = [t.track_id for t in result.tracking.tracked]
        assert ids == [1]  # same persistent ByteTrack id every frame

    def test_object_becomes_risky_as_it_approaches(self, pipeline):
        from src.risk.risk_assessor import RiskLevel

        last = None
        for n in range(8):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            last = pipeline.process_frame(frame)
        assert last.risk.highest >= RiskLevel.MEDIUM

    def test_navigation_eventually_stops(self, pipeline):
        commands = []
        for n in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = pipeline.process_frame(frame)
            commands.append(result.navigation.command)
        assert NavigationCommand.STOP in commands  # person ends in the path

    def test_haptic_follows_navigation(self, pipeline):
        for n in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = pipeline.process_frame(frame)
        assert pipeline.haptic.current_command is result.navigation.command

    def test_metrics_collect_measured_values(self, pipeline):
        metrics = EvaluationMetrics()
        for n in range(5):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            metrics.record(pipeline.process_frame(frame))
        summary = metrics.summary()
        assert summary["frames_processed"] == 5
        assert summary["avg_pipeline_ms"] > 0
        usage = EvaluationMetrics.system_usage()
        assert 0 <= usage.cpu_percent <= 100
        assert usage.ram_available_gb > 0
