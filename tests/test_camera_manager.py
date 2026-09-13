"""Tests for the CameraManager (no physical webcam required)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.camera import (
    CameraManager,
    FramePacket,
    InputState,
    ManagerConfig,
    SourceType,
    VideoInputError,
    WebcamSource,
)
from src.camera.camera_manager import FPSMeter, LatencyTracker
from src.camera.video_source import VideoSource


class FakeSource(VideoSource):
    """Deterministic in-memory source for manager tests."""

    source_type = SourceType.VIDEO_FILE

    def __init__(self, frames=5):
        self._frames = [np.full((48, 64, 3), 128, dtype=np.uint8) for _ in range(frames)]
        self._cursor = 0
        self._open = False
        self.released = False

    @property
    def name(self):
        return "fake"

    def open(self):
        self._open = True
        return True

    def read(self):
        if self._cursor >= len(self._frames):
            return None
        frame = self._frames[self._cursor]
        self._cursor += 1
        return frame

    def release(self):
        self.released = True
        self._open = False

    def is_open(self):
        return self._open

    def get_info(self):
        from src.camera.video_source import SourceInfo

        return SourceInfo(source_type=self.source_type, name="fake", opened=self._open)


class TestManagerLifecycle:
    def test_no_input_initially(self):
        manager = CameraManager()
        assert manager.state is InputState.NO_INPUT
        assert manager.read_packet() is None
        assert manager.source_info is None

    def test_start_without_source_raises(self):
        manager = CameraManager()
        with pytest.raises(VideoInputError):
            manager.start()

    def test_start_read_stop(self):
        manager = CameraManager()
        manager._source = FakeSource(frames=3)
        manager.start()
        assert manager.state is InputState.RUNNING
        packet = manager.read_packet()
        assert packet is not None
        assert packet.frame_number == 0
        assert isinstance(packet, FramePacket)
        assert packet.source_type is SourceType.VIDEO_FILE
        manager.stop()
        assert manager.state is InputState.STOPPED
        assert manager.read_packet() is None

    def test_sequential_frame_numbers(self):
        manager = CameraManager()
        manager._source = FakeSource(frames=5)
        manager.start()
        numbers = [manager.read_packet().frame_number for _ in range(5)]
        assert numbers == [0, 1, 2, 3, 4]
        manager.stop()

    def test_end_of_video_state(self):
        manager = CameraManager()
        manager._source = FakeSource(frames=1)
        manager.start()
        assert manager.read_packet() is not None
        assert manager.read_packet() is None  # exhausted
        assert manager.state is InputState.END_OF_VIDEO
        # Repeated reads past the end must stay safe.
        assert manager.read_packet() is None
        assert manager.state is InputState.END_OF_VIDEO

    def test_switch_source_releases_previous(self):
        manager = CameraManager()
        first = FakeSource(frames=2)
        second = FakeSource(frames=2)
        manager._source = first
        manager.start()
        manager._release_source()
        manager._source = second
        assert first.released is True
        manager.stop()

    def test_invalid_frame_dropped(self):
        manager = CameraManager()
        source = FakeSource(frames=1)
        source.read = lambda: np.zeros((0, 0), dtype=np.uint8)
        manager._source = source
        manager.start()
        assert manager.read_packet() is None  # invalid frame never leaves


class TestPauseResume:
    def test_pause_and_resume(self):
        manager = CameraManager()
        manager._source = FakeSource(frames=10)
        manager.start()
        manager.pause()
        assert manager.state is InputState.PAUSED
        assert manager.read_packet() is None
        manager.resume()
        assert manager.state is InputState.RUNNING
        assert manager.read_packet() is not None
        manager.stop()


class TestWebcamBackendSelection:
    def test_invalid_camera_raises(self):
        source = WebcamSource(index=99, backend="any")
        with patch("cv2.VideoCapture") as mock_capture:
            handle = MagicMock()
            handle.isOpened.return_value = False
            mock_capture.return_value = handle
            with pytest.raises(VideoInputError):
                source.open()
            # Every attempted backend must release its handle.
            assert handle.release.called

    def test_webcam_open_success(self):
        source = WebcamSource(index=0, backend="any")
        with patch("cv2.VideoCapture") as mock_capture:
            handle = MagicMock()
            handle.isOpened.return_value = True
            handle.get.return_value = 30.0
            mock_capture.return_value = handle
            assert source.open() is True
            assert source.is_open() is True
            source.release()
            assert handle.release.called
            assert source.is_open() is False


class TestMeasurement:
    def test_fps_meter(self):
        meter = FPSMeter()
        assert meter.fps == 0.0
        base = 1000.0
        times = iter([base + i * 0.1 for i in range(11)])
        for _ in range(11):
            meter.tick(next(times))
        assert meter.fps == pytest.approx(10.0, rel=0.01)

    def test_latency_tracker(self):
        tracker = LatencyTracker()
        with tracker.measure("input"):
            pass
        assert tracker.last_ms("input") >= 0
        tracker.record("detection", 12.5)
        assert tracker.average_ms("detection") == 12.5

    def test_manager_measured_fps_is_zero_before_read(self):
        manager = CameraManager()
        assert manager.measured_fps == 0.0

    def test_manager_config_defaults(self):
        cfg = ManagerConfig()
        assert cfg.webcam_index == 0
        assert cfg.webcam_width == 640
        assert cfg.webcam_height == 480
        assert cfg.frame_skip == 0


class TestVideoFileIntegration:
    def test_manager_reads_real_video(self, sample_video_path):
        manager = CameraManager()
        manager.use_video_file(sample_video_path)
        manager.start()
        packets = []
        while True:
            packet = manager.read_packet()
            if packet is None:
                break
            packets.append(packet)
        assert len(packets) > 0
        assert packets[0].frame_number == 0
        assert packets[-1].frame_number == len(packets) - 1
        assert manager.state is InputState.END_OF_VIDEO
        manager.stop()

    def test_manager_restart_after_end(self, sample_video_path):
        manager = CameraManager()
        manager.use_video_file(sample_video_path)
        manager.start()
        while manager.read_packet() is not None:
            pass
        assert manager.state is InputState.END_OF_VIDEO
        manager.restart()
        packet = manager.read_packet()
        assert packet is not None
        assert packet.frame_number == 0
        manager.stop()
