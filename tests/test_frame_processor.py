"""Tests for frame preprocessing: validation, resizing, skipping."""

import time

import numpy as np
import pytest

from src.camera import FramePacket, SourceType
from src.camera.frame_processor import FrameProcessor


@pytest.fixture
def frame():
    """A 1280x720 BGR test frame."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


class TestFrameValidation:
    def test_valid_frame(self, frame):
        assert FramePacket.validate(frame) is True
        assert FrameProcessor.validate(frame) is True

    def test_none_frame_invalid(self):
        assert FramePacket.validate(None) is False

    def test_empty_frame_invalid(self):
        assert FramePacket.validate(np.zeros((0, 0, 3), dtype=np.uint8)) is False

    def test_non_array_invalid(self):
        assert FramePacket.validate([1, 2, 3]) is False

    def test_process_rejects_invalid(self):
        processor = FrameProcessor()
        assert processor.process(None, 0) is None


class TestFramePacket:
    def test_packet_fields(self, frame):
        packet = FramePacket(
            frame=frame,
            frame_number=7,
            timestamp=1.5,
            source_type=SourceType.WEBCAM,
        )
        assert packet.frame_number == 7
        assert packet.timestamp == 1.5
        assert packet.source_type is SourceType.WEBCAM
        assert packet.width == 1280
        assert packet.height == 720

    def test_packet_validity(self, frame):
        packet = FramePacket(frame=frame, frame_number=0, timestamp=0.0,
                             source_type=SourceType.VIDEO_FILE)
        assert packet.is_valid is True

    def test_to_rgb_returns_rgb(self, frame):
        packet = FramePacket(frame=frame, frame_number=0, timestamp=0.0,
                             source_type=SourceType.WEBCAM)
        rgb = packet.to_rgb()
        assert rgb.shape == frame.shape
        assert packet.frame is frame  # original untouched

    def test_to_dict_summary(self, frame):
        packet = FramePacket(frame=frame, frame_number=3, timestamp=0.4,
                             source_type=SourceType.VIDEO_FILE)
        summary = packet.to_dict()
        assert summary["frame_number"] == 3
        assert summary["source_type"] == "video_file"
        assert summary["width"] == 1280


class TestResizing:
    def test_downscale_preserves_aspect(self, frame):
        processor = FrameProcessor(max_width=640)
        out = processor.resize(frame)
        assert out.shape[1] == 640
        assert out.shape[0] == 360  # 720 * 640/1280

    def test_no_upscale_small_frames(self):
        small = np.zeros((240, 320, 3), dtype=np.uint8)
        processor = FrameProcessor(max_width=640)
        out = processor.resize(small)
        assert out.shape == (240, 320, 3)

    def test_exact_target_size(self, frame):
        processor = FrameProcessor(target_size=(640, 480))
        out = processor.resize(frame)
        assert out.shape == (480, 640, 3)

    def test_no_resize_needed_returns_same_array(self, frame):
        frame_small = np.zeros((480, 640, 3), dtype=np.uint8)
        processor = FrameProcessor(max_width=640)
        assert processor.resize(frame_small) is frame_small


class TestFrameSkipping:
    def test_skip_zero_processes_everything(self):
        processor = FrameProcessor(frame_skip=0)
        assert all(processor.should_process(i) for i in range(10))

    def test_skip_one_processes_every_second(self):
        processor = FrameProcessor(frame_skip=1)
        assert processor.should_process(0) is True
        assert processor.should_process(1) is False
        assert processor.should_process(2) is True

    def test_skip_two_processes_every_third(self):
        processor = FrameProcessor(frame_skip=2)
        assert processor.should_process(0) is True
        assert processor.should_process(1) is False
        assert processor.should_process(2) is False
        assert processor.should_process(3) is True

    def test_process_returns_none_when_skipped(self, frame):
        processor = FrameProcessor(frame_skip=1)
        assert processor.process(frame, 1) is None
        assert processor.process(frame, 0) is not None


class TestTimestamps:
    def test_webcam_style_timestamp(self):
        before = time.perf_counter()
        packet = FramePacket(
            frame=np.zeros((48, 64, 3), dtype=np.uint8),
            frame_number=0,
            timestamp=time.perf_counter(),
            source_type=SourceType.WEBCAM,
        )
        assert before <= packet.timestamp

    def test_video_style_timestamp_from_property(self, sample_video_path):
        from src.camera import VideoFileSource

        source = VideoFileSource(sample_video_path)
        source.open()
        source.read()  # advance one frame
        stamp = source.get_timestamp()
        assert stamp >= 0.0
        source.release()
