"""Tests for webcam/video source classes (no physical webcam required)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.camera import (
    InputState,
    SourceType,
    VideoFileSource,
    VideoInputError,
    WebcamSource,
)
from src.utils.paths import PROJECT_ROOT


class TestWebcamSource:
    def test_camera_indices_supported(self):
        for index in (0, 1, 2):
            source = WebcamSource(index=index)
            assert source.index == index
            assert source.source_type is SourceType.WEBCAM

    def test_invalid_camera_index_raises(self):
        source = WebcamSource(index=99, backend="any")
        with patch("cv2.VideoCapture") as mock_capture:
            handle = MagicMock()
            handle.isOpened.return_value = False
            mock_capture.return_value = handle
            with pytest.raises(VideoInputError, match="Unable to open webcam"):
                source.open()

    def test_not_open_before_open(self):
        source = WebcamSource(index=0)
        assert source.is_open() is False
        assert source.read() is None  # safe read when not open

    def test_release_is_idempotent(self):
        source = WebcamSource(index=0)
        source.release()
        source.release()  # must not raise
        assert source.is_open() is False

    def test_read_failure_returns_none(self):
        source = WebcamSource(index=0)
        handle = MagicMock()
        handle.isOpened.return_value = True
        handle.read.return_value = (False, None)
        source._capture = handle
        assert source.read() is None

    def test_info_reports_requested_values_when_closed(self):
        source = WebcamSource(index=2, width=640, height=480, fps=15)
        info = source.get_info()
        assert info.source_type is SourceType.WEBCAM
        assert info.opened is False
        assert info.width == 640
        assert info.height == 480

    def test_error_message_is_actionable(self):
        source = WebcamSource(index=7, backend="any")
        with patch("cv2.VideoCapture") as mock_capture:
            handle = MagicMock()
            handle.isOpened.return_value = False
            mock_capture.return_value = handle
            with pytest.raises(VideoInputError) as exc_info:
                source.open()
            assert "webcam 7" in str(exc_info.value).lower()


class TestVideoFileSource:
    def test_missing_file_raises(self, tmp_path):
        source = VideoFileSource(tmp_path / "does_not_exist.mp4")
        with pytest.raises(VideoInputError, match="does not exist"):
            source.open()

    def test_unsupported_extension_content_still_validated(self, corrupt_video_path):
        """Corrupted video must raise, never crash."""
        source = VideoFileSource(corrupt_video_path)
        with pytest.raises(VideoInputError, match="Unable to read"):
            source.open()

    def test_valid_video_opens(self, sample_video_path):
        source = VideoFileSource(sample_video_path)
        assert source.open() is True
        assert source.is_open() is True
        source.release()
        assert source.is_open() is False

    def test_video_metadata(self, sample_video_path):
        source = VideoFileSource(sample_video_path)
        source.open()
        info = source.get_info()
        assert info.source_type is SourceType.VIDEO_FILE
        assert info.opened is True
        assert info.width == 64
        assert info.height == 48
        assert info.fps == pytest.approx(10.0, rel=0.01)
        assert info.frame_count >= 12
        assert info.duration_seconds > 0
        source.release()

    def test_video_read_all_frames_then_end(self, sample_video_path):
        source = VideoFileSource(sample_video_path)
        source.open()
        count = 0
        while source.read() is not None:
            count += 1
        assert count >= 12
        # Reading past the end must return None, never raise.
        assert source.read() is None
        source.release()

    def test_video_restart_resets_position(self, sample_video_path):
        source = VideoFileSource(sample_video_path)
        source.open()
        first = source.read()
        source.read()
        source.restart()
        second = source.read()
        assert first.shape == second.shape
        source.release()

    def test_open_is_idempotent(self, sample_video_path):
        source = VideoFileSource(sample_video_path)
        assert source.open() is True
        handle = source._capture
        assert source.open() is True
        assert source._capture is handle  # same handle, not reopened
        source.release()

    def test_release_is_idempotent(self, sample_video_path):
        source = VideoFileSource(sample_video_path)
        source.release()
        source.release()
        assert source.is_open() is False
