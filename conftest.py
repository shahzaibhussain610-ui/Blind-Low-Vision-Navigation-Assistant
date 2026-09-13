import sys
from pathlib import Path

# Ensure the project root is importable when running pytest from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.logging.logger import setup_logging


@pytest.fixture(scope="session", autouse=True)
def _logging():
    """Initialize project logging once for the whole test session."""
    setup_logging()


@pytest.fixture(scope="session")
def sample_video_path(tmp_path_factory):
    """Generate a small synthetic .avi video for source tests.

    Uses the MJPG codec, which is available in every OpenCV build.
    Returns the path to the generated file.
    """
    import cv2
    import numpy as np

    path = tmp_path_factory.mktemp("videos") / "sample.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
    )
    assert writer.isOpened(), "Failed to create synthetic test video"
    for i in range(12):  # 12 frames = 1.2 s at 10 fps
        frame = np.full((48, 64, 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path


@pytest.fixture(scope="session")
def corrupt_video_path(tmp_path_factory):
    """Create a file with a video extension but invalid content."""
    path = tmp_path_factory.mktemp("bad") / "corrupt.mp4"
    path.write_bytes(b"\x00\x01\x02not-a-real-video\x00" * 20)
    return path
