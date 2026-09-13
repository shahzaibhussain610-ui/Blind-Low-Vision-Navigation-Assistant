"""Centralized project path management using pathlib.

This module is the single source of truth for filesystem locations.
It never depends on the current working directory.
"""

from pathlib import Path

# Root of this project = parent of the src/ package directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

CONFIG_DIR: Path = PROJECT_ROOT / "config"
TESTS_DIR: Path = PROJECT_ROOT / "tests"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
DATA_SAMPLE_DIR: Path = DATA_DIR / "sample"

MODELS_DIR: Path = PROJECT_ROOT / "models"
DETECTION_MODELS_DIR: Path = MODELS_DIR / "detection"
DEPTH_MODELS_DIR: Path = MODELS_DIR / "depth"

RESULTS_DIR: Path = PROJECT_ROOT / "results"
DETECTIONS_RESULTS_DIR: Path = RESULTS_DIR / "detections"
DEPTH_RESULTS_DIR: Path = RESULTS_DIR / "depth"
TRACKING_RESULTS_DIR: Path = RESULTS_DIR / "tracking"
NAVIGATION_RESULTS_DIR: Path = RESULTS_DIR / "navigation"
EVALUATION_RESULTS_DIR: Path = RESULTS_DIR / "evaluation"

LOGS_DIR: Path = PROJECT_ROOT / "logs"

APP_CONFIG_FILE: Path = CONFIG_DIR / "app.yaml"
LOGGING_CONFIG_FILE: Path = CONFIG_DIR / "logging.yaml"

#: All directories that must exist for the project to run.
REQUIRED_DIRECTORIES: tuple = (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_SAMPLE_DIR,
    DETECTION_MODELS_DIR,
    DEPTH_MODELS_DIR,
    DETECTIONS_RESULTS_DIR,
    DEPTH_RESULTS_DIR,
    TRACKING_RESULTS_DIR,
    NAVIGATION_RESULTS_DIR,
    EVALUATION_RESULTS_DIR,
    LOGS_DIR,
)


def ensure_required_directories() -> list:
    """Create every required project directory if missing.

    Existing files are never touched. Returns the list of directories
    that were newly created (empty list when everything already existed).
    """
    created: list = []
    for directory in REQUIRED_DIRECTORIES:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
    return created
