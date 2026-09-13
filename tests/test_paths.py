"""Tests for project path management."""

from pathlib import Path

from src.utils.paths import (
    APP_CONFIG_FILE,
    CONFIG_DIR,
    DATA_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DATA_SAMPLE_DIR,
    LOGS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    TESTS_DIR,
    ensure_required_directories,
)


class TestPathDefinitions:
    def test_project_root_is_path(self):
        assert isinstance(PROJECT_ROOT, Path)
        assert PROJECT_ROOT.is_dir()

    def test_project_root_contains_key_files(self):
        assert (PROJECT_ROOT / "app.py").is_file()
        assert (PROJECT_ROOT / "src").is_dir()

    def test_standard_paths_are_path_objects(self):
        for path in (
            CONFIG_DIR,
            DATA_DIR,
            MODELS_DIR,
            RESULTS_DIR,
            LOGS_DIR,
            TESTS_DIR,
        ):
            assert isinstance(path, Path)

    def test_config_file_exists(self):
        assert APP_CONFIG_FILE.is_file()

    def test_subdirectory_paths(self):
        assert DATA_RAW_DIR.parent == DATA_DIR
        assert DATA_RAW_DIR.name == "raw"
        assert DATA_PROCESSED_DIR.name == "processed"
        assert DATA_SAMPLE_DIR.name == "sample"


class TestDirectoryCreation:
    def test_ensure_required_directories_creates_all(self):
        created = ensure_required_directories()
        from src.utils.paths import REQUIRED_DIRECTORIES

        for directory in REQUIRED_DIRECTORIES:
            assert directory.is_dir(), f"{directory} should exist"
        # Second call must not report anything new (idempotent).
        assert ensure_required_directories() == [] or created == []

    def test_idempotent(self):
        first = ensure_required_directories()
        second = ensure_required_directories()
        assert second == [] or (first == [] and second == [])

    def test_existing_files_not_deleted(self, tmp_path):
        from src.utils import paths as paths_module

        marker = tmp_path / "keep_me.txt"
        marker.write_text("data", encoding="utf-8")
        original = paths_module.REQUIRED_DIRECTORIES
        try:
            paths_module.REQUIRED_DIRECTORIES = (tmp_path,)
            paths_module.ensure_required_directories()
            assert marker.is_file() and marker.read_text() == "data"
        finally:
            paths_module.REQUIRED_DIRECTORIES = original
