"""Tests for the configuration manager."""

import pytest

from src.config.config_manager import ConfigManager, ConfigurationError
from src.utils.paths import APP_CONFIG_FILE


class TestConfigLoading:
    def test_app_yaml_loads(self):
        config = ConfigManager(APP_CONFIG_FILE)
        assert config._config, "Configuration dictionary should not be empty"

    def test_required_sections_exist(self):
        config = ConfigManager(APP_CONFIG_FILE)
        for section in ConfigManager.REQUIRED_SECTIONS:
            assert config.has(section), f"Missing required section: {section}"

    def test_nested_access(self):
        config = ConfigManager(APP_CONFIG_FILE)
        name = config.get("application.name")
        assert isinstance(name, str) and name

    def test_get_returns_default_for_missing_key(self):
        config = ConfigManager(APP_CONFIG_FILE)
        assert config.get("does.not.exist") is None
        assert config.get("does.not.exist", default=42) == 42

    def test_has_detects_keys(self):
        config = ConfigManager(APP_CONFIG_FILE)
        assert config.has("application.version") is True
        assert config.has("nope.nope") is False

    def test_as_dict_returns_copy(self):
        config = ConfigManager(APP_CONFIG_FILE)
        snapshot = config.as_dict()
        snapshot["application"]["name"] = "mutated"
        assert config.get("application.name") != "mutated"


class TestConfigErrorHandling:
    def test_missing_file_raises_configuration_error(self, tmp_path):
        with pytest.raises(ConfigurationError, match="could not be found"):
            ConfigManager(tmp_path / "missing.yaml")

    def test_invalid_yaml_raises_configuration_error(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("application: [unclosed, bracket", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="not valid YAML"):
            ConfigManager(bad_file)

    def test_non_mapping_root_raises(self, tmp_path):
        bad_file = tmp_path / "list.yaml"
        bad_file.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="mapping"):
            ConfigManager(bad_file)

    def test_missing_required_section_raises(self, tmp_path):
        bad_file = tmp_path / "incomplete.yaml"
        bad_file.write_text("application:\n  name: X\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="missing required section"):
            ConfigManager(bad_file)
