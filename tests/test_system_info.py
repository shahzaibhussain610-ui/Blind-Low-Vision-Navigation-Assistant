"""Tests for system information and device selection."""

from src.system.system_info import (
    CPUInfo,
    GPUInfo,
    RAMInfo,
    SystemInfo,
    check_gpu,
    collect_system_info,
    get_device,
    get_os_info,
    get_python_version,
    get_cpu_info,
    get_ram_info,
)
from src.system.system_status import SystemStatus, build_initial_status
from src.config.config_manager import ConfigManager
from src.utils.paths import APP_CONFIG_FILE


class TestBasicSystemInfo:
    def test_python_version_format(self):
        version = get_python_version()
        parts = version.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_os_info_returns_pair(self):
        os_name, os_version = get_os_info()
        assert os_name in ("Windows", "Linux", "macOS") or os_name
        assert isinstance(os_version, str)

    def test_cpu_info(self):
        cpu = get_cpu_info()
        assert isinstance(cpu, CPUInfo)
        assert cpu.logical_cores >= 1

    def test_ram_info(self):
        ram = get_ram_info()
        assert isinstance(ram, RAMInfo)
        assert ram.total_gb > 0
        assert ram.available_gb >= 0

    def test_check_gpu_does_not_crash(self):
        gpu = check_gpu()
        assert isinstance(gpu, GPUInfo)
        if gpu.available:
            assert gpu.name
        else:
            assert gpu.note  # CPU-mode explanation must be present


class TestDeviceSelection:
    def test_get_device_returns_valid_value(self):
        assert get_device() in ("cuda", "cpu")

    def test_get_device_cpu_forced(self):
        assert get_device(enable_gpu_if_available=False) == "cpu"

    def test_collect_system_info(self):
        info = collect_system_info()
        assert isinstance(info, SystemInfo)
        assert info.python_version != "Unknown"
        assert info.preferred_device in ("cuda", "cpu")
        assert info.to_dict()["preferred_device"] in ("cuda", "cpu")


class TestSystemStatus:
    def test_build_initial_status(self):
        config = ConfigManager(APP_CONFIG_FILE)
        system = collect_system_info()
        status = build_initial_status(config, system)
        assert isinstance(status, SystemStatus)
        assert status.foundation_ready is True
        assert status.ready is True
        assert status.preferred_device == system.preferred_device

    def test_future_modules_registered(self):
        config = ConfigManager(APP_CONFIG_FILE)
        system = collect_system_info()
        status = build_initial_status(config, system)
        # Parts 1-10 are all implemented in the complete pipeline.
        for name in ("camera", "detector", "depth", "tracker", "navigation", "haptic"):
            module = status.module(name)
            assert module.implemented is True, f"{name} should be implemented"

    def test_module_placeholder_for_unknown(self):
        status = SystemStatus()
        assert status.module("never_registered").state == "not_initialized"
