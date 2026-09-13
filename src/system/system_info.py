"""System information collection and CPU/GPU device selection.

PyTorch is imported lazily so the dashboard still works without torch.
Heavy AI models are never initialized here.
"""

from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

import psutil

from src.logging.logger import get_logger

logger = get_logger("system_info")


@dataclass
class CPUInfo:
    name: str = "Unknown"
    physical_cores: int = 0
    logical_cores: int = 0
    usage_percent: Optional[float] = None


@dataclass
class RAMInfo:
    total_gb: float = 0.0
    available_gb: float = 0.0
    used_percent: Optional[float] = None


@dataclass
class GPUInfo:
    available: bool = False
    name: str = "Not available"
    cuda_version: str = "Not available"
    torch_available: bool = False
    note: str = "Running in CPU-compatible mode"


@dataclass
class SystemInfo:
    python_version: str = "Unknown"
    os_name: str = "Unknown"
    os_version: str = "Unknown"
    cpu: CPUInfo = field(default_factory=CPUInfo)
    ram: RAMInfo = field(default_factory=RAMInfo)
    gpu: GPUInfo = field(default_factory=GPUInfo)
    preferred_device: str = "cpu"

    def to_dict(self) -> dict:
        return asdict(self)


def get_python_version() -> str:
    """Return a human-readable Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_os_info() -> tuple:
    """Return (os_name, os_version) for the current platform."""
    system = platform.system()
    if system == "Windows":
        return "Windows", platform.release()
    if system == "Linux":
        return "Linux", platform.release()
    if system == "Darwin":
        return "macOS", platform.release()
    return system or "Unknown", "Unknown"


def _detect_cpu_name() -> str:
    """Best-effort CPU model name detection across platforms."""
    if platform.system() == "Windows":
        return platform.processor() or "Unknown"
    if platform.system() == "Darwin":
        return platform.processor() or "Apple Silicon (Unknown)"
    try:  # Linux: read /proc/cpuinfo
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if "model name" in line and ":" in line:
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "Unknown"


def get_cpu_info() -> CPUInfo:
    """Return CPU name and core counts using platform + psutil."""
    return CPUInfo(
        name=_detect_cpu_name(),
        physical_cores=psutil.cpu_count(logical=False) or 0,
        logical_cores=psutil.cpu_count(logical=True) or 0,
        usage_percent=psutil.cpu_percent(interval=None),
    )


def get_ram_info() -> RAMInfo:
    """Return total/available RAM in GB using psutil."""
    memory = psutil.virtual_memory()
    return RAMInfo(
        total_gb=round(memory.total / (1024 ** 3), 1),
        available_gb=round(memory.available / (1024 ** 3), 1),
        used_percent=round(memory.percent, 1),
    )


def check_gpu(try_nvidia_smi: bool = True) -> GPUInfo:
    """Check whether CUDA-capable GPU access is possible (non-fatal)."""
    try:
        import torch  # noqa: PLC0415 - deliberate lazy import

        if torch.cuda.is_available():
            return GPUInfo(
                available=True,
                name=torch.cuda.get_device_name(0),
                cuda_version=torch.version.cuda or "unknown",
                torch_available=True,
                note="",
            )
        return GPUInfo(
            torch_available=True,
            note="Running in CPU-compatible mode (no CUDA-capable GPU found)",
        )
    except ImportError:
        note = "PyTorch is not installed — running in CPU-compatible mode"
        if try_nvidia_smi and shutil.which("nvidia-smi"):
            note = "NVIDIA driver detected but PyTorch is unavailable — CPU mode"
        return GPUInfo(torch_available=False, note=note)
    except Exception as exc:  # torch present but broken/unusable
        logger.debug("PyTorch CUDA check failed: %s", exc)
        return GPUInfo(
            torch_available=True,
            note=f"PyTorch found but CUDA is unusable — CPU mode ({exc.__class__.__name__})",
        )


def get_device(enable_gpu_if_available: bool = True) -> str:
    """Return the preferred computation device: 'cuda' or 'cpu'."""
    if not enable_gpu_if_available:
        return "cpu"
    try:
        import torch  # noqa: PLC0415 - deliberate lazy import

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def collect_system_info(enable_gpu_if_available: bool = True) -> SystemInfo:
    """Collect a full snapshot of system information and preferred device."""
    os_name, os_version = get_os_info()
    info = SystemInfo(
        python_version=get_python_version(),
        os_name=os_name,
        os_version=os_version,
        cpu=get_cpu_info(),
        ram=get_ram_info(),
        gpu=check_gpu(),
        preferred_device=get_device(enable_gpu_if_available),
    )
    logger.info(
        "System detected: Python %s | %s | CPU cores=%s | RAM %.1fGB | device=%s",
        info.python_version, info.os_name, info.cpu.logical_cores,
        info.ram.total_gb, info.preferred_device,
    )
    return info
