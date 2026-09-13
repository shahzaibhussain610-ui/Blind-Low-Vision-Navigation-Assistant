"""Reusable system-status registry.

Tracks readiness of the foundation and exposes a registry where future
modules (camera, detector, depth, tracker, navigation, haptic) can
register their own status without modifying this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from src.config.config_manager import ConfigManager
from src.logging.logger import get_logger
from src.system.system_info import SystemInfo

logger = get_logger("system_status")


@dataclass
class ModuleStatus:
    """Status of one application module."""

    name: str
    state: str = "not_initialized"  # e.g. ready, not_implemented, error
    detail: str = ""
    implemented: bool = False


@dataclass
class SystemStatus:
    """Aggregated readiness of the application and its modules."""

    foundation_ready: bool = False
    configuration_loaded: bool = False
    logging_ready: bool = False
    cpu_available: bool = False
    gpu_available: bool = False
    preferred_device: str = "cpu"
    modules: Dict[str, ModuleStatus] = field(default_factory=dict)

    def register_module(
        self, name: str, state: str, detail: str = "", implemented: bool = False
    ) -> ModuleStatus:
        """Register (or update) the status of a module by name."""
        status = ModuleStatus(name=name, state=state, detail=detail, implemented=implemented)
        self.modules[name] = status
        logger.debug("Module status registered: %s -> %s", name, state)
        return status

    def module(self, name: str) -> ModuleStatus:
        """Return a module's status, or a placeholder when unregistered."""
        if name not in self.modules:
            return ModuleStatus(name=name, state="not_initialized")
        return self.modules[name]

    @property
    def ready(self) -> bool:
        """True when the core foundation is fully operational."""
        return all(
            (
                self.foundation_ready,
                self.configuration_loaded,
                self.logging_ready,
                self.cpu_available,
            )
        )

    def to_dict(self) -> dict:
        return {
            "foundation_ready": self.foundation_ready,
            "configuration_loaded": self.configuration_loaded,
            "logging_ready": self.logging_ready,
            "cpu_available": self.cpu_available,
            "gpu_available": self.gpu_available,
            "preferred_device": self.preferred_device,
            "modules": {k: v.state for k, v in self.modules.items()},
        }


def build_initial_status(config: ConfigManager, system: SystemInfo) -> SystemStatus:
    """Build the Part 1 status: foundation ready, future modules pending."""
    status = SystemStatus(
        foundation_ready=True,
        configuration_loaded=True,
        logging_ready=True,
        cpu_available=system.cpu.logical_cores > 0,
        gpu_available=system.gpu.available,
        preferred_device=system.preferred_device,
    )
    # Part 2: the camera/input subsystem is implemented; future AI
    # modules remain placeholders (Parts 3-10).
    status.register_module(
        "camera", state="available", detail="Webcam & video input (Part 2)",
        implemented=True,
    )
    status.register_module(
        "detector", state="not_loaded",
        detail="YOLO11 detection (Part 3) — model loads on demand",
        implemented=True,
    )
    status.register_module(
        "depth", state="available",
        detail="Relative depth estimation (Part 4) — heuristic + optional MiDaS",
        implemented=True,
    )
    status.register_module(
        "tracker", state="available",
        detail="Object tracking & movement (Part 5) — ByteTrack + history",
        implemented=True,
    )
    status.register_module(
        "spatial", state="available",
        detail="Spatial & walkable-area analysis (Part 6) — zones + path",
        implemented=True,
    )
    status.register_module(
        "risk", state="available",
        detail="Risk assessment engine (Part 7) — SAFE..CRITICAL",
        implemented=True,
    )
    status.register_module(
        "navigation", state="available",
        detail="Navigation decision engine (Part 8) — PROCEED..STOP",
        implemented=True,
    )
    status.register_module(
        "haptic", state="simulation_only",
        detail="Virtual haptic feedback (Part 9) — three-motor simulation",
        implemented=True,
    )
    status.register_module(
        "evaluation", state="available",
        detail="Dashboard analytics & evaluation (Part 10)",
        implemented=True,
    )
    logger.info("System status built: device=%s, ready=%s", status.preferred_device, status.ready)
    return status
