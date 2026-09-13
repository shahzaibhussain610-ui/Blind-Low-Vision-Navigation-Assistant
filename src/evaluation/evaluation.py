"""Performance evaluation & metrics (Part 10).

Measures actual behavior — never fabricated values: processed frames,
detections, warnings, FPS, per-stage latency, and CPU/RAM usage.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import psutil

from src.logging.logger import get_logger
from src.navigation import NavigationCommand

logger = get_logger("evaluation")


@dataclass
class SystemUsage:
    """Instantaneous CPU / RAM usage."""

    cpu_percent: float
    ram_used_percent: float
    ram_available_gb: float

    def to_dict(self) -> dict:
        return {
            "cpu_percent": round(self.cpu_percent, 1),
            "ram_used_percent": round(self.ram_used_percent, 1),
            "ram_available_gb": round(self.ram_available_gb, 1),
        }


@dataclass
class EvaluationMetrics:
    """Rolling session metrics for the complete pipeline."""

    frames_processed: int = 0
    frames_skipped: int = 0
    total_detections: int = 0
    tracking_warnings: int = 0       # CAUTION / SLOW_DOWN / direction changes
    stop_warnings: int = 0
    command_counts: dict = field(default_factory=dict)
    pipeline_ms: deque = field(default_factory=lambda: deque(maxlen=120))
    detection_ms: deque = field(default_factory=lambda: deque(maxlen=120))
    start_time: float = field(default_factory=time.perf_counter)

    # ------------------------------------------------------------------ #
    def record(self, pipeline_result) -> None:
        """Record one pipeline frame result."""
        if pipeline_result.skipped:
            self.frames_skipped += 1
            return
        self.frames_processed += 1
        self.total_detections += (
            pipeline_result.tracking.count if pipeline_result.tracking else 0
        )
        if pipeline_result.navigation:
            command = pipeline_result.navigation.command
            self.command_counts[command.value] = (
                self.command_counts.get(command.value, 0) + 1
            )
            if command in (NavigationCommand.CAUTION, NavigationCommand.SLOW_DOWN,
                           NavigationCommand.MOVE_LEFT, NavigationCommand.MOVE_RIGHT):
                self.tracking_warnings += 1
            elif command is NavigationCommand.STOP:
                self.stop_warnings += 1
        if pipeline_result.processing_time_ms > 0:
            self.pipeline_ms.append(pipeline_result.processing_time_ms)
        if (pipeline_result.detection is not None
                and pipeline_result.detection.inference_time_ms > 0):
            self.detection_ms.append(pipeline_result.detection.inference_time_ms)

    def reset(self) -> None:
        self.frames_processed = 0
        self.frames_skipped = 0
        self.total_detections = 0
        self.tracking_warnings = 0
        self.stop_warnings = 0
        self.command_counts.clear()
        self.pipeline_ms.clear()
        self.detection_ms.clear()
        self.start_time = time.perf_counter()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _avg(values: deque) -> float:
        return sum(values) / len(values) if values else 0.0

    @property
    def avg_pipeline_ms(self) -> float:
        return self._avg(self.pipeline_ms)

    @property
    def avg_detection_ms(self) -> float:
        return self._avg(self.detection_ms)

    @property
    def avg_fps(self) -> float:
        """Measured end-to-end FPS over the session."""
        elapsed = time.perf_counter() - self.start_time
        if elapsed <= 0:
            return 0.0
        return self.frames_processed / elapsed

    @property
    def inference_fps(self) -> float:
        avg = self.avg_detection_ms
        return 1000.0 / avg if avg > 0 else 0.0

    def summary(self) -> dict:
        return {
            "frames_processed": self.frames_processed,
            "frames_skipped": self.frames_skipped,
            "total_detections": self.total_detections,
            "warnings": self.tracking_warnings,
            "stops": self.stop_warnings,
            "command_counts": dict(self.command_counts),
            "avg_pipeline_ms": round(self.avg_pipeline_ms, 1),
            "avg_detection_ms": round(self.avg_detection_ms, 1),
            "avg_fps": round(self.avg_fps, 2),
            "inference_fps": round(self.inference_fps, 2),
        }

    @staticmethod
    def system_usage() -> SystemUsage:
        """Sample current CPU / RAM usage."""
        vm = psutil.virtual_memory()
        return SystemUsage(
            cpu_percent=psutil.cpu_percent(interval=None),
            ram_used_percent=vm.percent,
            ram_available_gb=vm.available / (1024 ** 3),
        )
