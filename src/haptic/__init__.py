"""Haptic package (Part 9): virtual three-motor belt simulation."""

from src.haptic.haptic_belt import (
    HapticPattern,
    Motor,
    MotorState,
    PATTERN_LIBRARY,
    VirtualHapticBelt,
    build_pattern,
)

__all__ = ["HapticPattern", "Motor", "MotorState", "PATTERN_LIBRARY",
           "VirtualHapticBelt", "build_pattern"]
