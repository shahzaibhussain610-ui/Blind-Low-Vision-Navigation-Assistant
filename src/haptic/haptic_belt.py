"""Virtual haptic belt simulation (Part 9).

A software simulation of a three-motor haptic belt (LEFT / CENTER /
RIGHT virtual motors). Designed so a future ESP32 / physical-motor
integration can replace :class:`VirtualHapticBelt` with a hardware
driver exposing the same interface. No physical hardware is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.logging.logger import get_logger
from src.navigation.navigation_engine import NavigationCommand

logger = get_logger("haptic")


class Motor(str, Enum):
    """Virtual belt motor positions."""

    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"


@dataclass
class MotorState:
    """Simulated state of one virtual motor."""

    intensity: float = 0.0   # 0..1 vibration strength
    pulsing: bool = False    # whether the motor pulses over time
    pulse_hz: float = 0.0    # pulse frequency while active

    @property
    def active(self) -> bool:
        return self.intensity > 0.0

    def to_dict(self) -> dict:
        return {
            "intensity": round(self.intensity, 2),
            "active": self.active,
            "pulsing": self.pulsing,
            "pulse_hz": round(self.pulse_hz, 2),
        }


@dataclass
class HapticPattern:
    """A complete belt vibration pattern."""

    command: NavigationCommand
    description: str = ""
    motors: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "command": self.command.value,
            "description": self.description,
            "motors": {m.value: s.to_dict() for m, s in self.motors.items()},
        }


#: Command → (intensity, pulsing, pulse_hz, description).
PATTERN_LIBRARY = {
    NavigationCommand.PROCEED: (0.0, False, 0.0,
                                "No vibration — path is clear"),
    NavigationCommand.MOVE_LEFT: (0.7, False, 0.0,
                                  "Left motor pulse — move left"),
    NavigationCommand.MOVE_RIGHT: (0.7, False, 0.0,
                                   "Right motor pulse — move right"),
    NavigationCommand.CAUTION: (0.4, True, 2.0,
                                "Short repeated pulses — caution"),
    NavigationCommand.SLOW_DOWN: (0.6, True, 1.0,
                                  "Medium pulses — slow down"),
    NavigationCommand.STOP: (1.0, True, 6.0,
                             "Strong rapid pulses on all motors — stop"),
}


def build_pattern(command: NavigationCommand) -> HapticPattern:
    """Build the pattern mapped to a command (stateless helper)."""
    intensity, pulsing, pulse_hz, description = PATTERN_LIBRARY[command]
    motors = {motor: MotorState() for motor in Motor}
    if intensity > 0.0:
        if command is NavigationCommand.MOVE_LEFT:
            motors[Motor.LEFT] = MotorState(intensity, pulsing, pulse_hz)
        elif command is NavigationCommand.MOVE_RIGHT:
            motors[Motor.RIGHT] = MotorState(intensity, pulsing, pulse_hz)
        else:  # CAUTION / SLOW_DOWN / STOP: all motors
            for motor in motors:
                motors[motor] = MotorState(intensity, pulsing, pulse_hz)
    return HapticPattern(command=command, description=description, motors=motors)


class VirtualHapticBelt:
    """Three-motor virtual haptic belt (software simulation).

    Future hardware integration note: an ESP32/physical belt driver
    would implement the same ``apply_command`` / ``motors`` interface;
    this class is the reference implementation.
    """

    def __init__(self, pulse_interval_s: float = 0.4, enabled: bool = True) -> None:
        self.pulse_interval_s = float(pulse_interval_s)
        self.enabled = enabled
        self.current_command = NavigationCommand.PROCEED
        self.pattern = build_pattern(NavigationCommand.PROCEED)

    def apply_command(self, command: NavigationCommand) -> HapticPattern:
        """Activate the pattern mapped to a navigation command."""
        if command is not self.current_command:
            logger.info("Haptic pattern: %s", command.value)
        self.current_command = command
        self.pattern = build_pattern(command)
        if not self.enabled:  # simulate "belt off": silence all motors
            for motor in self.pattern.motors.values():
                motor.intensity = 0.0
        return self.pattern

    def apply_decision(self, decision) -> HapticPattern:
        """Apply a :class:`NavigationDecision` (Part 8)."""
        return self.apply_command(decision.command)

    def test_pattern(self, command: NavigationCommand) -> HapticPattern:
        """Force a pattern for UI testing (independent of navigation)."""
        logger.info("Haptic test pattern: %s", command.value)
        self.current_command = command
        self.pattern = build_pattern(command)
        return self.pattern

    def reset(self) -> None:
        """Silence the belt."""
        self.apply_command(NavigationCommand.PROCEED)

    @property
    def motors(self) -> dict:
        return self.pattern.motors

    def to_dict(self) -> dict:
        return {
            "command": self.current_command.value,
            "enabled": self.enabled,
            "description": self.pattern.description,
            "motors": {m.value: s.to_dict() for m, s in self.motors.items()},
        }
