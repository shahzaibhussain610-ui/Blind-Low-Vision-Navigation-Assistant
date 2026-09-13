"""Tests for the virtual haptic belt (Part 9)."""

import pytest

from src.haptic import (
    HapticPattern,
    Motor,
    MotorState,
    PATTERN_LIBRARY,
    VirtualHapticBelt,
)
from src.navigation import NavigationCommand


class TestPatternMapping:
    def test_proceed_is_silent(self):
        pattern = build(NavigationCommand.PROCEED)
        assert all(not m.active for m in pattern.motors.values())

    def test_move_left_activates_left_motor(self):
        pattern = build(NavigationCommand.MOVE_LEFT)
        assert pattern.motors[Motor.LEFT].active
        assert not pattern.motors[Motor.CENTER].active
        assert not pattern.motors[Motor.RIGHT].active

    def test_move_right_activates_right_motor(self):
        pattern = build(NavigationCommand.MOVE_RIGHT)
        assert pattern.motors[Motor.RIGHT].active
        assert not pattern.motors[Motor.LEFT].active

    def test_caution_pulses_all(self):
        pattern = build(NavigationCommand.CAUTION)
        assert all(m.active and m.pulsing for m in pattern.motors.values())

    def test_slow_down_pulses_all(self):
        pattern = build(NavigationCommand.SLOW_DOWN)
        assert all(m.active and m.pulsing for m in pattern.motors.values())

    def test_stop_strongest_on_all(self):
        pattern = build(NavigationCommand.STOP)
        assert all(m.active and m.pulsing for m in pattern.motors.values())
        stop = build(NavigationCommand.STOP)
        caution = build(NavigationCommand.CAUTION)
        assert (stop.motors[Motor.CENTER].intensity
                > caution.motors[Motor.CENTER].intensity)

    def test_all_commands_have_patterns(self):
        for command in NavigationCommand:
            assert command in PATTERN_LIBRARY


def build(command):
    from src.haptic.haptic_belt import build_pattern

    return build_pattern(command)


class TestVirtualHapticBelt:
    def test_belt_starts_silent(self):
        belt = VirtualHapticBelt()
        assert belt.current_command is NavigationCommand.PROCEED
        assert not any(m.active for m in belt.motors.values())

    def test_apply_command_updates_motors(self):
        belt = VirtualHapticBelt()
        pattern = belt.apply_command(NavigationCommand.MOVE_LEFT)
        assert belt.current_command is NavigationCommand.MOVE_LEFT
        assert pattern.motors[Motor.LEFT].active
        assert belt.motors[Motor.LEFT].active

    def test_disable_silences_belt(self):
        belt = VirtualHapticBelt(enabled=False)
        pattern = belt.apply_command(NavigationCommand.STOP)
        assert all(not m.active for m in pattern.motors.values())

    def test_test_pattern_forces_state(self):
        belt = VirtualHapticBelt()
        belt.test_pattern(NavigationCommand.MOVE_RIGHT)
        assert belt.motors[Motor.RIGHT].active

    def test_reset(self):
        belt = VirtualHapticBelt()
        belt.apply_command(NavigationCommand.STOP)
        belt.reset()
        assert not any(m.active for m in belt.motors.values())

    def test_to_dict(self):
        belt = VirtualHapticBelt()
        belt.apply_command(NavigationCommand.STOP)
        data = belt.to_dict()
        assert data["command"] == "STOP"
        assert set(data["motors"].keys()) == {"LEFT", "CENTER", "RIGHT"}


class TestFutureHardwareReady:
    def test_interface_is_replaceable(self):
        """Documentation test: belt interface is hardware-replaceable."""
        belt = VirtualHapticBelt()
        # The methods a hardware driver must expose:
        assert callable(belt.apply_command)
        assert callable(belt.apply_decision)
        assert callable(belt.reset)
        assert isinstance(belt.motors, dict)
