"""Navigation decision engine (Part 8).

Converts spatial + risk information into one of: PROCEED, MOVE_LEFT,
MOVE_RIGHT, SLOW_DOWN, CAUTION, STOP. Safety-first: STOP beats every
other command; direction changes are debounced with hysteresis to avoid
unnecessary flipping.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.logging.logger import get_logger
from src.risk.risk_assessor import RiskAssessment, RiskLevel
from src.spatial.spatial_analyzer import SpatialResult
from src.tracking.tracker import MovementStatus

logger = get_logger("navigation")


class NavigationCommand(str, Enum):
    """Supported navigation commands (Part 8)."""

    PROCEED = "PROCEED"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    SLOW_DOWN = "SLOW_DOWN"
    CAUTION = "CAUTION"
    STOP = "STOP"


#: Severity ordering used for hysteresis (higher = more urgent).
_SEVERITY = {
    NavigationCommand.PROCEED: 0,
    NavigationCommand.CAUTION: 1,
    NavigationCommand.SLOW_DOWN: 2,
    NavigationCommand.MOVE_LEFT: 2,
    NavigationCommand.MOVE_RIGHT: 2,
    NavigationCommand.STOP: 3,
}


@dataclass
class NavigationDecision:
    """One navigation decision with its human-readable reason."""

    frame_number: int
    command: NavigationCommand
    reason: str
    held_frames: int = 0

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "command": self.command.value,
            "reason": self.reason,
            "held_frames": self.held_frames,
        }


class NavigationEngine:
    """Safety-first navigation command selection with hysteresis."""

    def __init__(
        self,
        min_command_interval: int = 8,
        hysteresis_frames: int = 3,
        uncertainty_stop: bool = True,
    ) -> None:
        self.min_command_interval = int(min_command_interval)
        self.hysteresis_frames = max(int(hysteresis_frames), 1)
        self.uncertainty_stop = bool(uncertainty_stop)
        self._pending: Optional[NavigationCommand] = None
        self._pending_count = 0
        self._last_change_frame = -10**9
        self._current = NavigationCommand.PROCEED
        self._current_reason = "No obstacles detected"

    @property
    def current(self) -> NavigationCommand:
        return self._current

    @property
    def current_reason(self) -> str:
        return self._current_reason

    # ------------------------------------------------------------------ #
    def decide(
        self,
        spatial: SpatialResult,
        risk: RiskAssessment,
        frame_number: int = 0,
    ) -> NavigationDecision:
        """Compute the navigation command for one frame."""
        desired, reason = self._evaluate(spatial, risk)
        command = self._apply_hysteresis(desired, frame_number)
        if command is self._current:
            # Keep the reason fresh even while the command is held.
            self._current_reason = reason
            held = frame_number - self._last_change_frame
            decision = NavigationDecision(
                frame_number=frame_number,
                command=command,
                reason=self._current_reason,
                held_frames=max(held, 0),
            )
        else:
            self._current = command
            self._current_reason = reason
            self._last_change_frame = frame_number
            decision = NavigationDecision(
                frame_number=frame_number, command=command, reason=reason
            )
        logger.info(
            "Navigation decision frame=%d: %s (%s)",
            frame_number, decision.command.value, decision.reason,
        )
        return decision

    # ------------------------------------------------------------------ #
    def _evaluate(self, spatial: SpatialResult, risk: RiskAssessment):
        """Pure logic: pick (command, reason) from spatial + risk input."""
        obstacles = spatial.path_obstacles
        highest = risk.highest

        # STOP: critical risk in/near the path.
        if highest >= RiskLevel.CRITICAL:
            target = risk.highest_object
            return NavigationCommand.STOP, (
                f"{target.class_name} (track {target.track_id}) critically close"
            )

        # STOP: uncertain object inside the walking path (conservative).
        if self.uncertainty_stop:
            uncertain = [o for o in obstacles if o.movement is MovementStatus.UNCERTAIN]
            if uncertain:
                return NavigationCommand.STOP, (
                    f"uncertain obstacle in path (track {uncertain[0].track_id})"
                )

        if not obstacles:
            return NavigationCommand.PROCEED, "walking path is clear"

        # Path has non-critical obstacles: prefer the clearest side.
        left_clear = spatial.zone_is_clear("LEFT")
        right_clear = spatial.zone_is_clear("RIGHT")
        if left_clear and not right_clear:
            return NavigationCommand.MOVE_LEFT, "obstacle ahead; left side is clear"
        if right_clear and not left_clear:
            return NavigationCommand.MOVE_RIGHT, "obstacle ahead; right side is clear"
        if left_clear and right_clear:
            return NavigationCommand.MOVE_LEFT, (
                "obstacle ahead; both sides clear (defaulting left)"
            )

        # Neither side fully clear: grade by risk level.
        if highest >= RiskLevel.HIGH:
            return NavigationCommand.STOP, "path blocked and no clear side"
        if highest >= RiskLevel.MEDIUM:
            return NavigationCommand.SLOW_DOWN, "partially blocked path; slowing down"
        return NavigationCommand.CAUTION, "minor obstacle in path; proceed with care"

    def _apply_hysteresis(self, desired: NavigationCommand, frame_number: int) -> NavigationCommand:
        """Debounce non-critical command changes."""
        if desired is self._current or desired is NavigationCommand.STOP:
            self._pending = None
            self._pending_count = 0
            return desired
        if frame_number - self._last_change_frame < self.min_command_interval:
            return self._current
        if desired is self._pending:
            self._pending_count += 1
        else:
            self._pending = desired
            self._pending_count = 1
        if self._pending_count >= self.hysteresis_frames:
            self._pending = None
            self._pending_count = 0
            return desired
        return self._current

    def reset(self) -> None:
        """Clear internal state (e.g. on input source change)."""
        self._pending = None
        self._pending_count = 0
        self._current = NavigationCommand.PROCEED
        self._current_reason = "Reset"
        self._last_change_frame = -10**9
        logger.info("Navigation engine reset")
