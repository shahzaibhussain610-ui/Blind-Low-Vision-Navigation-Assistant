"""Spatial & walkable-area analysis (Part 6).

Divides the view into LEFT / CENTER / RIGHT zones, defines a walking-path
corridor, and determines which objects obstruct it. Produces structured
results only — final risk (Part 7) and navigation (Part 8) decisions are
made downstream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

from src.detection.detection_result import Detection, ImagePosition
from src.logging.logger import get_logger
from src.tracking.tracker import MovementStatus, TrackedObject

logger = get_logger("spatial")


class ZoneState(str, Enum):
    """Obstruction state of a horizontal zone."""

    CLEAR = "CLEAR"
    PARTIALLY_BLOCKED = "PARTIALLY_BLOCKED"
    BLOCKED = "BLOCKED"


@dataclass
class SpatialObject:
    """Spatial analysis output for one tracked object."""

    track_id: int
    class_name: str
    position: ImagePosition
    in_walking_path: bool
    relative_depth: float
    depth_label: str
    overlap_ratio: float  # fraction of the path corridor covered by the box
    movement: MovementStatus

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "position": self.position.value,
            "in_walking_path": self.in_walking_path,
            "relative_depth": round(self.relative_depth, 3),
            "depth_label": self.depth_label,
            "overlap_ratio": round(self.overlap_ratio, 3),
            "movement": self.movement.value,
        }


@dataclass
class SpatialResult:
    """Structured spatial analysis for one frame."""

    frame_number: int
    frame_width: int
    frame_height: int
    path_x1: int
    path_x2: int
    objects: List[SpatialObject] = field(default_factory=list)
    zone_states: Dict[str, str] = field(default_factory=dict)
    processing_time_ms: float = 0.0

    @property
    def path_blocked(self) -> bool:
        """True when the walking corridor is fully blocked."""
        return ZoneState.BLOCKED.value in self.zone_states.values()

    @property
    def path_obstacles(self) -> List[SpatialObject]:
        """Objects inside the walking path."""
        return [o for o in self.objects if o.in_walking_path]

    def zone_is_clear(self, zone: str) -> bool:
        return self.zone_states.get(zone) == ZoneState.CLEAR.value

    @property
    def clear_zones(self) -> list:
        return [z for z, s in self.zone_states.items() if s == ZoneState.CLEAR.value]

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "path": [self.path_x1, self.path_x2],
            "zones": dict(self.zone_states),
            "path_blocked": self.path_blocked,
            "objects": [o.to_dict() for o in self.objects],
        }


class SpatialAnalyzer:
    """Zone + walking-path analysis over tracked objects."""

    def __init__(
        self,
        path_width_ratio: float = 0.40,
        path_min_depth: float = 0.30,
    ) -> None:
        self.path_width_ratio = float(path_width_ratio)
        self.path_min_depth = float(path_min_depth)

    def analyze(
        self,
        frame_width: int,
        frame_height: int,
        tracked_objects: list,
        frame_number: int = 0,
    ) -> SpatialResult:
        """Analyze one frame's tracked objects spatially."""
        start = time.perf_counter()
        width = max(int(frame_width), 1)
        corridor = width * self.path_width_ratio
        path_x1 = int((width - corridor) / 2)
        path_x2 = int((width + corridor) / 2)

        objects = [
            self._analyze_object(tracked, path_x1, path_x2)
            for tracked in tracked_objects
        ]
        result = SpatialResult(
            frame_number=frame_number,
            frame_width=frame_width,
            frame_height=frame_height,
            path_x1=path_x1,
            path_x2=path_x2,
            objects=objects,
            zone_states=self._evaluate_zones(objects),
            processing_time_ms=(time.perf_counter() - start) * 1000.0,
        )
        logger.debug(
            "Spatial analysis frame=%s zones=%s path_objects=%d",
            frame_number, result.zone_states, len(result.path_obstacles),
        )
        return result

    # ------------------------------------------------------------------ #
    def _analyze_object(self, tracked: TrackedObject, path_x1: int, path_x2: int) -> SpatialObject:
        detection: Detection = tracked.detection
        overlap = self._path_overlap_ratio(detection, path_x1, path_x2)
        # In the walking path when it horizontally overlaps the corridor
        # AND is near enough (relative depth) to matter.
        in_path = overlap > 0.0 and tracked.relative_depth >= self.path_min_depth
        return SpatialObject(
            track_id=tracked.track_id,
            class_name=tracked.class_name,
            position=detection.position,
            in_walking_path=in_path,
            relative_depth=tracked.relative_depth,
            depth_label=tracked.depth_label,
            overlap_ratio=overlap,
            movement=tracked.movement,
        )

    @staticmethod
    def _path_overlap_ratio(detection: Detection, path_x1: int, path_x2: int) -> float:
        """Fraction of the path corridor's width covered by the box."""
        overlap = min(detection.x2, path_x2) - max(detection.x1, path_x1)
        if overlap <= 0:
            return 0.0
        corridor_width = max(path_x2 - path_x1, 1)
        return min(overlap / corridor_width, 1.0)

    def _evaluate_zones(self, objects: List[SpatialObject]) -> Dict[str, str]:
        """Classify LEFT / CENTER / RIGHT zones as clear or blocked.

        A zone is PARTIALLY_BLOCKED when it contains a near object,
        and BLOCKED when that object substantially covers the walking
        corridor or is approaching.
        """
        states: Dict[str, str] = {}
        for zone in ("LEFT", "CENTER", "RIGHT"):
            near_in_zone = [
                o for o in objects
                if o.position.value == zone
                and o.relative_depth >= self.path_min_depth
            ]
            if not near_in_zone:
                states[zone] = ZoneState.CLEAR.value
            elif any(
                (o.in_walking_path and o.overlap_ratio >= 0.5)
                or o.movement is MovementStatus.APPROACHING
                for o in near_in_zone
            ):
                states[zone] = ZoneState.BLOCKED.value
            else:
                states[zone] = ZoneState.PARTIALLY_BLOCKED.value
        return states
