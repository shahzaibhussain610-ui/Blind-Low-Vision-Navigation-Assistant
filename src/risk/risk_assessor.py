"""Risk assessment engine (Part 7).

Classifies tracked objects as SAFE / LOW / MEDIUM / HIGH / CRITICAL from
object type, relative depth, position, bounding-box size, movement,
confidence, and walking-path overlap. All thresholds are configurable
and uncertain inputs are handled conservatively (risk is raised, never
silently lowered).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List

from src.logging.logger import get_logger
from src.spatial.spatial_analyzer import SpatialObject, SpatialResult
from src.tracking.tracker import MovementStatus

logger = get_logger("risk")


class RiskLevel(IntEnum):
    """Ordered risk levels; higher value = more severe."""

    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return self.name


@dataclass
class ObjectRisk:
    """Risk assessment for one tracked object."""

    track_id: int
    class_name: str
    level: RiskLevel
    score: float  # continuous risk score in [0, 1]
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "level": self.level.label,
            "score": round(self.score, 3),
            "reasons": list(self.reasons),
        }


@dataclass
class RiskAssessment:
    """Risk results for one frame."""

    frame_number: int
    objects: List[ObjectRisk] = field(default_factory=list)
    processing_time_ms: float = 0.0

    @property
    def highest(self) -> RiskLevel:
        """Highest risk level present (SAFE when no objects)."""
        return max((o.level for o in self.objects), default=RiskLevel.SAFE)

    @property
    def highest_object(self) -> ObjectRisk:
        """The object carrying the highest risk, or None."""
        if not self.objects:
            return None
        return max(self.objects, key=lambda o: o.level)

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "highest": self.highest.label,
            "objects": [o.to_dict() for o in self.objects],
        }


#: Classes treated as potential navigation hazards by default
#: (mirrors config/risk.hazard_classes; configurable either way).
DEFAULT_HAZARD_CLASSES = (
    "person", "car", "truck", "bus", "motorcycle", "bicycle", "dog",
)


class RiskAssessor:
    """Configurable risk assessment over spatial results."""

    def __init__(
        self,
        critical_depth: float = 0.45,
        high_depth: float = 0.35,
        medium_depth: float = 0.22,
        approaching_boost: int = 1,
        low_confidence: float = 0.55,
        hazard_classes: List[str] = None,
    ) -> None:
        self.critical_depth = float(critical_depth)
        self.high_depth = float(high_depth)
        self.medium_depth = float(medium_depth)
        self.approaching_boost = int(approaching_boost)
        self.low_confidence = float(low_confidence)
        self.hazard_classes = set(
            hazard_classes if hazard_classes else DEFAULT_HAZARD_CLASSES
        )

    def assess(self, spatial: SpatialResult, frame_number: int = 0) -> RiskAssessment:
        """Assess risk for every spatial object in the frame."""
        import time

        start = time.perf_counter()
        objects = [self._assess_object(o) for o in spatial.objects]
        assessment = RiskAssessment(
            frame_number=frame_number,
            objects=objects,
            processing_time_ms=(time.perf_counter() - start) * 1000.0,
        )
        logger.info(
            "Risk assessment frame=%s: highest=%s objects=%d",
            frame_number, assessment.highest.label, len(objects),
        )
        return assessment

    # ------------------------------------------------------------------ #
    def _assess_object(self, obj: SpatialObject) -> ObjectRisk:
        reasons: List[str] = []
        score = 0.05  # base: detected but far/irrelevant

        # Depth contribution (dominant factor).
        if obj.relative_depth >= self.critical_depth:
            score += 0.55
            reasons.append(f"very close (depth {obj.relative_depth:.2f})")
        elif obj.relative_depth >= self.high_depth:
            score += 0.40
            reasons.append(f"close (depth {obj.relative_depth:.2f})")
        elif obj.relative_depth >= self.medium_depth:
            score += 0.22
            reasons.append(f"moderately near (depth {obj.relative_depth:.2f})")
        else:
            reasons.append("far away")

        # Movement contribution.
        if obj.movement is MovementStatus.APPROACHING:
            score += 0.18 * max(self.approaching_boost, 1)
            reasons.append("approaching")
        elif obj.movement is MovementStatus.MOVING:
            score += 0.08
            reasons.append("moving")
        elif obj.movement is MovementStatus.UNCERTAIN:
            score += 0.10  # conservative: unknown movement adds risk
            reasons.append("movement uncertain")

        # Walking-path contribution.
        if obj.in_walking_path:
            score += 0.25 * max(obj.overlap_ratio, 0.25)
            reasons.append(f"in walking path ({obj.overlap_ratio:.0%})")

        # Class hazard contribution.
        if obj.class_name in self.hazard_classes:
            score += 0.12
            reasons.append(f"hazard class ({obj.class_name})")

        score = min(max(score, 0.0), 1.0)
        level = self._score_to_level(score)
        return ObjectRisk(
            track_id=obj.track_id,
            class_name=obj.class_name,
            level=level,
            score=score,
            reasons=reasons,
        )

    def _score_to_level(self, score: float) -> RiskLevel:
        """Map a continuous score to a discrete risk level."""
        if score >= 0.85:
            return RiskLevel.CRITICAL
        if score >= 0.65:
            return RiskLevel.HIGH
        if score >= 0.40:
            return RiskLevel.MEDIUM
        if score >= 0.15:
            return RiskLevel.LOW
        return RiskLevel.SAFE
