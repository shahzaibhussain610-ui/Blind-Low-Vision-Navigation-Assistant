"""Full navigation pipeline (Part 10).

Orchestrates Part 2 input → Part 3 detection → Part 4 depth → Part 5
tracking → Part 6 spatial → Part 7 risk → Part 8 navigation → Part 9
haptic into one reusable engine. The dashboard only renders results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from src.depth import RelativeDepthEstimator
from src.detection.detection_result import DetectionResult
from src.detection.visualization import annotate
from src.haptic import VirtualHapticBelt
from src.logging.logger import get_logger
from src.navigation import NavigationDecision, NavigationEngine
from src.risk import RiskAssessment, RiskAssessor
from src.spatial import SpatialAnalyzer, SpatialResult
from src.tracking import ObjectTracker, TrackingResult

logger = get_logger("pipeline")


@dataclass
class PipelineFrameResult:
    """Everything produced for one processed frame."""

    detection: Optional[DetectionResult] = None
    tracking: Optional[TrackingResult] = None
    spatial: Optional[SpatialResult] = None
    risk: Optional[RiskAssessment] = None
    navigation: Optional[NavigationDecision] = None
    processing_time_ms: float = 0.0
    skipped: bool = False  # frame skipped by frame_skip policy


class NavigationPipeline:
    """Complete detection → … → haptic pipeline over Part 2 frames."""

    def __init__(
        self,
        detector,                       # YOLODetector (Part 3)
        depth_estimator: RelativeDepthEstimator,
        tracker: ObjectTracker,
        spatial_analyzer: SpatialAnalyzer,
        risk_assessor: RiskAssessor,
        navigation_engine: NavigationEngine,
        haptic_belt: VirtualHapticBelt,
        frame_skip: int = 0,
    ) -> None:
        self.detector = detector
        self.depth = depth_estimator
        self.tracker = tracker
        self.spatial = spatial_analyzer
        self.risk = risk_assessor
        self.navigation = navigation_engine
        self.haptic = haptic_belt
        self.frame_skip = max(int(frame_skip), 0)
        self._frame_counter = 0

    def process_frame(self, frame) -> PipelineFrameResult:
        """Run the complete pipeline on one BGR frame."""
        start = time.perf_counter()
        number = self._frame_counter
        self._frame_counter += 1

        result = PipelineFrameResult()
        if self.frame_skip > 0 and number % (self.frame_skip + 1) != 0:
            result.skipped = True
            return result

        # Part 5: tracking (runs YOLO + ByteTrack, yields detections + ids).
        tracking = self.tracker.update(frame, number, timestamp=number / 15.0)
        result.tracking = tracking

        # Part 4: relative depth per tracked object, then re-classify
        # movement with the fresh depth values.
        for tracked in tracking.tracked:
            depth = self.depth.estimate_one(tracked.detection)
            tracked.relative_depth = depth.relative_depth
            tracked.depth_label = depth.depth_label
        self.tracker.reclassify(tracking)

        # Parts 6-9: spatial → risk → navigation → haptic.
        spatial = self.spatial.analyze(
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            tracked_objects=tracking.tracked,
            frame_number=number,
        )
        result.spatial = spatial
        risk = self.risk.assess(spatial, frame_number=number)
        result.risk = risk
        decision = self.navigation.decide(spatial, risk, frame_number=number)
        result.navigation = decision
        self.haptic.apply_decision(decision)

        result.detection = self._to_detection_result(tracking)
        result.processing_time_ms = (time.perf_counter() - start) * 1000.0
        return result

        # ------------------------------------------------------------------ #
    def annotated_frame(self, frame, result: PipelineFrameResult):
        """Return the frame annotated with boxes, walking-path and status HUD.

        The HUD banner (green = clear, red = DANGER + reason) is drawn inline
        so every rendered/exported frame communicates near-object danger
        without the caller having to add anything manually.
        """
        from src.risk import RiskLevel  # lazy: avoids cycle at import time

        annotated = annotate(frame, result.detection)
        annotated = self._draw_path(annotated, result.spatial)
        annotated = self._draw_object_status(annotated, result.tracking)
        return self._draw_status_banner(annotated, result.risk, result.navigation, RiskLevel)

    # -- Per-object distance status (SAFE / CAUTION / DANGER) ------------- #
    #: Approximate scene range used to map relative depth (0..1, higher =
    #: nearer) to metres. Heuristic cameras have no true metric scale, so
    #: this is a calibration constant, not a measurement.
    MAX_RANGE_M = 12.0
    #: Objects estimated closer than this are flagged DANGER.
    DANGER_DISTANCE_M = 0.3
    #: Objects estimated closer than this are flagged CAUTION.
    CAUTION_DISTANCE_M = 1.0

    @classmethod
    def _object_status(cls, relative_depth: float):
        """Return (estimated metres, status label, BGR colour) for a depth."""
        distance_m = max(0.0, (1.0 - float(relative_depth))) * cls.MAX_RANGE_M
        if distance_m <= cls.DANGER_DISTANCE_M:
            return distance_m, "DANGER", (0, 0, 255)      # red
        if distance_m <= cls.CAUTION_DISTANCE_M:
            return distance_m, "CAUTION", (0, 140, 255)   # orange
        return distance_m, "SAFE", (60, 180, 75)          # green

    @classmethod
    def _draw_object_status(cls, frame, tracking):
        """Draw a distance + status label under every tracked object box."""
        if tracking is None:
            return frame
        import cv2

        for tracked in tracking.tracked:
            detection = tracked.detection
            distance_m, status, color = cls._object_status(tracked.relative_depth)
            x1, y1, x2, y2 = (int(round(v)) for v in detection.bbox)
            label = f"{detection.class_name} {distance_m:.1f}m {status}"
            scale = 0.45
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            top = y2 + 4
            if top + label_size[1] + 6 > frame.shape[0]:  # keep label on-screen
                top = max(y1 - label_size[1] - 6, 0)
            left = max(x1, 0)
            cv2.rectangle(
                frame, (left, top),
                (left + label_size[0] + 4, top + label_size[1] + 6),
                color, thickness=-1,
            )
            cv2.putText(
                frame, label, (left + 2, top + label_size[1] + 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA,
            )
        return frame

    @staticmethod
    def _draw_status_banner(
        frame, risk, navigation, _risk_level_cls
    ):
        """Overlay a top banner summarising risk + navigation command."""
        import cv2

        risk_label = risk.highest.label if risk is not None else "CLEAR"
        danger = risk is not None and risk.highest >= _risk_level_cls.HIGH
        command = navigation.command.value if navigation is not None else "—"
        reason = navigation.reason if navigation is not None else "no navigation info"
        if command == "STOP":
            danger = True

        height, width = frame.shape[:2]
        banner_h = max(54, int(height * 0.09))
        color = (0, 0, 220) if danger else (0, 130, 60)  # BGR: red / green
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, banner_h), color, thickness=-1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        label = f"DANGER — {risk_label}" if danger else f"STATUS — {risk_label}"
        cv2.putText(
            frame, label, (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"{command}: {reason}", (12, banner_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        return frame

    @staticmethod
    def _draw_path(frame, spatial: SpatialResult):
        """Overlay the walking-path corridor and zone state on a frame."""
        if spatial is None:
            return frame
        import cv2

        height = frame.shape[0]
        blocked = spatial.zone_states.get("CENTER") != "CLEAR"
        color = (0, 0, 255) if blocked else (0, 200, 0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (spatial.path_x1, 0), (spatial.path_x2, height),
                      color, thickness=-1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.line(frame, (spatial.path_x1, 0), (spatial.path_x1, height), color, 2)
        cv2.line(frame, (spatial.path_x2, 0), (spatial.path_x2, height), color, 2)
        return frame

    def reset(self) -> None:
        """Reset stateful stages (tracker history, navigation, haptic)."""
        self.tracker.reset()
        self.navigation.reset()
        self.haptic.reset()
        self._frame_counter = 0
        logger.info("Pipeline reset")

    def _to_detection_result(self, tracking: TrackingResult) -> DetectionResult:
        """Convert tracking output to a Part 3 result for display."""
        detections = [t.detection for t in tracking.tracked]
        avg_ms = tracking.processing_time_ms
        return DetectionResult(
            frame_number=tracking.frame_number,
            timestamp=tracking.timestamp,
            detections=detections,
            inference_time_ms=avg_ms,
            inference_fps=(1000.0 / avg_ms if avg_ms > 0 else 0.0),
            model_name=self.detector.model_name,
            device=self.detector.device,
        )
