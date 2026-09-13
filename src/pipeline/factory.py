"""Build the complete pipeline from app configuration (Part 10)."""

from __future__ import annotations

from src.config.config_manager import ConfigManager
from src.depth import RelativeDepthEstimator
from src.detection import ClassFilter, YOLODetector
from src.haptic import VirtualHapticBelt
from src.navigation import NavigationEngine
from src.pipeline.pipeline import NavigationPipeline
from src.risk import RiskAssessor
from src.spatial import SpatialAnalyzer
from src.tracking import ObjectTracker


def build_detector(config: ConfigManager) -> YOLODetector:
    """Create a YOLODetector from configuration values."""
    cf = config.get("detection.class_filter", {}) or {}
    return YOLODetector(
        model_name=str(config.get("detection.model", "yolo11n.pt")),
        confidence=float(config.get("detection.confidence_threshold", 0.40)),
        iou=float(config.get("detection.iou_threshold", 0.45)),
        image_size=int(config.get("detection.image_size", 640)),
        max_detections=int(config.get("detection.max_detections", 100)),
        device=str(config.get("detection.device", "auto")),
        class_filter=ClassFilter(
            mode=str(cf.get("mode", "navigation")),
            custom_classes=set(cf.get("custom_classes", []) or []),
        ),
    )


def build_pipeline(config: ConfigManager) -> NavigationPipeline:
    """Construct every pipeline stage from configuration and wire them."""
    detector = build_detector(config)
    depth = RelativeDepthEstimator(
        mode=str(config.get("depth.mode", "heuristic")),
        near_threshold=float(config.get("depth.near_threshold", 0.62)),
        far_threshold=float(config.get("depth.far_threshold", 0.35)),
    )
    tracker = ObjectTracker(
        detector=detector,
        tracker_config=str(config.get("tracking.tracker", "bytetrack.yaml")),
        history_length=int(config.get("tracking.history_length", 12)),
        static_threshold_px=float(config.get("tracking.static_threshold_px", 6.0)),
        approaching_depth_delta=float(config.get("tracking.approaching_depth_delta", 0.03)),
        min_history=int(config.get("tracking.min_history", 3)),
    )
    spatial = SpatialAnalyzer(
        path_width_ratio=float(config.get("spatial.path_width_ratio", 0.40)),
        path_min_depth=float(config.get("spatial.path_min_depth", 0.30)),
    )
    risk = RiskAssessor(
        critical_depth=float(config.get("risk.critical_depth", 0.45)),
        high_depth=float(config.get("risk.high_depth", 0.35)),
        medium_depth=float(config.get("risk.medium_depth", 0.22)),
        approaching_boost=int(config.get("risk.approaching_boost", 1)),
        low_confidence=float(config.get("risk.low_confidence", 0.55)),
        hazard_classes=list(config.get("risk.hazard_classes", []) or []),
    )
    navigation = NavigationEngine(
        min_command_interval=int(config.get("navigation.min_command_interval", 8)),
        hysteresis_frames=int(config.get("navigation.hysteresis_frames", 3)),
        uncertainty_stop=bool(config.get("navigation.uncertainty_stop", True)),
    )
    haptic = VirtualHapticBelt(
        pulse_interval_s=float(config.get("haptic.pulse_interval_s", 0.4)),
        enabled=bool(config.get("haptic.enabled", True)),
    )
    frame_skip = int(config.get("input.processing.frame_skip", 0))
    return NavigationPipeline(
        detector=detector, depth_estimator=depth, tracker=tracker,
        spatial_analyzer=spatial, risk_assessor=risk,
        navigation_engine=navigation, haptic_belt=haptic,
        frame_skip=frame_skip,
    )
