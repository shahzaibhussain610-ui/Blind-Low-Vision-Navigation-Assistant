"""YOLO11 object detector built on the Ultralytics framework.

The detector is initialized once and reused for every frame. The rest
of the application depends only on project-specific
:class:`DetectionResult` objects, never on Ultralytics result objects.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from src.camera.frame import FramePacket
from src.detection.class_filter import ClassFilter
from src.detection.detection_result import Detection, DetectionResult
from src.logging.logger import get_logger
from src.system.system_info import get_device

logger = get_logger("detector")


class ModelLoadError(Exception):
    """Raised when the YOLO model cannot be loaded."""


class InferenceError(Exception):
    """Raised when inference fails on a valid frame."""


class YOLODetector:
    """Reusable YOLO11 detector.

    Args:
        model_name: Model name or path (e.g. ``"yolo11n.pt"``).
        confidence: Confidence threshold in (0, 1).
        iou: IoU threshold for NMS.
        image_size: Inference input size (pixels, multiple of 32).
        max_detections: Maximum detections per frame.
        device: ``"auto"`` (use Part 1 device selection), ``"cpu"``,
            or ``"cuda"``.
        class_filter: Configured :class:`ClassFilter`.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence: float = 0.40,
        iou: float = 0.45,
        image_size: int = 640,
        max_detections: int = 100,
        device: str = "auto",
        class_filter: Optional[ClassFilter] = None,
    ) -> None:
        if not (0.0 < confidence < 1.0):
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        self.model_name = model_name
        self.confidence_threshold = float(confidence)
        self.iou_threshold = float(iou)
        self.image_size = int(image_size)
        self.max_detections = int(max_detections)
        self._device_request = device
        self.class_filter = class_filter or ClassFilter()
        self._model = None
        self._device = "cpu"
        self._warm = False
        self._unavailable_classes: list = []

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        """Load the YOLO model once and resolve the compute device.

        Raises:
            ModelLoadError: When the model cannot be loaded.
        """
        if self.is_loaded:
            return
        # Device selection via Part 1: CUDA only if available + enabled.
        if self._device_request == "auto":
            self._device = get_device(enable_gpu_if_available=True)
        else:
            self._device = self._device_request
        try:
            from ultralytics import YOLO  # local import; heavy dependency
        except ImportError as exc:
            raise ModelLoadError(
                "Ultralytics is not installed. Please install it "
                "(see requirements.txt) to enable object detection."
            ) from exc
        try:
            logger.info(
                "YOLO model loading: %s (device=%s, conf=%.2f, iou=%.2f, imgsz=%d)",
                self.model_name, self._device, self.confidence_threshold,
                self.iou_threshold, self.image_size,
            )
            self._model = YOLO(self.model_name)
        except Exception as exc:
            logger.error("YOLO model loading failed: %s", exc)
            raise ModelLoadError(
                f"Unable to load YOLO11 model '{self.model_name}'. "
                "Please verify the model configuration and installation."
            ) from exc
        # Bind the filter to the model's real class mapping.
        names = getattr(self._model, "names", {}) or {}
        self._unavailable_classes = self.class_filter.set_model_classes(names)
        logger.info("Model successfully loaded: %s (%d classes)", self.model_name, len(names))
        self.warmup()

    @property
    def unavailable_classes(self) -> list:
        """Requested classes missing from the pretrained model."""
        return list(self._unavailable_classes)

    @property
    def class_names(self) -> dict:
        """The loaded model's {class_id: class_name} mapping."""
        if self._model is None:
            return {}
        return dict(getattr(self._model, "names", {}) or {})

    def warmup(self) -> None:
        """Run one controlled warmup inference on a blank frame.

        The first inference triggers initialization (kernel selection,
        graph tracing) and is much slower than steady state. Warmup keeps
        that cost out of measured performance.
        """
        if self._model is None or self._warm:
            return
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        start = time.perf_counter()
        try:
            self._run_model(blank)
            self._warm = True
            logger.info(
                "Model warmup completed in %.0f ms (not included in steady-state metrics)",
                (time.perf_counter() - start) * 1000.0,
            )
        except Exception as exc:  # warmup is best-effort
            logger.warning("Model warmup failed (non-fatal): %s", exc)
            self._warm = True

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _run_model(self, frame: np.ndarray):
        """Invoke the Ultralytics model with configured parameters."""
        return self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            max_det=self.max_detections,
            device=self._device,
            verbose=False,
        )

    def detect_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> DetectionResult:
        """Run detection on one BGR frame.

        Invalid frames never reach the model.

        Raises:
            InferenceError: When inference fails despite a valid frame.
        """
        if not FramePacket.validate(frame):
            logger.warning("Invalid frame: inference skipped (frame_number=%s)", frame_number)
            return DetectionResult(
                frame_number=frame_number, timestamp=timestamp,
                model_name=self.model_name, device=self._device,
            )
        if not self.is_loaded:
            self.load()
        height, width = frame.shape[:2]
        try:
            start = time.perf_counter()
            results = self._run_model(frame)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        except Exception as exc:
            logger.error("Inference failure on frame %s: %s", frame_number, exc)
            raise InferenceError(f"YOLO inference failed: {exc}") from exc
        detections = self._extract_detections(
            results, width, height, frame_number, timestamp
        )
        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
        return DetectionResult(
            frame_number=frame_number,
            timestamp=timestamp,
            detections=detections,
            inference_time_ms=elapsed_ms,
            inference_fps=fps,
            model_name=self.model_name,
            device=self._device,
            frame_width=width,
            frame_height=height,
        )

    def detect(self, packet: FramePacket) -> DetectionResult:
        """Convenience wrapper: run detection on a Part 2 FramePacket."""
        return self.detect_frame(packet.frame, packet.frame_number, packet.timestamp)

    # ------------------------------------------------------------------ #
    # Conversion
    # ------------------------------------------------------------------ #
    def _extract_detections(self, results, width: int, height: int,
                            frame_number: int, timestamp: float) -> list:
        """Convert raw Ultralytics output into project Detection objects."""
        detections: list = []
        if not results:
            return detections
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        names = getattr(self._model, "names", {})
        allowed = self.class_filter.allowed_ids()
        for box in boxes:
            try:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            except Exception as exc:  # malformed box entry
                logger.debug("Skipping malformed box: %s", exc)
                continue
            if allowed is not None and class_id not in allowed:
                continue
            detection = Detection(
                class_id=class_id,
                class_name=names.get(class_id, str(class_id)),
                confidence=confidence,
                x1=x1, y1=y1, x2=x2, y2=y2,
                frame_number=frame_number,
                timestamp=timestamp,
                frame_width=width,
                frame_height=height,
            ).clamp_to_frame(width, height)
            if detection.is_valid(width, height):
                detections.append(detection)
        return detections

    def release(self) -> None:
        """Drop the model reference (frees memory when garbage collected)."""
        if self._model is not None:
            self._model = None
            logger.info("Detector released")
