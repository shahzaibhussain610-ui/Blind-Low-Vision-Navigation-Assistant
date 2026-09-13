"""Blind/Low-Vision Navigation Assistant — Streamlit dashboard (Part 10).

Run with:  streamlit run app.py

Complete pipeline:
  Webcam/Video -> YOLO11 Detection -> Relative Depth
  -> Object Tracking -> Spatial Analysis -> Risk Assessment
  -> Navigation Decision -> Virtual Haptic Feedback -> Dashboard & Logging
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import cv2
import streamlit as st

# Make the src package importable regardless of the launch directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config_manager import ConfigManager, ConfigurationError  # noqa: E402
from src.logging.logger import get_logger, setup_logging  # noqa: E402
from src.camera import (  # noqa: E402
    CameraManager,
    FramePacket,
    InputState,
    ManagerConfig,
    VideoInputError,
)
from src.detection import annotate  # noqa: E402
from src.system.system_info import collect_system_info  # noqa: E402
from src.system.system_status import build_initial_status  # noqa: E402
from src.utils.modules import CURRENT_PHASE  # noqa: E402
from src.utils.paths import APP_CONFIG_FILE, ensure_required_directories  # noqa: E402
from src.pipeline import (  # noqa: E402 (Part 10)
    build_pipeline,
    PipelineFrameResult,
    NavigationPipeline,
)
from src.evaluation import EvaluationMetrics  # noqa: E402 (Part 10)
from src.navigation import NavigationCommand, NavigationDecision  # noqa: E402 (Part 8)
from src.risk import RiskLevel  # noqa: E402 (Part 7)
from src.spatial import ZoneState  # noqa: E402 (Part 6)
from src.tracking import MovementStatus  # noqa: E402 (Part 5)
from src.haptic import Motor  # noqa: E402 (Part 9)

APP_TITLE = "Blind/Low-Vision Navigation Assistant"
APP_SUBTITLE = "Software-Only AI Prototype — Complete Pipeline"


def inject_dashboard_styles() -> None:
    """Apply the dashboard's visual language without changing Streamlit logic."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --ink: #e8f0eb;
            --muted: #91a39b;
            --panel: #17211e;
            --panel-soft: #1d2a26;
            --line: rgba(190, 224, 207, 0.14);
            --lime: #c9f36b;
            --lime-dark: #18220e;
            --orange: #ffb86b;
        }

        .stApp {
            background:
                radial-gradient(circle at 88% 4%, rgba(201, 243, 107, 0.09), transparent 24rem),
                radial-gradient(circle at 8% 30%, rgba(93, 160, 137, 0.08), transparent 22rem),
                #0d1412;
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
        }
        .block-container { max-width: 1180px; padding: 2.8rem 2.5rem 4rem; }
        h1, h2, h3, [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
        h1 { letter-spacing: -0.04em; font-size: clamp(2rem, 4vw, 3.8rem) !important; line-height: 1 !important; }
        h2, h3 { letter-spacing: -0.025em; }
        p, label, .stCaption { color: var(--muted); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { visibility: hidden; }
        hr { border-color: var(--line); margin: 1.6rem 0; }

        .hero {
            position: relative; overflow: hidden; padding: 2.1rem 2.2rem 2rem;
            border: 1px solid var(--line); border-radius: 20px;
            background: linear-gradient(135deg, rgba(29, 42, 38, .96), rgba(15, 24, 21, .9));
            box-shadow: 0 24px 70px rgba(0,0,0,.22);
        }
        .hero:after { content: ''; position: absolute; width: 12rem; height: 12rem; right: -3rem; top: -5rem;
            border: 1px solid rgba(201,243,107,.24); border-radius: 50%; box-shadow: 0 0 0 1.5rem rgba(201,243,107,.04), 0 0 0 3rem rgba(201,243,107,.025); }
        .eyebrow { color: var(--lime); font: 700 .73rem 'Space Grotesk', sans-serif; letter-spacing: .14em; text-transform: uppercase; }
        .hero-title { position: relative; z-index: 1; color: var(--ink); font: 700 clamp(2rem, 4vw, 3.7rem)/1 'Space Grotesk', sans-serif; max-width: 730px; margin: .65rem 0 .7rem; }
        .hero-copy { position: relative; z-index: 1; max-width: 660px; color: var(--muted); font-size: 1rem; margin: 0; }
        .live-pill { display: inline-flex; align-items: center; gap: .5rem; margin-top: 1.25rem; padding: .45rem .75rem; border: 1px solid rgba(201,243,107,.28); border-radius: 99px; color: var(--lime); background: rgba(201,243,107,.08); font-size: .78rem; font-weight: 600; }
        .live-dot { width: .45rem; height: .45rem; border-radius: 50%; background: var(--lime); box-shadow: 0 0 0 .22rem rgba(201,243,107,.13); }
        .section-kicker { margin: 1.7rem 0 .55rem; color: var(--lime); font: 700 .72rem 'Space Grotesk', sans-serif; letter-spacing: .14em; text-transform: uppercase; }
        .source-shell { padding: 1.2rem 1.35rem 1.4rem; border: 1px solid var(--line); border-radius: 16px; background: rgba(23,33,30,.7); }
        div[data-testid="stRadio"] > label { color: var(--ink); font: 600 1rem 'Space Grotesk', sans-serif; }
        div[data-testid="stRadio"] div[role="radiogroup"] { gap: .55rem; }
        div[data-testid="stRadio"] div[role="radiogroup"] label { border: 1px solid var(--line); border-radius: 10px; padding: .7rem 1rem; background: rgba(255,255,255,.025); transition: border .2s, background .2s; }
        div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) { border-color: var(--lime); background: rgba(201,243,107,.11); color: var(--lime); }
        .stButton > button { border-radius: 10px; border: 1px solid var(--line); min-height: 2.7rem; font-weight: 600; }
        .stButton > button[kind="primary"] { background: var(--lime); color: var(--lime-dark); border-color: var(--lime); }
        .stButton > button[kind="primary"]:hover { background: #ddff88; border-color: #ddff88; color: var(--lime-dark); }
        [data-testid="stMetric"] { padding: .85rem 1rem; border: 1px solid var(--line); border-radius: 12px; background: rgba(255,255,255,.025); }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: var(--ink); }
        [data-testid="stFileUploader"] { border: 1px dashed rgba(201,243,107,.32); border-radius: 12px; background: rgba(201,243,107,.035); padding: .3rem; }
        .stAlert { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------- #
# Startup (no AI models, no camera — Part 1 only)
# --------------------------------------------------------------------- #
def initialize_application():
    """Run the full foundation startup sequence.

    Returns:
        (config, system_info, status) on success, (None, error_message, None)
        on a recoverable configuration/startup failure.
    """
    config = ConfigManager(APP_CONFIG_FILE)  # 1. Load configuration
    ensure_required_directories()            # 2. Create project directories
    setup_logging(                           # 3. Initialize logging
        level_override=config.get("logging.level")
    )
    enable_gpu = bool(config.get("system.enable_gpu_if_available", True))
    system = collect_system_info(enable_gpu_if_available=enable_gpu)  # 4+5.
    status = build_initial_status(config, system)                     # 6.
    return config, system, status


# --------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------- #
def render_sidebar(config: ConfigManager) -> None:
    """Render the application sidebar with project info."""
    with st.sidebar:
        st.header("Application")

        st.caption("Project")
        st.write(config.get("application.name", APP_TITLE))

        st.caption("Version")
        st.write(config.get("application.version", "1.0.0"))

        st.caption("Environment")
        st.write(config.get("application.environment", "development"))

        st.divider()
        st.caption(
            "No hardware required · Fully local · CPU-compatible"
        )


def build_pipeline_from_config(config: ConfigManager) -> NavigationPipeline:
    """Build the complete Part 10 pipeline from configuration."""
    from src.pipeline import build_pipeline

    return build_pipeline(config)


# --------------------------------------------------------------------- #
# Part 10 — Pipeline controls & results panels
# --------------------------------------------------------------------- #
RESOLUTION_OPTIONS_P10 = {
    "320 × 240": (320, 240),
    "640 × 480": (640, 480),
    "1280 × 720": (1280, 720),
}


def render_pipeline_controls(config: ConfigManager) -> None:
    """Render the Part 10 pipeline toggle and configuration knobs.

    The full pipeline (detection -> depth -> tracking -> spatial ->
    risk -> navigation -> haptic) is activated by a single toggle. When
    off, only YOLO11 detection (Part 3) runs.
    """
    st.header("Navigation Pipeline (Parts 4–10)")
    st.caption(
        "Exposure -> Object Detection -> Relative Depth -> "
        "Object Tracking -> Spatial Analysis -> Risk Assessment -> "
        "Navigation Decision -> Virtual Haptic Feedback"
    )

    run_full_pipeline = st.toggle(
        "▶ Run Full Pipeline",
        value=False,
        key="run_full_pipeline",
        help=(
            "When on, every frame runs detection + depth + tracking + "
            "spatial + risk + navigation + haptic. When off, only YOLO11 "
            "detection (Part 3) runs."
        ),
    )

    if run_full_pipeline:
        col1, col2 = st.columns(2)
        with col1:
            st.number_input(
                "Min command interval (frames)",
                min_value=0,
                max_value=60,
                value=8,
                help="Minimum frames between non-critical command changes.",
            )
        with col2:
            st.number_input(
                "Hysteresis frames",
                min_value=1,
                max_value=20,
                value=3,
                help="How many frames a new command must persist before switching.",
            )
        st.divider()
        st.caption(
            "Pipeline controls — changes apply immediately to the live "
            "pipeline object. Run the pipeline on webcam or video frames "
            "below to see live results."
        )
    else:
        pipeline = st.session_state.pop("pipeline", None)
        st.session_state.pop("evaluation_metrics", None)
        if pipeline is not None:
            if st.session_state.get("detector") is pipeline.detector:
                st.session_state.pop("detector", None)
            try:
                pipeline.detector.release()
            except Exception as exc:
                get_logger("app.pipeline").warning(
                    "Error releasing pipeline detector: %s", exc
                )
        st.caption(
            "Full pipeline is off. Only YOLO11 detection (Part 3) will run. "
            "Toggle 'Run Full Pipeline' to enable Parts 4–10."
        )



# --------------------------------------------------------------------- #
# Part 2 — Input Source UI
# --------------------------------------------------------------------- #
RESOLUTION_OPTIONS = {
    "320 × 240": (320, 240),
    "640 × 480": (640, 480),
    "1280 × 720": (1280, 720),
}


def build_input_config(config: ConfigManager, index: int, resolution: str) -> ManagerConfig:
    """Create a ManagerConfig from app configuration + UI selections."""
    width, height = RESOLUTION_OPTIONS.get(resolution, (640, 480))
    return ManagerConfig(
        webcam_index=index,
        webcam_width=width,
        webcam_height=height,
        webcam_fps=float(config.get("input.webcam.target_fps", 15)),
        webcam_backend=str(config.get("input.webcam.backend", "auto")),
        frame_skip=int(config.get("input.processing.frame_skip", 0)),
    )


def build_detector_from_config(config: ConfigManager):
    """Create a YOLODetector from app configuration values."""
    from src.detection import ClassFilter, YOLODetector

    cf_cfg = config.get("detection.class_filter", {}) or {}
    class_filter = ClassFilter(
        mode=str(cf_cfg.get("mode", "navigation")),
        custom_classes=set(cf_cfg.get("custom_classes", []) or []),
    )
    return YOLODetector(
        model_name=str(config.get("detection.model", "yolo11n.pt")),
        confidence=float(config.get("detection.confidence_threshold", 0.40)),
        iou=float(config.get("detection.iou_threshold", 0.45)),
        image_size=int(config.get("detection.image_size", 640)),
        max_detections=int(config.get("detection.max_detections", 100)),
        device=str(config.get("detection.device", "auto")),
        class_filter=class_filter,
    )


def render_detection_controls(config: ConfigManager, system=None) -> None:
    """Object Detection section: model/device info + enable toggle."""
    from src.detection import NAVIGATION_CLASSES

    st.header("Object Detection")
    device_label = (
        system.preferred_device.upper() if system is not None else "AUTO"
    )
    st.markdown(
        f"**Model:** `{config.get('detection.model', 'yolo11n.pt')}` (YOLO11, COCO-pretrained) · "
        f"**Device:** {device_label}"
    )
    st.caption(
        "Navigation-relevant classes supported by the pretrained model: "
        f"{', '.join(sorted(NAVIGATION_CLASSES)[:12])}… — classes not present in the "
        "model (e.g. stairs) cannot be detected until a custom model is trained."
    )
    enabled = st.toggle("Enable object detection", key="detection_enabled", value=True)
    if enabled and "detector" not in st.session_state:
        try:
            with st.spinner("Loading YOLO11 model (one-time)…"):
                st.session_state.detector = build_detector_from_config(config)
                st.session_state.detector.load()
            st.session_state.detector_missing = sorted(
                st.session_state.detector.unavailable_classes
            )
        except Exception as exc:
            get_logger("app.detection").error("Model load failed: %s", exc)
            st.error(
                "Unable to load YOLO11 model.\n\n"
                "Please verify the model configuration and installation."
            )
            st.session_state.detection_enabled = False
    if not enabled and "detector" in st.session_state:
        # Tear down the loaded model and clear ALL detector-related state.
        # release() can itself fail (e.g. the model is already closed), so it
        # must never prevent cleanup; remove stale derived state too.
        try:
            st.session_state.detector.release()
        except Exception as exc:
            get_logger("app.detection").warning(
                "Error releasing detector: %s", exc
            )
        finally:
            st.session_state.pop("detector", None)
            st.session_state.pop("detector_missing", None)


def run_detection_on_packet(packet: FramePacket, container) -> None:
    """Run detection on a packet and render the annotated frame + panel.

    Both the annotated frame and the detection results panel are rendered
    inside ``container`` so they stay together in the live frame area.
    """
    detector = st.session_state.get("detector")
    if detector is None:
        container.image(packet.to_rgb(), width="stretch")
        return
    try:
        result = detector.detect(packet)
    except Exception as exc:
        get_logger("app.detection").error("Inference failed: %s", exc)
        container.error("Detection failed on this frame; showing raw frame.")
        container.image(packet.to_rgb(), width="stretch")
        return
    annotated = annotate(packet.frame, result)
    container.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
    render_detection_results(result, detector, container)


def run_pipeline_on_packet(packet: FramePacket, container, config: ConfigManager) -> None:
    """Process one live frame through Parts 3-9 and render measured output."""
    session = st.session_state
    pipeline = session.get("pipeline")
    if pipeline is None:
        pipeline = build_pipeline_from_config(config)
        session.pipeline = pipeline
        session.evaluation_metrics = EvaluationMetrics()
    session.detector = pipeline.detector

    try:
        result = pipeline.process_frame(packet.frame)
        session.evaluation_metrics.record(result)
    except Exception as exc:
        get_logger("app.pipeline").error("Pipeline failed: %s", exc, exc_info=True)
        container.error("The full pipeline failed on this frame; showing the raw frame.")
        container.image(packet.to_rgb(), width="stretch")
        return

    if result.skipped:
        container.image(packet.to_rgb(), width="stretch")
        return

    annotated = pipeline.annotated_frame(packet.frame, result)
    container.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
    render_pipeline_result(result, session.evaluation_metrics, container)


def render_pipeline_result(result, metrics: EvaluationMetrics, container) -> None:
    """Render pipeline decision and measured session metrics."""
    decision = result.navigation
    risk = result.risk
    summary = metrics.summary()
    columns = container.columns(4)
    columns[0].metric("Navigation", decision.command.value if decision else "-")
    columns[1].metric("Risk", risk.highest.label if risk else "-")
    columns[2].metric("Pipeline FPS", f"{summary['avg_fps']:.1f}")
    columns[3].metric("Frames", str(summary["frames_processed"]))
    container.caption(
        f"Detections: {summary['total_detections']} · "
        f"Average latency: {summary['avg_pipeline_ms']:.1f} ms · "
        f"Warnings: {summary['warnings']} · Stops: {summary['stops']}"
    )


def render_detection_results(result, detector, container=None) -> None:
    """Structured detection panel: objects, positions, performance.

    Renders into ``container`` (a Streamlit DeltaGenerator) when provided —
    e.g. the live frame placeholder, so the panel stays with the annotated
    frame — and falls back to the main page layout otherwise.
    """
    render = container if container is not None else st
    render.subheader("Detected Objects")
    if result.detections:
        table = render.container()
        header = table.columns([3, 2, 2, 2])
        for col, title in zip(header, ("Object", "Confidence", "Position", "Size (px)")):
            col.markdown(f"**{title}**")
        for d in result.detections:
            row = table.columns([3, 2, 2, 2])
            row[0].write(d.class_name)
            row[1].write(f"{d.confidence:.2f}")
            row[2].write(d.position.value)  # image-space only, NOT navigation
            row[3].write(f"{d.width:.0f}×{d.height:.0f}")
    else:
        render.info("No objects detected in this frame.")

    met = render.columns(4)
    met[0].metric("Inference Time", f"{result.inference_time_ms:.0f} ms")
    met[1].metric("Inference FPS", f"{result.inference_fps:.1f}")
    met[2].metric("Detections", str(result.count))
    met[3].metric("Device", result.device.upper())
    if getattr(detector, "_unavailable_classes", None):
        render.caption(
            "Not detectable with this pretrained model: "
            + ", ".join(detector.unavailable_classes)
            + " (custom training required)."
        )


def render_webcam_ui(config: ConfigManager) -> None:
    """Webcam selection, start/stop, and live frame display."""
    st.markdown('<div class="section-kicker">Live capture</div>', unsafe_allow_html=True)
    st.subheader("Webcam processing")
    col_left, col_right = st.columns(2)
    index = col_left.selectbox("Camera Index", [0, 1, 2], index=0)
    resolution = col_right.selectbox(
        "Requested Resolution", list(RESOLUTION_OPTIONS), index=1
    )

    session = st.session_state
    if "webcam_manager" not in session:
        session.webcam_manager = None
        session.webcam_running = False

    start_col, stop_col = st.columns(2)
    if start_col.button("▶ Start Camera", type="primary", disabled=session.webcam_running):
        manager = None
        try:
            manager = CameraManager(build_input_config(config, index, resolution))
            manager.use_webcam(index=index)
            info = manager.start()
            session.webcam_manager = manager
            session.webcam_running = True
            st.success(f"{info.name} started.")
        except VideoInputError as exc:
            logger = get_logger("app.input")
            logger.error("Webcam start failed: %s", exc)
            st.error(
                "Unable to open the selected webcam.\n\n"
                "Please check the camera connection and selected camera index."
            )
        except Exception as exc:
            get_logger("app.input").error(
                "Unexpected webcam error: %s", exc, exc_info=True
            )
            st.error(f"Unexpected webcam error: {exc}")
        finally:
            # On a failed start the manager is not stored in the session, but
            # it may already have opened the camera device (use_webcam/start).
            # Release it so the webcam is not left locked for other apps, and
            # keep stale session state from pointing at a dead manager.
            if session.webcam_manager is None and manager is not None:
                try:
                    manager.stop()
                except Exception as exc:
                    get_logger("app.input").warning(
                        "Error releasing webcam after failed start: %s", exc
                    )

    if stop_col.button("■ Stop Camera", disabled=not session.webcam_running):
        manager = session.webcam_manager
        if manager is not None:
            manager.stop()
        session.webcam_manager = None
        session.webcam_running = False
        st.info("Camera stopped and released.")

    if not session.webcam_running:
        st.caption("No webcam input is active. Frames are processed locally only.")
        return

    # Live loop: display actual frames until Stop is pressed.
    manager = session.webcam_manager
    if manager is None:
        st.error("The webcam session is unavailable. Please start the camera again.")
        session.webcam_running = False
        return
    frame_ph = st.empty()
    metrics_ph = st.empty()
    loop_stop = st.button("■ Stop Camera (live)")
    if loop_stop:
        manager.stop()
        session.webcam_manager = None
        session.webcam_running = False
        st.rerun()

    packet = manager.read_packet()
    if packet is not None and packet.is_valid:
        if st.session_state.get("run_full_pipeline"):
            run_pipeline_on_packet(packet, frame_ph, config)
        elif st.session_state.get("detection_enabled"):
            run_detection_on_packet(packet, frame_ph)
        else:
            frame_ph.image(packet.to_rgb(), width="stretch")
    elif manager.state is InputState.ERROR:
        st.error(f"Input error: {manager.error_message}")
        manager.stop()
        session.webcam_manager = None
        session.webcam_running = False
    else:
        st.warning("Waiting for frames from the webcam…")


def render_video_ui(config: ConfigManager) -> None:
    """Local video upload, metadata display, playback controls."""
    st.markdown('<div class="section-kicker">Local analysis</div>', unsafe_allow_html=True)
    st.subheader("Video processing")
    supported = config.get("input.video.supported_extensions", [".mp4", ".avi", ".mov", ".mkv"])
    uploaded = st.file_uploader(
        "Upload a video (processed locally — nothing is sent to any server)",
        type=[ext.lstrip(".") for ext in supported],
    )
    session = st.session_state
    if "video_manager" not in session:
        session.video_manager = None
        session.video_playing = False
        session.video_temp_path = None

    if uploaded is None:
        if session.video_manager is not None:
            session.video_manager.stop()
            session.video_manager = None
            session.video_playing = False
        if session.get("video_staged_path"):
            _remove_temp_video(session.video_staged_path)
            session.video_staged_path = None
            session.video_temp_path = None
        st.caption("Select a local video file to view its metadata and play it.")
        return

    temp_path = _stage_uploaded_video(uploaded)
    if temp_path is None:
        return

    if session.video_manager is None:
        manager = CameraManager()
        try:
            manager.use_video_file(temp_path)
            info = manager.start()  # validates readability + collects metadata
            session.video_manager = manager
        except VideoInputError as exc:
            get_logger("app.input").error("Video load failed: %s", exc)
            st.error(
                "Unable to read the selected video.\n\n"
                "Please select a valid supported video file."
            )
            return

    manager = session.video_manager
    info = manager.source_info
    if info is None:
        st.error("Video metadata is unavailable for the selected file.")
        return
    st.markdown("**Video Information**")
    cols = st.columns(5)
    cols[0].metric("Resolution", f"{info.width}×{info.height}")
    cols[1].metric("FPS", f"{info.fps:.2f}" if info.fps else "unknown")
    cols[2].metric("Frames", str(info.frame_count) if info.frame_count else "unknown")
    cols[3].metric(
        "Duration",
        f"{info.duration_seconds:.1f}s" if info.duration_seconds else "unknown",
    )
    cols[4].metric("State", manager.state.value)

    # --- Save the processed/output video (prominent option shown on upload) ---
    with st.expander("💾 Save Result Video", expanded=True):
        st.caption(
            "Run the full pipeline (detection → depth → tracking → spatial → "
            "risk → navigation) over this clip once and save an annotated output "
            "video to `results/`. On CPU this takes a little while — progress is "
            "shown below, then a Download button appears."
        )
        if st.button("⬇ Process & Save Output Video", type="primary"):
            progress = st.progress(0.0, text="Preparing…")

            def _progress(fraction):
                progress.progress(fraction, text=f"Processing… {fraction * 100:.0f}%")

            base = Path(temp_path).stem
            try:
                out_path, stats = export_processed_video(
                    config, temp_path, f"annotated_{base}", progress_cb=_progress
                )
            except Exception as exc:
                get_logger("app.export").exception("Export failed: %s", exc)
                progress.progress(1.0, text="Failed")
                st.error(f"Export failed: {exc}")
                return
            if not out_path:
                return
            progress.progress(1.0, text="Done")
            st.success(f"Saved: `{out_path}`")
            if stats:
                st.write(
                    f"{stats['frames']} frames processed in {stats['elapsed_s']:.1f}s "
                    f"({stats['rate']:.1f} frames/s · {stats['width']}×{stats['height']} "
                    f"@ {stats['out_fps']:.1f}fps)."
                )
            try:
                with open(out_path, "rb") as fh:
                    st.download_button(
                        "⬇ Download Result Video",
                        data=fh.read(),
                        file_name=Path(out_path).name,
                        mime="video/mp4",
                    )
            except OSError as exc:
                st.warning(f"Saved to disk, but download unavailable: {exc}")

    if manager.state is InputState.END_OF_VIDEO:
        st.success("END OF VIDEO", icon="🏁")
        if st.button("↺ Restart Video"):
            manager.restart()
            session.video_playing = True
            st.rerun()
        return

    play_col, pause_col, restart_col, stop_col = st.columns(4)
    if play_col.button("▶ Start", disabled=session.video_playing):
        if manager.state is InputState.PAUSED:
            manager.resume()
        session.video_playing = True
        st.rerun()
    if pause_col.button("⏸ Pause", disabled=not session.video_playing):
        manager.pause()
        session.video_playing = False
        st.rerun()
    if restart_col.button("↺ Restart"):
        manager.restart()
        manager.resume()
        session.video_playing = True
        st.rerun()
    if stop_col.button("■ Stop"):
        manager.stop()
        session.video_manager = None
        session.video_playing = False
        if session.get("video_staged_path"):
            _remove_temp_video(session.video_staged_path)
            session.video_staged_path = None
            session.video_temp_path = None
        st.info("Video stopped and resources released.")
        st.rerun()

    if not session.video_playing:
        st.caption("Playback paused or not started.")
        return

    frame_ph = st.empty()
    packet = manager.read_packet()
    if packet is not None and packet.is_valid:
        if st.session_state.get("run_full_pipeline"):
            run_pipeline_on_packet(packet, frame_ph, config)
        elif st.session_state.get("detection_enabled"):
            run_detection_on_packet(packet, frame_ph)
        else:
            frame_ph.image(packet.to_rgb(), width="stretch")
    elif manager.state is InputState.END_OF_VIDEO:
        session.video_playing = False
        st.rerun()
    else:
        st.warning("Waiting for video frames…")


def _stage_uploaded_video(uploaded_file):
    """Write an uploaded video to a local temp file for OpenCV decoding.

    Returns the temp path, or None when the file exceeds the size limit.
    """
    session = st.session_state
    if session.video_temp_path == uploaded_file.name + f"_{uploaded_file.size}":
        return session.get("video_staged_path")
    import os
    import tempfile

    max_mb = 500
    if uploaded_file.size > max_mb * 1024 * 1024:
        st.error(f"Video is larger than {max_mb} MB. Please choose a smaller file.")
        return None
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(uploaded_file.getvalue())
    handle.close()
    if session.get("video_staged_path"):
        _remove_temp_video(session.video_staged_path)
    session.video_staged_path = handle.name
    session.video_temp_path = uploaded_file.name + f"_{uploaded_file.size}"
    get_logger("app.input").info("Video staged locally: %s", uploaded_file.name)
    return handle.name


def _remove_temp_video(path) -> None:
    import os

    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# --------------------------------------------------------------------- #
# Part 10 — Offline video export (process whole clip -> annotated file)
# --------------------------------------------------------------------- #
EXPORT_MAX_WIDTH = 640


def _make_video_writer(base_path, fps: float, size) -> tuple:
    """Open a cv2 writer preferring MP4/H.264, falling back to AVI/MJPG.

    Returns ``(writer, final_path)``. Raises ``RuntimeError`` if no writer
    could be opened on this OpenCV build.
    """
    for suffix, fourcc in ((".mp4", "mp4v"), (".mp4", "avc1")):
        cand = base_path.with_suffix(suffix)
        writer = cv2.VideoWriter(
            str(cand), getattr(cv2, "VideoWriter_fourcc")(*fourcc), fps, size
        )
        if writer.isOpened():
            return writer, cand
        writer.release()
    avi_path = base_path.with_suffix(".avi")
    writer = cv2.VideoWriter(
        str(avi_path), getattr(cv2, "VideoWriter_fourcc")(*"MJPG"), fps, size
    )
    if writer.isOpened():
        return writer, avi_path
    writer.release()
    raise RuntimeError("No usable OpenCV video writer found (mp4/avi).")


def export_processed_video(config, temp_path, output_base, progress_cb=None):
    """Run the full pipeline over a local video and write an annotated file.

    Args:
        config: ConfigManager used to build the pipeline.
        temp_path: Path to the staged source video.
        output_base: File name (no extension) where the result is written.
        progress_cb: Optional ``callable(fraction: float)`` for a progress bar.

    Returns ``(output_path, stats_dict)`` on success, else ``(None, None)``
    with an error/warning already rendered to the UI.
    """
    pipeline = build_pipeline_from_config(config)
    pipeline.reset()
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        get_logger("app.export").error("Could not open %s for processing", temp_path)
        pipeline.detector.release()
        st.error("Unable to open the selected video for processing.")
        return None, None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    if not (fps > 0) or fps != fps:  # guard 0/negative/NaN
        fps = 15.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)

    width, height = src_w, src_h
    if width > EXPORT_MAX_WIDTH:
        scale = EXPORT_MAX_WIDTH / width
        width, height = EXPORT_MAX_WIDTH, max(1, int(round(height * scale)))

    from src.utils.paths import RESULTS_DIR

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    writer, out_path = _make_video_writer(
        RESULTS_DIR / output_base, fps, (width, height)
    )

    import time

    logger = get_logger("app.export")
    frame_no = 0
    processed = 0
    start = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = _prepare_export_frame(frame, width, height)
            result = pipeline.process_frame(frame)
            if result.skipped:
                annotated = frame
                writer.write(annotated)
            else:
                processed += 1
                annotated = pipeline.annotated_frame(frame, result)
                writer.write(annotated)
            frame_no += 1
            if progress_cb is not None:
                progress_cb(min(frame_no / total, 1.0))
        writer.release()
    finally:
        cap.release()
        try:
            pipeline.detector.release()
        except Exception as exc:
            logger.warning("Error releasing detector after export: %s", exc)
        pipeline.reset()

    elapsed = time.perf_counter() - start
    stats = {
        "frames": frame_no,
        "processed": processed,
        "elapsed_s": elapsed,
        "rate": frame_no / elapsed if elapsed > 0 else 0.0,
        "width": width,
        "height": height,
        "out_fps": fps,
    }
    if frame_no == 0:
        try:
            out_path.unlink()
        except OSError:
            pass
        st.warning("No frames were read from the video; nothing was saved.")
        return None, None
    logger.info(
        "Exported annotated video: %s (%d frames in %.1fs)",
        out_path, frame_no, elapsed,
    )
    return str(out_path), stats


def _prepare_export_frame(frame, width: int, height: int):
    """Resize a frame to the export target size; returns a BGR ndarray."""
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def render_input_section(config: ConfigManager) -> None:
    """Render the only dashboard workflow: webcam or video processing."""
    st.session_state.setdefault("detection_enabled", True)
    st.session_state.setdefault("run_full_pipeline", True)

    mode = st.radio(
        "Choose an input source",
        ["Webcam", "Video File"],
        horizontal=True,
    )
    if mode == "Webcam":
        render_webcam_ui(config)
    else:
        render_video_ui(config)


def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_dashboard_styles()
    st.markdown(
        '<section class="hero">'
        '<div class="eyebrow">Computer vision / local-first</div>'
        f'<div class="hero-title">{APP_TITLE}</div>'
        '<p class="hero-copy">Real-time visual awareness for safer movement. '
        'Choose a camera or upload a clip to begin local processing.</p>'
        '<div class="live-pill"><span class="live-dot"></span> Ready for an input source</div>'
        '</section>',
        unsafe_allow_html=True,
    )

    # Startup with friendly error handling (no ugly tracebacks for
    # normal configuration problems).
    try:
        config, _, _ = initialize_application()
        render_input_section(config)
    except ConfigurationError as exc:
        st.error(
            "Configuration file could not be loaded.\n\n"
            f"{exc}\n\nPlease verify `config/app.yaml` and restart the app.",
            icon="❌",
        )
        st.stop()
    except Exception as exc:  # unexpected: log details, concise UI message
        setup_logging()
        logger = get_logger("app")
        logger.critical("Unexpected startup error:\n%s", traceback.format_exc())
        st.error(
            f"An unexpected error occurred during startup: "
            f"{exc.__class__.__name__}: {exc}",
            icon="❌",
        )
        if st.checkbox("Show technical details", value=False):
            st.code(traceback.format_exc(), language="python")
        st.stop()


if __name__ == "__main__":
    main()
