# 🧭 Blind / Low-Vision Navigation Assistant

> A **software-only AI prototype** that investigates real-time camera-based
> environmental awareness and **virtual haptic navigation feedback** for blind and
> low-vision users.

The system ingests a **webcam or video file**, runs state-of-the-art computer
vision (**YOLO11** object detection), estimates **relative depth**, tracks objects
with **ByteTrack**, analyses the **walkable path and zones**, assesses **risk**,
issues a **navigation command** (PROCEED → STOP), and renders the outcome as
**virtual haptic feedback** + a live **Streamlit dashboard** — no physical
hardware required.

| | |
|---|---|
| **Tech stack** | Python 3.10+ · Streamlit · OpenCV · NumPy · PyTorch · Ultralytics YOLO11 · ByteTrack · PyYAML · psutil · pytest |
| **Status** | **Part 10 — Complete Pipeline, Testing & Evaluation** (all 10 parts implemented & integrated) |
| **Hardware required** | **None.** CPU-compatible; GPU (CUDA) auto-used when available |
| **Safety** | ⚠️ Research prototype. **Not** a certified road-crossing / navigation safety aid |

---

## Table of Contents

1. [What this project is](#what-this-project-is)
2. [Safety disclaimer](#-safety-disclaimer)
3. [How it works — the pipeline](#how-it-works--the-pipeline)
4. [Technology stack (start → end)](#technology-stack-start--end)
5. [Feature breakdown by part](#feature-breakdown-by-part)
6. [Project structure](#project-structure)
7. [Installation](#installation)
8. [Usage — running the dashboard](#usage--running-the-dashboard)
9. [Dashboard tour](#dashboard-tour)
10. [Configuration reference](#configuration-reference)
11. [Running tests](#running-tests)
12. [Module reference](#module-reference)
13. [Understanding the algorithms](#understanding-the-algorithms)
14. [Extending the project](#extending-the-project)
15. [Privacy](#privacy)
16. [Known limitations](#known-limitations)
17. [Roadmap / future work](#roadmap--future-work)

---

## What this project is

This is an end-to-end **research prototype** built incrementally (Parts 1 → 10). It
answers the question:

> *"Can a purely software pipeline — using only an ordinary laptop camera —
> give a blind/low-vision user useful, real-time *sense* of what is ahead, where
> safe walking space is, and how urgent a situation is?"*

Every stage is a real, working component:

- **Part 1 — Foundation** — project scaffolding, config, logging, system/device
  detection, paths, structured status registry.
- **Part 2 — Video Input** — webcam + video-file capture, frame packets, frame
  preprocessing (skip, resize), and playback control.
- **Part 3 — Object Detection** — real-time **YOLO11** (`yolo11n.pt`) detection
  with a navigation-aware class filter.
- **Part 4 — Depth** — fast, CPU-friendly **relative depth** proxy (bounding-box
  geometry) with an optional MiDaS monocular backend.
- **Part 5 — Tracking** — **ByteTrack** multi-object tracking + movement
  classification (static / moving / approaching / moving-away / uncertain).
- **Part 6 — Spatial** — LEFT / CENTER / RIGHT zones and a walking-path corridor.
- **Part 7 — Risk** — per-object risk scoring (SAFE → CRITICAL).
- **Part 8 — Navigation** — safety-first command engine with hysteresis
  (PROCEED / MOVE_LEFT / MOVE_RIGHT / SLOW_DOWN / CAUTION / STOP).
- **Part 9 — Haptic** — virtual three-motor haptic belt simulation.
- **Part 10 — Dashboard, Testing & Evaluation** — the full integrated app plus
  measured performance tracking.

---

## ⚠️ Safety disclaimer

**This is a research prototype and must NOT be used as a road-crossing or
navigation safety aid.**

- "Depth" is **relative** (0–1, higher = closer), **never** a metric distance.
- Detection can miss objects, produce false positives, misclassify, and fail in
  poor lighting, occlusion, or unusual viewpoints.
- The haptic "belt" is a **software simulation** for interface research.

---

## How it works — the pipeline

The complete data flow, executed per frame:

```text
Webcam / Video File
   │  (Part 2)
   ▼
YOLO11 Object Detection ──► (Part 3) bounding boxes + classes + confidence
   │
   ▼
Relative Depth Estimation ──► (Part 4) 0..1 proximity per object
   │
   ▼
ByteTrack Multi-Object Tracking ──► (Part 5) stable IDs + movement
   │
   ▼
Spatial & Walkable-Area Analysis ──► (Part 6) zones + path corridor
   │
   ▼
Risk Assessment ──► (Part 7) SAFE..CRITICAL per object
   │
   ▼
Navigation Decision ──► (Part 8) PROCEED..STOP (hysteresis)
   │
   ▼
Virtual Haptic Feedback ──► (Part 9) LEFT / CENTER / RIGHT vibration pattern
   │
   ▼
Streamlit Dashboard + Logging ──► (Part 10) live visuals, metrics, analytics
```

In code, all stages are orchestrated by a single reusable engine,
[`NavigationPipeline.process_frame`](src/pipeline/pipeline.py), which the
dashboard only *renders* — keeping the AI logic UI-independent and fully testable.
---

## Technology stack (start → end)

Every technology used, from foundation to evaluation, why it is used, and where it
enters the pipeline:

| Technology | What it is / why it's used | Enters at |
|---|---|---|
| **Python 3.10+** | Core language; typing, dataclasses, enums for clean interfaces | Part 1 |
| **Streamlit** | Web dashboard UI (live frames, metrics, controls, analytics) | Part 1–10 |
| **PyYAML** | Human-readable configuration (`config/*.yaml`) | Part 1 |
| **NumPy** | Numerical foundation, arrays, distance math (`np.hypot`) | Part 2 |
| **OpenCV (`cv2`)** | Webcam/video I/O, frame preprocessing, annotation, corridor overlay | Part 2–3, 10 |
| **Ultralytics YOLO11** | Real-time object detection model `yolo11n.pt` (COCO-pretrained) | Part 3 |
| **PyTorch** | Inference backend (CPU by default; auto-uses CUDA GPU when present) | Part 3–4 |
| **ByteTrack** | Multi-object tracking + persistent IDs (via Ultralytics `model.track`) | Part 5 |
| **Relative-depth heuristic** | CPU-friendly proximity proxy from box area + vertical position | Part 4 |
| **MiDaS (`torch.hub`)** | Optional real monocular depth model (small) when explicitly enabled | Part 4 *(optional)* |
| **psutil** | Live CPU / RAM measurement for evaluation | Part 1, 10 |
| **python-dotenv** | `.env` overrides for `APP_ENV` / `LOG_LEVEL` | Part 1 |
| **pytest** | Unit / integration / end-to-end test suite | Part 1–10 |

> **GPU?** Never required. If an NVIDIA CUDA device is present and
> `system.enable_gpu_if_available` is `true`, YOLO runs on GPU automatically;
> otherwise everything falls back to CPU gracefully.

---

## Feature breakdown by part

### Part 1 — Project Foundation

- `ConfigManager` loads `config/app.yaml` with dotted-path access
  (`config.get("detection.confidence_threshold")`).
- Centralized logging via `logging.yaml` → `logs/application.log`.
- `collect_system_info()` gathers CPU, RAM, device (CUDA availability),
  and picks a preferred compute device.
- `ensure_required_directories()` creates `data/ models/ results/ logs/`.
- `SystemStatus` registry and a `PipelineModule` list that drives the project
  status report.

### Part 2 — Video Input

- `CameraManager` wraps webcam **and** video-file streams with a unified
  `FramePacket` (BGR frame + frame number + timestamp).
- States: `IDLE / RUNNING / PAUSED / END_OF_VIDEO / ERROR` (`InputState`).
- Webcam resolution options: 320×240, 640×480, 1280×720.
- Video file support: `.mp4 .avi .mov .mkv` (≤ 500 MB upload).

### Part 3 — YOLO11 Object Detection

- `YOLODetector` loads `yolo11n.pt` **once** and reuses it per frame.
- Configurable confidence (`0.40`), IoU (`0.45`), image size (`640`),
  max detections (`100`), device (`auto`).
- `ClassFilter` with modes `all | navigation | custom`; `navigation` keeps
  mobility-relevant COCO classes (person, bicycle, car, motorcycle, bus, truck,
  traffic light, stop sign, …).
- Produces project-native `Detection` / `DetectionResult` objects (never
  Ultralytics objects) — the rest of the app is decoupled from the framework.
- Honest behavior: classes the model can't see (e.g. **stairs**) are simply not
  detected — no fake detections ever produced.

### Part 4 — Relative Depth Estimation

Two backends in `RelativeDepthEstimator`:

- **`heuristic` (default)** — CPU-friendly proxy: larger box area + lower box
  bottom (ground-plane cue) ⇒ higher proximity. Output is `relative_depth ∈
  [0, 1]` (higher = closer) with a label `NEAR / MIDDLE / FAR`.
- **`midas` (optional)** — real monocular depth via MiDaS small from `torch.hub`
  (needs network on first load; falls back to heuristic if unavailable).

### Part 5 — Object Tracking & Movement

- Uses Ultralytics' **ByteTrack** (`bytetrack.yaml`) through the same loaded
  model (`model.track(persist=True)`) — no double model load.
- Per-track history (default 12 frames) classifies movement:
  `STATIC / MOVING / APPROACHING / MOVING_AWAY / UNCERTAIN`.
- Movement combines **pixel displacement** (static ≤ 6 px) with **relative-depth
  trend** (an object that grows steadily is APPROACHING).

### Part 6 — Spatial & Walkable-Area Analysis

- `SpatialAnalyzer` divides the frame into **LEFT / CENTER / RIGHT** zones and a
  central **walking-path corridor** (default 40% of frame width).
- Each zone is `CLEAR / PARTIALLY_BLOCKED / BLOCKED`.
- Every object reports `in_walking_path` + `overlap_ratio`.

### Part 7 — Risk Assessment

- `RiskAssessor.assess()` scores each object `SAFE / LOW / MEDIUM / HIGH /
  CRITICAL` from: **relative depth** (dominant), **movement**, **path overlap**,
  **hazard class**, and **low confidence**.
- Conservative by design: uncertainty never silently lowers risk.

### Part 8 — Navigation Decisions

- `NavigationEngine` turns spatial + risk into one command:
  `PROCEED / MOVE_LEFT / MOVE_RIGHT / SLOW_DOWN / CAUTION / STOP`.
- **Safety-first:** `STOP` beats every other command (critical object, or an
  uncertain obstacle inside the path).
- **Hysteresis** debounces non-critical changes (min interval + persistence
  frames) to avoid commands flipping frame-to-frame.

### Part 9 — Virtual Haptic Feedback

- `VirtualHapticBelt` = a three-motor simulation `LEFT / CENTER / RIGHT`.
- Each command maps to a vibration pattern (intensity, pulsing, frequency):

| Command | Motor(s) | Pattern |
|---|---|---|
| PROCEED | none | silent — path clear |
| MOVE_LEFT | LEFT | left pulse |
| MOVE_RIGHT | RIGHT | right pulse |
| CAUTION | all | short repeated pulses |
| SLOW_DOWN | all | medium pulses |
| STOP | all | strong rapid pulses (highest urgency) |

### Part 10 — Dashboard, Testing & Evaluation

- Full app (`app.py`) with live annotated frames + metrics.
- `EvaluationMetrics` tracks **measured** (never fabricated) FPS, latency,
  detections, warnings, stops, and CPU/RAM.
- Complete pytest suite (Part 1 → 10).
---

## Project structure

```text
blind_navigation_ai/
├── app.py                      # Streamlit entry point (Parts 1–10 dashboard)
├── conftest.py                 # shared pytest fixtures
├── requirements.txt            # pinned Python dependencies
├── .env.example                # optional APP_ENV / LOG_LEVEL overrides
├── config/
│   ├── app.yaml                # all pipeline configuration (see below)
│   └── logging.yaml            # logging format, file & console sinks
├── src/
│   ├── camera/                 # Part 2 — CameraManager, sources, FramePacket, FrameProcessor, ManagerConfig
│   ├── config/                 # Part 1 — ConfigManager, ConfigurationError
│   ├── detection/              # Part 3 — YOLODetector, Detection(Result), ClassFilter, annotate
│   ├── depth/                  # Part 4 — RelativeDepthEstimator, heuristic + optional MiDaS
│   ├── tracking/               # Part 5 — ObjectTracker, TrackedObject, MovementStatus
│   ├── spatial/                # Part 6 — SpatialAnalyzer, SpatialResult, ZoneState
│   ├── risk/                   # Part 7 — RiskAssessor, RiskLevel, ObjectRisk
│   ├── navigation/             # Part 8 — NavigationEngine, NavigationCommand, NavigationDecision
│   ├── haptic/                 # Part 9 — VirtualHapticBelt, Motor, build_pattern
│   ├── pipeline/               # Part 10 — NavigationPipeline, PipelineFrameResult, build_pipeline, factory
│   ├── evaluation/             # Part 10 — EvaluationMetrics, SystemUsage
│   ├── system/                 # system_info, system_status, device selection
│   ├── logging/                # logger setup
│   └── utils/                  # paths, modules registry
├── tests/                      # pytest suite, one file per part/responsibility
├── data/                       # raw / processed / sample assets
├── models/                     # yolo11n.pt (+ optional depth checkpoints)
├── results/                    # detections / depth / tracking / navigation / evaluation output
├── logs/                       # application.log
└── notebooks/                  # exploratory work
```

---

## Installation

Requires **Python 3.10+**.

1. **Create and activate a virtual environment** (from the project root):

   ```bash
   python -m venv .venv
   ```

   - Windows (cmd):  `.venv\Scripts\activate`
   - Windows (PowerShell):  `.venv\Scripts\Activate.ps1`
   - Linux / macOS:  `source .venv/bin/activate`

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   > On first object-detection use, Ultralytics downloads `yolo11n.pt` (~5 MB)
   > and warms up the model. The optional MiDaS backend downloads once via
   > `torch.hub` (requires an internet connection at that moment).

3. **Launch the dashboard:**

   ```bash
   streamlit run app.py
   ```

   The app opens in your browser (default `http://localhost:8501`).

### Streamlit Cloud deployment

The project uses `opencv-python-headless` because Streamlit Cloud does not
provide desktop OpenGL libraries such as `libGL.so.1`. Streamlit's file watcher
is also disabled in `.streamlit/config.toml` to avoid Linux inotify limit errors.

---

## Usage — running the dashboard

1. **System section** — shows your CPU, RAM, preferred device, and a
   "Project Status" list confirming all 10 parts are ready.

2. **Input Source** — choose **Webcam** (pick index + resolution) or
   **Video File** (upload `.mp4/.avi/.mov/.mkv`). Use ▶ / ⏸ / ↺ / ■ controls.

3. **Pipeline controls** — enable **Object Detection** (loads YOLO once) and
   toggle **Run Full Pipeline** to activate Parts 4–10 (depth, tracking, spatial,
   risk, navigation, haptic).

4. **Live output** — the annotated frame shows detection boxes, the **walking-path
   corridor** (green = clear / red = blocked), zone states, risk, the active
   **navigation command**, and the virtual haptic belt state.

5. **Evaluation** — live measured FPS, per-stage latency, warnings, stops, and
   CPU/RAM usage (never fabricated).

---

## Dashboard tour

| Section | What it shows |
|---|---|
| **System** | Device info, compute device, environment |
| **Input Source** | Webcam or video, resolution, playback state, measured input FPS |
| **Object Detection** | Model + device, enable toggle, detected-object table, inference ms/FPS |
| **Full Pipeline** | Enable Parts 4–10 and view aggregated outputs |
| **Project Status** | Per-part readiness checklist |
| **Evaluation** | Session analytics: frames, detections, warnings, stops, FPS, CPU/RAM |
---

## Configuration reference

### `config/app.yaml` (key sections)

```yaml
system:
  cpu_mode: true                    # run on CPU
  enable_gpu_if_available: true     # use CUDA if present

input:
  webcam:
    default_index: 0                # camera index (0 not assumed to exist)
    width: 640                      # capture resolution
    height: 480
    target_fps: 15
    backend: "auto"                 # auto | dshow | msmf | any
  video:
    supported_extensions: [".mp4", ".avi", ".mov", ".mkv"]
    max_upload_mb: 500
  processing:
    frame_skip: 0                   # 0=every frame, 1=every 2nd, ...
    max_width: 640

detection:
  model: "yolo11n.pt"
  confidence_threshold: 0.40
  iou_threshold: 0.45
  image_size: 640
  max_detections: 100
  device: "auto"                    # auto | cpu | cuda
  warmup: true
  class_filter:
    mode: "navigation"              # all | navigation | custom
    custom_classes: []

depth:
  mode: "heuristic"                 # heuristic | midas
  near_threshold: 0.62
  far_threshold: 0.35

tracking:
  tracker: "bytetrack.yaml"
  history_length: 12
  static_threshold_px: 6.0
  approaching_depth_delta: 0.03
  min_history: 3

spatial:
  path_width_ratio: 0.40            # walking corridor (fraction of frame width)
  path_min_depth: 0.30

risk:
  critical_depth: 0.45
  high_depth: 0.35
  medium_depth: 0.22
  approaching_boost: 1
  low_confidence: 0.55
  hazard_classes: ["person", "car", "truck", "bus", "motorcycle", "bicycle", "dog"]

navigation:
  min_command_interval: 8           # frames between non-critical changes
  hysteresis_frames: 3              # frames a new command must persist
  uncertainty_stop: true            # STOP on uncertain obstacle in path

haptic:
  enabled: true
  pulse_interval_s: 0.4
```

### `config/logging.yaml`

Configures `logs/application.log` plus console output using the format
`%(asctime)s | %(levelname)s | %(name)s | %(message)s`.

### `.env`

Copy `.env.example` to `.env` to override (optional):

```bash
APP_ENV=development
LOG_LEVEL=INFO
```

---

## Running tests

```bash
# From the project root, inside the virtual environment:
pytest tests/ -v

# Quick summary:
pytest tests/ -q
```

The suite covers Parts 1–10: configuration, system info, video/camera input,
detection, depth, tracking, spatial, risk, navigation, haptic, the end-to-end
pipeline, and evaluation. Unit/integration tests run on CPU with no camera/model
required.

---

## Module reference (public API)

| Package | Key classes / enums |
|---|---|
| `src.camera` | `CameraManager`, `FramePacket`, `InputState`, `ManagerConfig`, `VideoInputError` |
| `src.config` | `ConfigManager`, `ConfigurationError` |
| `src.detection` | `YOLODetector`, `Detection`, `DetectionResult`, `ClassFilter`, `annotate`, `NAVIGATION_CLASSES`, `ModelLoadError`, `InferenceError` |
| `src.depth` | `RelativeDepthEstimator`, `ObjectDepth`, `DepthError` |
| `src.tracking` | `ObjectTracker`, `TrackedObject`, `TrackingResult`, `MovementStatus` |
| `src.spatial` | `SpatialAnalyzer`, `SpatialResult`, `SpatialObject`, `ZoneState` |
| `src.risk` | `RiskAssessor`, `RiskAssessment`, `ObjectRisk`, `RiskLevel` |
| `src.navigation` | `NavigationEngine`, `NavigationDecision`, `NavigationCommand` |
| `src.haptic` | `VirtualHapticBelt`, `Motor`, `MotorState`, `HapticPattern`, `build_pattern` |
| `src.pipeline` | `NavigationPipeline`, `PipelineFrameResult`, `build_pipeline`, `build_detector` |
| `src.evaluation` | `EvaluationMetrics`, `SystemUsage` |
| `src.system` | `SystemStatus`, `build_initial_status`, `collect_system_info`, `get_device` |
| `src.utils` | paths, `PIPELINE_MODULES`, `CURRENT_PHASE` |
---

## Understanding the algorithms

**Relative depth (heuristic).** `proximity = clamp(0.65·area + 0.35·bottom)`,
where `area` is the box area as a fraction of the frame and `bottom` is how low
the box sits (ground-plane assumption). Purely a monotonic, CPU-cheap proxy —
**not** metric distance.

**Risk score.** For each object,
`score = 0.05 + depth_contribution + movement_contribution + 0.25·path_overlap
+ 0.12·hazard_class (+ low-confidence bump)`, then clamped to [0, 1] and mapped:

| Score | Level |
|---|---|
| ≥ 0.85 | CRITICAL |
| ≥ 0.65 | HIGH |
| ≥ 0.40 | MEDIUM |
| ≥ 0.15 | LOW |
| else | SAFE |

**Navigation decision tree.** `STOP` if any CRITICAL object or an uncertain
obstacle is in the path → otherwise prefer the clear side
(`MOVE_LEFT`/`MOVE_RIGHT`) → else grade by highest risk
(`STOP`/`SLOW_DOWN`/`CAUTION`), falling back to `PROCEED` when the path is empty.

**Hysteresis.** A candidate non-critical command must persist for
`hysteresis_frames` and the last change must be ≥ `min_command_interval` frames in
the past before it is applied — preventing command chatter.

---

## Extending the project

- **Add a new pipeline stage** — implement it as a class, wire it in
  `src/pipeline/factory.py::build_pipeline` and
  `NavigationPipeline.process_frame`, then add a test under `tests/`.
- **Real haptic hardware** — replace `VirtualHapticBelt` with a driver exposing
  the same `apply_command` / `motors` interface (e.g. an ESP32 + vibration motors
  on a belt). The rest of the app is unchanged.
- **Real depth** — set `depth.mode: "midas"` for monocular depth, or plug a
  stereo/custom estimator behind `RelativeDepthEstimator`.
- **Custom detector classes** — train/finetune a YOLO model (e.g. to add
  *stairs*, *curb*) and point `detection.model` at it.

---

## Privacy

All webcam and video processing is **fully local**. Frames are never uploaded or
recorded. Uploaded videos are staged only in the OS temp folder and deleted when
playback stops. No API keys or cloud services are used.

---

## Known limitations

- YOLO detection is image-space only; anything the model was not trained on
  cannot be detected (no fake detections).
- Depth is relative, never metric; performance is heuristic and hardware-bound.
- LEFT/CENTER/RIGHT describe image position — navigation *commands* are produced
  only by Part 8 (when the full pipeline runs).
- CPU performance depends on hardware; the dashboard always shows *measured*
  values, never promised ones.
- Object tracking and risk scoring can be confused by occlusion and rapid motion.
- **This prototype does not guarantee safe navigation.**

---

## Roadmap / future work

- [x] Part 1 — Project Foundation
- [x] Part 2 — Video Input
- [x] Part 3 — YOLO11 Object Detection
- [x] Part 4 — Relative Depth Estimation
- [x] Part 5 — Object Tracking & Movement
- [x] Part 6 — Spatial & Walkable-Area Analysis
- [x] Part 7 — Risk Assessment Engine
- [x] Part 8 — Navigation Decision Engine
- [x] Part 9 — Virtual Haptic Feedback
- [x] Part 10 — Complete Dashboard, Testing & Evaluation

**Beyond Part 10:** custom-trained detection (stairs/curbs), metric depth
(stereo/ToF), physical haptic belt driver, audio/speech alerts, trajectory
prediction, and on-device optimization (TensorRT / ONNX).

---

*Blind/Low-Vision Navigation Assistant · research prototype · not a certified
safety aid.*