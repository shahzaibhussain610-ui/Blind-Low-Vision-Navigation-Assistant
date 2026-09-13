# 🧭 Blind/Low-Vision Navigation Assistant — Step-by-Step Execution Guide

This document walks through **exactly how the application runs**, from the moment
you type the launch command to the per-frame AI pipeline. Every step names the
real file and function that executes, so you can follow along in the code.

---

## 0. Before you begin — prerequisites

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
# Windows (cmd):        .venv\Scripts\activate
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Linux / macOS:        source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

> On the **first** run where you enable object detection, Ultralytics downloads
> `yolo11n.pt` (~5 MB) automatically. It is not bundled.

---

## 1. Launch

```bash
streamlit run app.py
```

Streamlit starts a local server and opens your browser at `http://localhost:8501`.
It executes the module `app.py` top-to-bottom, ending at line 711:

```python
if __name__ == "__main__":
    main()          # app.py:711
```

From here, `main()` (line 659) drives everything.

---

## 2. The three execution layers

| Layer | Where | What happens |
|---|---|---|
| **A. Startup (runs once)** | `main()` → `initialize_application()` | Load config, create dirs, set up logging, detect system, build status |
| **B. UI render** | `main()` → `render_*` functions | Draw sidebar, system status, input controls, pipeline toggles |
| **C. Live per-frame loop** | webcam / video handlers | Read frames → detect → (optional full pipeline) → display |

`main()` runs layers A and B, then **reacts to your button presses** to enter
layer C.

---

## 3. Layer A — Startup, step by step

`main()` calls `initialize_application()` (line 54). Inside it, six steps run in
order:

### Step A1 · Load configuration
```python
config = ConfigManager(APP_CONFIG_FILE)        # src/utils/paths.py → config/app.yaml
```
- `ConfigManager.__init__` (src/config/config_manager.py:37) calls `load()`.
- `yaml.safe_load` reads `config/app.yaml`.
- `_validate()` (line 82) checks the **required sections** exist:
  `application, system, performance, logging, paths`.
- ❌ If the file is missing / invalid / missing a section, it raises
  `ConfigurationError`, which `main()` catches (line 690) and shows a friendly
  error + `st.stop()`.

### Step A2 · Create required directories
```python
ensure_required_directories()   # src/utils/paths.py:53
```
Creates (if missing): `data/raw|processed|sample`, `models/detection|depth`,
`results/detections|depth|tracking|navigation|evaluation`, and `logs/`.

### Step A3 · Set up logging
```python
setup_logging(level_override=config.get("logging.level"))   # src/logging/logger.py
```
Writes to `logs/application.log` + console using the `config/logging.yaml` format
`%(asctime)s | %(levelname)s | %(name)s | %(message)s`.

### Step A4 · Detect compute device
```python
enable_gpu = bool(config.get("system.enable_gpu_if_available", True))
system = collect_system_info(enable_gpu_if_available=enable_gpu)  # src/system/system_info.py:157
```
- Collects Python/OS version, CPU (psutil), RAM (psutil), GPU (lazy `torch`).
- Chooses `preferred_device = "cuda"` only if CUDA works, else `"cpu"`.

### Step A5 · Build initial status
```python
status = build_initial_status(config, system)   # src/system/system_status.py
```

### Result
`initialize_application()` returns `(config, system, status)` to `main()`.

---

## 4. Layer B — Render the screens (startup only)

`main()` then draws each section in order:

| # | Call | Shows |
|---|---|---|
| B1 | `render_sidebar(config)` | App name/version/env, **Current Phase**, future-pipeline roadmap |
| B2 | `render_system_section(system, status, config)` | Python/OS/CPU/RAM/GPU + Compute Device metrics |
| B3 | `render_input_section(config)` | Radio: **Webcam** or **Video File** |
| B4 | `render_pipeline_controls(config)` | **▶ Run Full Pipeline** toggle (+ knobs when on) |
| B5 | `render_pipeline_section()` | Implemented-vs-future module checklist |
| B6 | `render_project_status(config, status)` | Per-part "Ready" status list |

💡 **No AI model and no camera are started here** — object detection only loads when you enable it, and the camera only opens when you press Start.
---

## 5. Layer C1 — Choose an input source

`render_input_section` (line 623) shows a horizontal radio:

- **Webcam** → `render_webcam_ui(config)` (line 405)
- **Video File** → `render_video_ui(config)` (line 480)

---

## 6. Running the **Webcam**, step by step

1. Select **Camera Index** `[0,1,2]` and **Requested Resolution** (320×240 / 640×480 / 1280×720).
2. Press **▶ Start Camera** (line 420):
   ```python
   manager = CameraManager(build_input_config(config, index, resolution))  # app.py:268
   manager.use_webcam(index=index)       # src/camera/camera_manager.py
   manager.start()                       # opens cv2.VideoCapture
   session.webcam_manager = manager
   session.webcam_running = True
   ```
   - `build_input_config` (line 268) maps your UI choices into a `ManagerConfig`
     (index, width, height, fps, backend, frame_skip).
   - ❌ If the camera can't open, `VideoInputError` is raised → friendly error,
     camera stays off.
3. **Live loop** — the page re-reders and executes once per user interaction:
   ```python
   packet = manager.read_packet()                 # camera_manager.py:219
   if packet is not None and packet.is_valid:
       if st.session_state.get("detection_enabled"):
           run_detection_on_packet(packet, frame_ph)   # Part 3 (or Parts 4–10)
       else:
           frame_ph.image(packet.to_rgb(), ...)        # raw frame only
       render_frame_metrics(packet, manager, metrics_ph)
   ```
   - `read_packet()` grabs one raw BGR frame, validates it, ticks `FPSMeter`,
     and wraps it in a `FramePacket` (frame + number + timestamp).
4. Holding **■ Stop Camera (live)** → `manager.stop()` (releases OpenCV) and
   `st.rerun()`.

---

## 7. Running a **Video File**, step by step

1. Upload a `.mp4/.avi/.mov/.mkv` file (≤ 500 MB).
2. `_stage_uploaded_video(uploaded)` copies it to the **OS temp folder**.
3. **Video Information** appears (resolution / FPS / frames / duration).
4. Playback controls: ▶ Start, ⏸ Pause, ↺ Restart, ■ Stop (lines 541–564).
5. While playing, the same live loop runs:
   ```python
   packet = manager.read_packet()
   # detection_enabled ? run_detection_on_packet : show image
   ```
6. At end of video, state becomes `InputState.END_OF_VIDEO` → restart option.
7. On **Stop**, the temp file is deleted (`_remove_temp_video`, line 610).

> 🔒 **Privacy:** videos are processed fully locally and the temp copy is removed
> when playback stops.

---

## 8. Object detection for one frame (Part 3)

`run_detection_on_packet(packet, frame_ph)` (line 357):

```python
result = detector.detect(packet)      # YOLODetector.detect → detect_frame
annotated = annotate(packet.frame, result)   # src/detection/visualization.py
container.image(...)                          # show annotated frame
render_detection_results(result, detector)    # table + metrics
```

Inside `YOLODetector.detect_frame` (src/detection/detector.py):

| Step | What runs |
|---|---|
| 1 | Validate the frame (`FramePacket.validate`) |
| 2 | If not loaded: `load()` → resolve device, `YOLO("yolo11n.pt")`, warmup (`warmup: true`) |
| 3 | `_run_model(frame)` → `model.predict(...)` (times it) |
| 4 | `_extract_detections(results, w, h, ...)` |
| 5 | For each box: apply `ClassFilter.allowed_ids()` (mode `navigation` keeps mobility classes) |
| 6 | Build a project `Detection`, `clamp_to_frame(...)`, validate geometry |
| 7 | Return a `DetectionResult` (model, device, inference ms, FPS) |

**On first enable:** the toggle handler (line 335) builds the detector via
`build_detector_from_config` and calls `.load()`, storing it in
`st.session_state.detector`. Toggling off calls `.release()` and deletes it.
---

## 9. Layer C2 — The Full Pipeline, frame by frame (Parts 4–10)

When **▶ Run Full Pipeline** is on and a frame arrives, the frame is routed
through the complete `NavigationPipeline`. The pipeline is built from config by
`build_pipeline_from_config(config)` → `src/pipeline/factory.py::build_pipeline`,
which constructs every stage:

```
YOLODetector → RelativeDepthEstimator → ObjectTracker → SpatialAnalyzer
            → RiskAssessor → NavigationEngine → VirtualHapticBelt
```

`NavigationPipeline.process_frame(frame)` (src/pipeline/pipeline.py:64) runs one
frame through **8 ordered steps**:

| # | Stage | Code (source) | Produces |
|---|---|---|---|
| 1 | Frame-skip policy | `if self.frame_skip > 0 and number % (skip+1) != 0` | skip / continue |
| 2 | **Track (YOLO + ByteTrack)** | `tracking = tracker.update(frame, number, ts)` — `src/tracking/tracker.py` | `TrackingResult` (IDs + detections) |
| 3 | **Relative depth** | for each tracked object: `depth.estimate_one(detection)` — `src/depth/depth_estimator.py:85` | `relative_depth` (0–1) + `depth_label` (NEAR/MIDDLE/FAR) |
| 4 | **Re-classify movement** | `tracker.reclassify(tracking)` — depth-aware movement | STATIC/MOVING/APPROACHING/MOVING_AWAY/UNCERTAIN |
| 5 | **Spatial analysis** | `spatial.analyze(w, h, tracked, number)` — `src/spatial/spatial_analyzer.py:108` | zones (LEFT/CENTER/RIGHT) + path corridor + overlap |
| 6 | **Risk assessment** | `risk.assess(spatial, frame_number)` — `src/risk/risk_assessor.py:113` | per-object SAFE→CRITICAL + score + reasons |
| 7 | **Navigation decision** | `navigation.decide(spatial, risk, number)` — `src/navigation/navigation_engine.py:90` | PROCEED/MOVE_LEFT/MOVE_RIGHT/SLOW_DOWN/CAUTION/STOP |
| 8 | **Virtual haptic** | `haptic.apply_decision(decision)` — `src/haptic/haptic_belt.py:122` | LEFT/CENTER/RIGHT motor pattern |

Finally `process_frame` records `processing_time_ms` and returns a
`PipelineFrameResult` holding every stage's output.

To draw the annotated frame with the walking-path overlay:

```python
annotated = pipeline.annotated_frame(frame, result)
# = annotate(frame, result.detection)  +  _draw_path(...)   # pipeline.py:106
```
`_draw_path` overlays the corridor as **green (path clear = CENTER zone CLEAR)**
or **red (blocked)**.

---

## 10. Inside one pipeline stage — what actually computes

- **Depth (`estimate_one` → `_estimate_heuristic`)**:
  `proximity = clamp(0.65·norm_area + 0.35·bottom_ratio)` where `norm_area` is the
  box area ÷ frame area (×4) and `bottom_ratio` is the box's bottom ÷ frame height.
- **Spatial** (`analyze`): computes a central corridor
  `width·path_width_ratio`; an object is `in_walking_path` when it overlaps the
  corridor *and* `relative_depth ≥ path_min_depth`. Zones become `PARTIALLY_BLOCKED`
  / `BLOCKED` from near/approaching objects.
- **Risk** (`assess` → `_assess_object`): builds a score from depth (dominant) +
  movement + `0.25·overlap` + `0.12·hazard_class`, clamps to `[0,1]`, maps to a
  level via thresholds → `CRITICAL ≥ 0.85 · HIGH ≥ 0.65 · MEDIUM ≥ 0.40 · LOW ≥ 0.15`.
- **Navigation** (`decide`): safety-first — `STOP` wins (critical risk, or an
  uncertain obstacle in the path); else prefer the clear side; else grade by risk;
  `PROCEED` when the path is empty. Non-critical changes are **debounced** by
  `min_command_interval` + `hysteresis_frames`.
- **Haptic** (`apply_decision`): maps the command to a vibration pattern
  (intensity, pulsing, pulse_hz) across the 3 virtual motors.
---

## 11. Evaluation & metrics (Part 10)

The dashboard keeps an `EvaluationMetrics` object (src/evaluation/evaluation.py).
For every pipeline result it records:
- frames processed/skipped, total detections
- command counts, warnings (CAUTION/SLOW_DOWN/direction), **stops**
- rolling average pipeline & detection latency, **measured FPS**
- CPU/RAM via `EvaluationMetrics.system_usage()` (psutil)

All values are **measured**, never fabricated.

---

## 12. Error handling & teardown

### Startup errors (handled in `main()`)
- `ConfigurationError` → friendly message + `st.stop()` (line 690).
- Any other exception → logged with traceback, concise UI message (line 697).

### Runtime (input/detection)
- Webcam/video open failure → `VideoInputError` → friendly error, resources released.
- Inference failure on a frame → error logged, raw frame shown (line 365).
- **Invalid/empty frames are dropped**, never processed.

### Resource release
- **Camera/video:** `manager.stop()` → `_release_source()` closes `cv2.VideoCapture`.
- **Detector:** toggle off → `detector.release()` (drops the model reference).
- **Temp video:** deleted on stop (`_remove_temp_video`).
- **Pipeline:** `pipeline.reset()` resets tracker history, navigation state, haptic.

---

## 13. Quick call-graph (summary)

```text
streamlit run app.py
  └─ main()
       ├─ initialize_application()
       │    ├─ ConfigManager(config/app.yaml)     → validate sections
       │    ├─ ensure_required_directories()
       │    ├─ setup_logging()
       │    ├─ collect_system_info()              → device = cuda | cpu
       │    └─ build_initial_status()
       ├─ render_sidebar / render_system_section
       ├─ render_input_section  → Webcam | Video
       ├─ render_pipeline_controls
       └─ render_pipeline_section / render_project_status

On a live frame (detection):      run_detection_on_packet
                                     └─ detector.detect(packet)      (YOLO11)
                                          └─ annotate(frame, result)

On a live frame (full pipeline):  NavigationPipeline.process_frame(frame)
                                     ├─ tracker.update(...)          (ByteTrack)
                                     ├─ depth.estimate_one(...)      (relative depth)
                                     ├─ tracker.reclassify(...)      (movement)
                                     ├─ spatial.analyze(...)         (zones/path)
                                     ├─ risk.assess(...)             (SAFE..CRITICAL)
                                     ├─ navigation.decide(...)       (PROCEED..STOP)
                                     └─ haptic.apply_decision(...)   (LEFT/CENTER/RIGHT)
```

*End of step-by-step execution guide.*