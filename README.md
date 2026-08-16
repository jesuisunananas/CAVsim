# Multi-Camera V2X Perception Pipeline

A real-time, multi-camera object detection and localization system. It ingests video streams from wide-angle cameras, detects objects using YOLOv8, projects their 2D pixel positions into GPS coordinates using pinhole camera geometry, deduplicates cross-camera detections, and uploads structured records to a V2X (Vehicle-to-Everything) API.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Camera Calibration](#camera-calibration)
- [Running the Pipeline](#running-the-pipeline)

---

## Architecture Overview

The system is composed of three layers that work sequentially: calibration, detection/projection, and aggregation/upload.

```
┌─────────────────────────────────────────────────────────────────┐
│                        CALIBRATION LAYER                        │
│                                                                 │
│  validate.py  ──────►  pitch_yaw_minimize.py                    │
│  (extracts u,v          (scipy Nelder-Mead minimization         │
│   pixel coords          finds optimal pitch & yaw angles        │
│   from reference        by minimizing avg Euclidean error       │
│   images/frames)        across all calibration points)          │
│                                  │                              │
│                                  ▼                              │
│                     calibration_errors.csv                      │
│                     (optimal pitch, yaw per camera)             │
└──────────────────────────────┬──────────────────────────────────┘
                               │ pitch_deg, yaw_deg
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DETECTION LAYER                           │
│                   VideoObjectDetector                           │
│                     (process_video.py)                          │
│                                                                 │
│  Video Frame                                                    │
│      │                                                          │
│      ▼                                                          │
│  YOLOv8 Detection  ──►  Bounding Box  ──►  Bottom-Center (u,v) │
│                                                   │             │
│                         Intrinsic Matrix (K)      │             │
│                         + Distortion Coeffs       │             │
│                                   │               │             │
│                                   ▼               ▼             │
│                         Undistort pixel  ──►  Camera Ray        │
│                                                   │             │
│                         Pitch/Yaw Rotation (Rx·Ry)│             │
│                                                   ▼             │
│                                          World Ray (dx,dy,dz)   │
│                                                   │             │
│                         Ground Plane Intersection (t = H / dy)  │
│                                                   │             │
│                                                   ▼             │
│                                         Local (X, Z) in meters  │
│                                                   │             │
│                         Heading rotation + flat-earth approx    │
│                                                   │             │
│                                                   ▼             │
│                                         GPS (Latitude, Longitude)│
└──────────────────────────────┬──────────────────────────────────┘
                               │ Per-camera detections
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AGGREGATION LAYER                          │
│                    MultiCameraPipeline                          │
│                     (process_video.py)                          │
│                                                                 │
│  Camera 1 detections ──┐                                        │
│  Camera 2 detections ──┼──► Haversine deduplication            │
│  Camera N detections ──┘    (merge if distance < threshold)     │
│                                        │                        │
│                                        ▼                        │
│                             Temporal track assignment           │
│                             (consistent global IDs)             │
│                                        │                        │
│                                        ▼                        │
│                              V2X API Upload / JSON output        │
└─────────────────────────────────────────────────────────────────┘
```

### How the pieces fit together

**`calibration_flow.md`** documents the full mathematical model — the intrinsic matrix $K$, pixel-to-ray projection, rotation matrices, ground plane intersection, and the cost function being minimized. Read this first to understand the geometry.

**`pitch_yaw_minimize.py`** implements the calibration optimizer. You populate its `calibration_points` list with `(u, v)` pixel coordinates paired with their known real-world `(X, Z)` positions (measured on the ground relative to the camera pole). The script runs a Nelder-Mead minimization to find the pitch and yaw angles that minimize average reprojection error across all points, then writes `calibration_errors.csv`.

**`validate.py`** is the data-extraction companion to the optimizer. It spins up a `MultiCameraPipeline` against static images or video frames from a specific camera view and runs detection, letting you visually confirm that detected pixel coordinates correspond to ground-truth positions before feeding them into the minimizer.

**`process_video.py`** is the production runtime. It contains two classes:
- `VideoObjectDetector` — wraps a single camera stream. Runs YOLOv8, applies the calibrated pitch/yaw, and produces GPS-tagged detection records.
- `MultiCameraPipeline` — orchestrates multiple detectors, synchronizes frames, deduplicates overlapping detections using Haversine distance, maintains global track IDs across frames, and handles upload to the V2X API.

---

## Repository Structure

```
.
├── src/co_perception/         # Importable package -- everything below is `from co_perception....`
│   ├── ingest/
│   │   └── kinesis_utils.py       # AWS KVS/HLS URL helpers
│   ├── perception/
│   │   └── tracking_utils.py      # AppearanceExtractor, KalmanTracker
│   └── mapping/
│       └── vis_map.py             # Detection-map HTML generation
├── scripts/
│   └── process_video.py       # Entry point: VideoObjectDetector + MultiCameraPipeline. Run from repo root.
├── models/
│   ├── yolov8n.pt              # Base YOLOv8 weights
│   └── best.pt                 # Fine-tuned weights
├── deploy/
│   ├── cavsim.service          # systemd unit (edit paths for your target machine first)
│   └── setup_service.sh
├── output/                     # Generated at runtime -- detections JSON, map HTML, tracking video
├── calibration/
│   ├── validate.py               # Calibration validation runner
│   └── pitch_yaw_minimize.py     # Camera angle optimizer (scipy Nelder-Mead)
├── training/
│   └── yolo/                   # YOLOv8 training scripts + BDD-derived dataset config (offline, not
│                                # part of the runtime pipeline -- note: data.yaml and
│                                # extract_matching_labels.py currently hardcode absolute paths from
│                                # the original training machine, not this repo -- update before use)
├── experiments/                # Exploratory, not wired into the runtime pipeline (not imported anywhere)
│   ├── Fast-SCNN-pytorch/         # Empty as of this reorg -- semantic segmentation, never populated
│   └── Nerf_py/                   # NeRF/3D reconstruction tooling (sr.py, streetview.py, view_pcd.py)
├── requirements.txt          # Python dependencies
├── docs
│   ├── calibration_flow.md       # Mathematical reference for the calibration model
│   └── video_pipeline.md         # Mathematical reference for the full V2X pipeline
└── camera_views/              # Not included -- your own recorded/live video files go here
    └── ch1/
        └── center/           # Reference images/frames used by validate.py
```

---

## Setup

### 1. Install Conda

If you don't have Conda installed, download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and follow the installer instructions for your OS.

### 2. Create the environment

```bash
conda create -n v2x-pipeline python=3.10 -y
conda activate v2x-pipeline
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** YOLOv8 (`ultralytics`) will automatically download model weights (e.g., `yolov8n.pt`) on first use if they are not already present locally. Ensure you have an internet connection on first run.

### 4. (Optional) GPU support

If you have a CUDA-capable GPU, install the matching PyTorch build before installing the rest of the requirements:

```bash
# Example for CUDA 11.8 — adjust the index URL for your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Camera Calibration

Calibration must be completed before running the production pipeline. The goal is to determine the precise **pitch** and **yaw** angle of each physical camera.

### Step 1 — Collect reference frames

Place representative images or short video clips from a camera view into the appropriate `camera_views/` subdirectory (e.g., `camera_views/ch1/center/`).

### Step 2 — Extract pixel coordinates with `validate.py`

Run `validate.py` to open a live detection window over your reference frames. Note the `(u, v)` pixel coordinates of objects whose real-world `(X, Z)` positions (in meters, relative to the camera pole) you have measured on the ground.

```bash
python -m calibration.validate
```

Update the `VideoObjectDetector` constructor arguments in `validate.py` to match your camera's known parameters:

| Parameter | Description |
|---|---|
| `K` | Intrinsic matrix (focal length, principal point) |
| `camera_height` | Height of camera above ground in meters |
| `pitch_deg` / `yaw_deg` | Initial angle estimates (refined by optimizer) |
| `heading_deg` | True compass heading of the camera |
| `origin_lat` / `origin_lon` | GPS coordinates of the camera pole |

### Step 3 — Run the angle optimizer

Add the `(u, v, true_X, true_Z)` pairs collected in Step 2 to the `calibration_points` list in `pitch_yaw_minimize.py`, then run:

```bash
python calibration/pitch_yaw_minimize.py
```

The script will print the optimal pitch and yaw angles and write a `calibration_errors.csv` showing per-point reprojection errors. Aim for an average error below ~0.5 meters for reliable GPS output.

```
✅ MULTI-POINT CALIBRATION COMPLETE
----------------------------------------
Optimal Pitch: -XX.XX degrees
Optimal Yaw:   -XX.XX degrees
Average Error: X.XX meters per point
----------------------------------------
```

### Step 4 — Update production parameters

Copy the optimal `pitch_deg` and `yaw_deg` values into the corresponding `VideoObjectDetector` constructor call in `scripts/process_video.py`.

---

## Running the Pipeline

Once calibration is complete, run the main pipeline against your live or recorded video streams:

`VideoObjectDetector` and `MultiCameraPipeline` are defined in `scripts/process_video.py` itself (they're the entry point, not part of the `co_perception` package), so the way you configure and run a camera is by editing that file's `if __name__ == "__main__":` block directly, then running it from the repo root:

```bash
python3 scripts/process_video.py
```

That block looks like this (the actual code path, not something you import separately):

```python
import numpy as np

K = np.array([
    [1325.4,      0, 1280.0],
    [     0, 1325.4,  960.0],
    [     0,      0,      1]
], dtype=np.float64)

base_lat = 37.91560117034595
base_lon = -122.33478756387032

cam1 = VideoObjectDetector(
    model_path='models/yolov8n.pt',
    conf=0.3,
    K=K,
    dist_coeffs=None,
    camera_height=7.0,
    pitch_deg=-103.63,   # <-- from calibration
    yaw_deg=-166.80,     # <-- from calibration
    heading_deg=200.0,
    device_id="cam-001-ch1",
    origin_lat=base_lat,
    origin_lon=base_lon,
    city="Richmond",
    state="CA",
    country="USA"
)

pipeline = MultiCameraPipeline(detectors=[cam1])

pipeline.process_streams(
    video_paths=["path/to/stream1.mp4"],
    show_live=True,
    upload=False,          # Set True to push to V2X API
    output_json="output/detections.json",
    output_video="output/tracking.mp4",
    output_image=None,
    output_validate=False
)
```

### `process_streams` parameters

| Parameter | Type | Description |
|---|---|---|
| `video_paths` | `list[str]` | One path per camera stream, in the same order as `detectors` |
| `show_live` | `bool` | Display annotated frames in a live OpenCV window |
| `upload` | `bool` | Upload detection records to the V2X API |
| `output_json` | `str \| None` | Path to write all detections as JSON |
| `output_video` | `str \| None` | Path to write annotated output video |
| `output_image` | `str \| None` | Path to write a single annotated frame |
| `output_validate` | `bool` | Print per-frame detection details for debugging |

### Adding more cameras

Instantiate one `VideoObjectDetector` per camera with its own calibrated parameters, then pass all detectors to `MultiCameraPipeline`. Overlapping detections between cameras are automatically merged using Haversine distance thresholding.

```python
pipeline = MultiCameraPipeline(detectors=[cam1, cam2, cam3])
pipeline.process_streams(video_paths=["ch1.mp4", "ch2.mp4", "ch3.mp4"], ...)
```
