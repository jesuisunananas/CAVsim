# Multi-Camera V2X Perception Pipeline

A real-time, multi-camera object detection and localization system. It ingests video streams from wide-angle cameras, detects objects using YOLOv8, projects their 2D pixel positions into GPS coordinates using pinhole camera geometry, deduplicates cross-camera detections, and uploads structured records to a V2X (Vehicle-to-Everything) API.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Camera Calibration](#camera-calibration)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Live Viewer](#live-viewer)

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
├── config/
│   └── pipeline.yaml           # Per-machine config: ingestion source, detection/camera params,
│                                # output destinations -- see Configuration below
├── src/co_perception/         # Importable package -- everything below is `from co_perception....`
│   ├── config.py                  # Loads/validates config/pipeline.yaml
│   ├── common/
│   │   ├── broadcast.py           # Local Unix-socket pub/sub (verbatim copy of camera/stream's --
│   │   │                          # pure stdlib, no GStreamer dependency, safe to duplicate)
│   │   └── protocol.py            # Wire format for this project's own output broadcast
│   ├── ingest/
│   │   ├── kinesis_utils.py       # AWS KVS/HLS URL helpers
│   │   ├── decode_protocol.py     # Read-only reimplementation of camera/stream's decode-broadcast framing
│   │   └── frame_sources.py       # Uniform local_socket / local_file / aws_kvs frame source
│   ├── perception/
│   │   └── tracking_utils.py      # AppearanceExtractor, KalmanTracker
│   ├── mapping/
│   │   └── vis_map.py             # Detection-map HTML generation
│   └── output/
│       └── broadcast_sink.py      # JPEG-encodes annotated frames onto a local broadcast socket
├── scripts/
│   ├── process_video.py       # Entry point: VideoObjectDetector + MultiCameraPipeline, config-driven.
│   │                          # Paths are repo-root-relative internally, so it runs from any CWD.
│   └── ws_broadcast_server.py # Bridges the local output broadcast to a browser-facing WebSocket
│                              # (separate process from process_video.py -- see Live Viewer below)
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
│   ├── Fast-SCNN-pytorch/         # Git submodule -> github.com/Tramac/Fast-SCNN-pytorch @ 0638517
│   │                              # (registered in .gitmodules; run `git submodule update --init` after clone)
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

This project uses a plain `venv` (not Conda) at `.venv/` in the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt websockets
```

`websockets` isn't in `requirements.txt` (it's only needed by `scripts/ws_broadcast_server.py`, not the detection pipeline itself) — install it alongside if you want the live viewer.

> **Note:** YOLOv8 (`ultralytics`) will automatically download model weights (e.g., `yolov8n.pt`) on first use if they are not already present locally. Ensure you have an internet connection on first run.

### GPU support

`requirements.txt` pins `torch==2.9.1` — its default PyPI wheel already bundles a recent CUDA runtime, so on a machine with an NVIDIA GPU and a reasonably current driver (checked with `nvidia-smi`), no special index URL is needed; a plain install picks a CUDA-enabled build automatically. Verify after installing:

```bash
.venv/bin/python3 -c "import torch; print(torch.cuda.is_available())"
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

Copy the optimal `pitch_deg` and `yaw_deg` values into the matching camera's entry under `detection.cameras` in `config/pipeline.yaml`.

---

## Configuration

Everything that differs machine-to-machine — where video comes from, per-camera calibration, and where results go — lives in `config/pipeline.yaml`, loaded by `src/co_perception/config.py`. Nothing in `scripts/process_video.py` is hardcoded to a specific deployment anymore; the same code runs unmodified here or on the Orin edge box by pointing it at a different copy of this file (`python3 scripts/process_video.py path/to/other-config.yaml`).

Three sections, matching the three pipeline stages:

### `ingestion` — where frames come from

```yaml
ingestion:
  mode: local_socket   # local_socket | local_file | aws_kvs
  channels:
    - channel: 0
      socket_path: /tmp/camera_decode_ch0.sock   # local_socket
      # file_path: camera_views/ch1/event1/clip.ts   # local_file
      # kvs_stream_name: v2x-backend-cam-ch1          # aws_kvs
```

- **`local_socket`** (this machine's default) — subscribes directly to the `camera/stream` project's `decode` stage broadcast (`/tmp/camera_decode_ch{N}.sock`), the same local Unix-socket pub/sub `upload_aws` already consumes. Frames arrive already decoded, each carrying a calibrated wall-clock timestamp (`abs_time`), so this mode needs no decoding of its own and gets cross-camera synchronization for free instead of the old per-stream `CAP_PROP_POS_MSEC` approach. Only valid on a machine also running that pipeline.
- **`local_file`** — reads a video file from disk, same as the original `cv2.VideoCapture(path)` behavior.
- **`aws_kvs`** — pulls an HLS stream from AWS Kinesis Video Streams by stream name, same as the original `"v2x-backend-cam" in path` behavior.

### `detection` — the method

`model_path`, `conf`, and one entry per camera under `cameras:` (intrinsics `K`, `camera_height`, `pitch_deg`/`yaw_deg`/`heading_deg`, `device_id`, `origin_lat`/`origin_lon`, city/state/country) — exactly the fields `VideoObjectDetector`'s constructor already took, now read from config instead of hand-typed per deployment.

### `output` — where results go

```yaml
output:
  save:      { enabled: true, json_path: output/multi_cam_detections.json, video_path: null }
  upload:    { enabled: false, endpoint: https://.../detections }
  broadcast: { enabled: true, socket_path: /tmp/coperception_output.sock }
```

All three can be on at once — save is local-file persistence, upload is the external V2X API push, broadcast is the new local Unix-socket fan-out (channel-tagged, JPEG-encoded annotated frames) that `scripts/ws_broadcast_server.py` picks up to feed the live viewer. `save.video_path: null` means don't write an mp4; set a path to enable it.

## Running the Pipeline

```bash
.venv/bin/python3 scripts/process_video.py                        # uses config/pipeline.yaml
.venv/bin/python3 scripts/process_video.py path/to/other.yaml      # or a different config
```

`VideoObjectDetector` and `MultiCameraPipeline` still live in `scripts/process_video.py` (they're the entry point, not part of the `co_perception` package) — but its `__main__` block now just loads the config, builds one `VideoObjectDetector` and one `ingest.frame_sources.FrameSource` per configured camera, and calls `pipeline.process_streams(sources=..., ...)`. To add a camera, add an entry under both `ingestion.channels` and `detection.cameras` in the config — no code changes needed.

### `process_streams` parameters

| Parameter | Type | Description |
|---|---|---|
| `sources` | `list[FrameSource]` | One per camera, same order as `detectors` — see `ingest/frame_sources.py` |
| `show_live` | `bool` | Display annotated frames in a live OpenCV window |
| `upload` | `bool` | Upload detection records to the V2X API |
| `output_json` | `str \| None` | Path to write all detections as JSON |
| `output_video` | `str \| None` | Path to write annotated output video |
| `output_image` | `str \| None` | Path to write a single annotated frame |
| `output_validate` | `bool` | Print per-frame detection details for debugging |
| `broadcast_sink` | `BroadcastSink \| None` | If set, each channel's annotated frame is broadcast locally every tick |

## Live Viewer

The `apps/web` project (a separate SvelteKit app, sibling directory) displays these annotated frames in a browser, channel-by-channel or all four in a grid. Getting a frame from here to a browser tab is two hops:

```
process_video.py  --(local broadcast, /tmp/coperception_output.sock)-->  ws_broadcast_server.py  --(WebSocket, :8766)-->  browser
```

`process_video.py` never talks WebSocket or knows a browser exists — it just broadcasts locally when `output.broadcast.enabled` is true, the same pattern `camera/stream`'s `decode` stage already uses for `upload_aws`. `ws_broadcast_server.py` is a deliberately separate process (not a thread inside `process_video.py`) so a bug or restart on the browser-facing side can't take down detection, and vice versa — same reasoning `camera/stream` used to split `demux`/`decode`/`upload_aws` apart.

Run both processes:

```bash
.venv/bin/python3 scripts/process_video.py &
.venv/bin/python3 scripts/ws_broadcast_server.py &
```

The bridge listens on `ws://127.0.0.1:8766` — see `apps/web`'s own README for how the frontend connects to it and the plan for putting this behind nginx at `/perception/ws` alongside the app itself.
