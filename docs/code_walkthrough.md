# Code Walkthrough

How a frame gets from a camera to a fused, GPS-located, tracked object record — input, detection, fusion, output, in the order they actually run.

## Input

**Script**: `scripts/process_video.py`, **class/method**: `MultiCameraPipeline.process_streams()`.

Video sources are opened per-camera near the top of the method: each entry in `video_paths` either contains `"v2x-backend-cam"` (routed through `co_perception.ingest.kinesis_utils.get_kvs_hls_url()` — an AWS Kinesis Video Streams HLS pull) or is treated as a local file path — both end up wrapped as a `cv2.VideoCapture` object. The main loop reads frames from all 4 captures, using each stream's own `CAP_PROP_POS_MSEC` to loosely synchronize them against a shared `global_msec` clock, and **skips every other frame** (`if frame_count != 1 and frame_count % 2 != 0: continue`) — i.e. it decodes at full rate and discards half the output, not a true before-decode downsample.

This is the extension point for a new video source (e.g. a local, already-decoded Unix-socket feed with a real per-frame timestamp) — see the ingest-source design discussion in project history for what that would look like.

## Detection

**Model**: YOLOv8 (Ultralytics), loaded per-camera in `VideoObjectDetector.__init__`: `self.model = YOLO(model_path)` — currently `models/yolov8n.pt` (the generic pretrained checkpoint; `models/best.pt`, presumably fine-tuned, is available but not the active default).

**Flow**, per frame per camera, still in `process_streams()`:
```python
results = detector.model.track(frame, persist=True, conf=detector.conf, tracker="botsort.yaml", verbose=False)
det_2d = detector.extract_detections(results[0], frame_count)
det_3d = detector.compute_3d_detections(det_2d, current_utc_str, current_epoch)
```
`.track()` runs YOLOv8 detection *and* BoT-SORT tracking (per-camera local track IDs) in one call. `compute_3d_detections` then projects each 2D bounding box into real-world XZ coordinates using the camera's intrinsics (`K`) and its calibrated pitch/yaw/height (see `calibration/`), and converts that into GPS lat/lon.

This stage runs **independently per camera** — each of the 4 `VideoObjectDetector` instances runs its own YOLO model on its own frame, no shared state yet.

## Fusion

Yes — real fusion across all 4 cameras, not four independent detectors run in parallel and concatenated.

**Method**: `MultiCameraPipeline.deduplicate()`, called once per frame-batch after all 4 cameras' detections for that frame are collected into one `raw_buffer`. Two stages:

1. **Spatial dedup** — for every pair of detections from *different* cameras, if they're the same object type and within a radius (8m for vehicles, 1.5m for people) by Haversine distance, they're merged into one record, keeping whichever had higher confidence. This is what collapses the same physical object seen by two overlapping camera views into one detection.
2. **Cross-frame global tracking** — a Kalman filter (`KalmanTracker`, in `src/co_perception/perception/tracking_utils.py`) per tracked object predicts where it should be next; new detections are matched to existing global tracks by GPS-predicted position, with an appearance embedding (`AppearanceExtractor`, a small CNN, cosine similarity — computed for people only) as a tiebreaker when the spatial match is ambiguous. This is what lets an object walk out of one camera's view and into another's while keeping the *same* `object_id` (e.g. `global_person_1`), instead of becoming two separate objects.

## Output

Written at the end of `process_streams()`, in a `finally` block, so it saves even on early exit. After the `co_perception` reorg, everything lands in `output/`:

- **JSON** — `output/multi_cam_detections.json` (the `output_json` argument): the full deduplicated/fused detection+track history, one record per object-detection event. Fields include GPS location, object type/global track ID, confidence, per-camera bbox + 3D world-position math, timestamps, and a geohash — shaped for upload to a V2X API (`event_id`, `expires_at`, `ts_event`), not just local debugging. Example record:
  ```json
  {
    "object_id": "global_person_1",
    "object_type": "person",
    "device_id": "cam-001-ch1",
    "track_id": 1,
    "confidence_score": 0.6855,
    "gps_location": {"latitude": 37.91525041, "longitude": -122.33485891},
    "geohash": "9q9p8",
    "timestamp_utc": "2026-06-27T11:04:43.010Z",
    "camera_data": {
      "bifocal_metadata": {
        "bbox": {"x1": 2268.9, "y1": 82.0, "x2": 2340.1, "y2": 285.2},
        "world_position": {"X": -7.47, "Z": 38.84, "distance": 39.55, "theta_deg": -10.88}
      }
    }
  }
  ```
- **Video** — optional, `output_video` argument (e.g. `output/tracking.mp4`), a 2×2 grid of annotated frames written via `cv2.VideoWriter`. Only produced if `output_video` is passed.
- **Live upload** — if `upload=True`, each frame-batch's fused detections are pushed immediately via `detector.upload_batch()` to a V2X API, independent of and in addition to the final JSON dump — these are two separate output paths, not one feeding the other.
