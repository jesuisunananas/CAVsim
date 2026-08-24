import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import nms
import cv2
import numpy as np
import json
import uuid
import time
import torch
import torch.nn.functional as F
import requests
from co_perception.perception import tracking_utils
from co_perception.ingest.frame_sources import build_frame_source
from co_perception.ingest.gpu_image_ops import (
    LETTERBOX_PAD_VALUE,
    LetterboxShapeResolver,
    letterbox_target_shape,
    track_input_from_uint8_batch,
)
from co_perception.output.broadcast_sink import BroadcastSink
from co_perception import config as pipeline_config
from datetime import datetime, timezone, timedelta
from math import radians, cos, sin, asin, sqrt
from co_perception.perception.tracking_utils import AppearanceExtractor, KalmanTracker

# Commit 3: manual batched-inference preprocessing, replicating
# ultralytics.data.augment.LetterBox and engine.predictor.BasePredictor.
# preprocess() exactly -- verified directly against the installed 8.3.243
# source, not from memory. Required because handing .track() a pre-built
# tensor SKIPS its own preprocessing entirely (predictor.py's preprocess():
# `not_tensor = not isinstance(im, torch.Tensor)`, and every letterbox/
# BGR->RGB/CHW/normalize step is gated on not_tensor) -- any mismatch here
# degrades detections silently, no error.
#
# A second, less obvious consequence of the tensor-input path: postprocess()
# rescales boxes via `ops.scale_boxes(img.shape[2:], boxes, orig_img.shape)`,
# and when the input isn't a list, `orig_imgs` becomes the input tensor
# itself (DetectionPredictor.postprocess: `if not isinstance(orig_imgs,
# list): orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]`).
# With no separate original-resolution image to reference, that rescale is
# from the letterboxed shape to itself -- a no-op. Returned box coordinates
# are therefore in letterboxed-tensor space, not the original 2560x1920
# frame's space compute_3d_detections expects. _rescale_xyxy_from_letterbox
# below undoes this.
def _letterbox(img_bgr, new_shape):
    """Resize+pad one HWC uint8 BGR image to new_shape=(h, w). Replicates
    LetterBox.__call__ with auto=False, scale_fill=False, scaleup=True,
    center=True, padding_value=114 -- the defaults BasePredictor.
    pre_transform uses whenever args.rect is False (the default), which is
    what a bare imgsz scalar would otherwise get via the normal (list-of-
    arrays) path. Returns (padded_bgr_uint8, r, pad_left, pad_top) so the
    transform can be inverted on returned box coordinates."""
    shape = img_bgr.shape[:2]  # (h, w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (round(shape[1] * r), round(shape[0] * r))  # (w, h)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    img = img_bgr
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                              value=(LETTERBOX_PAD_VALUE,) * 3)
    return img, r, left, top


def _letterboxed_batch_to_tensor(letterboxed_bgr_list, device):
    """N HWC uint8 BGR letterboxed frames (all the same shape) -> one
    (N, 3, H, W) float32 RGB tensor in [0, 1] on `device`. Replicates
    BasePredictor.preprocess()'s not-a-tensor branch (BGR->RGB, HWC->CHW,
    /255) -- that branch is exactly what's skipped when .track() is handed
    a tensor directly, so it has to happen here instead."""
    stacked = np.stack(letterboxed_bgr_list)  # (N, H, W, 3) BGR uint8
    stacked = stacked[..., ::-1]  # BGR -> RGB
    stacked = stacked.transpose((0, 3, 1, 2))  # NHWC -> NCHW
    stacked = np.ascontiguousarray(stacked)
    tensor = torch.from_numpy(stacked).to(device)
    return tensor.float() / 255.0


def _gpu_frame_to_bgr_numpy(rgb_tensor, target_hw=None):
    """CHW CUDA RGB uint8 -> HWC CPU BGR uint8 numpy, downscaling on GPU
    first if target_hw=(h, w) is given -- so annotation (off the hot
    path, but still touching every tick that draws) only ever transfers
    a small frame back to host memory, never the full-resolution one.
    Used by the gpu_decode path's draw_detections_3d bridge."""
    x = rgb_tensor
    if target_hw is not None and tuple(x.shape[1:]) != tuple(target_hw):
        x = F.interpolate(x.unsqueeze(0).float(), size=target_hw, mode="bilinear", align_corners=False)
        x = x.squeeze(0).round().clamp(0, 255).to(torch.uint8)
    hwc_rgb = x.permute(1, 2, 0).contiguous().cpu().numpy()  # (H, W, 3) RGB uint8
    return np.ascontiguousarray(hwc_rgb[:, :, ::-1])  # RGB -> BGR


def _gpu_crop_to_bgr_numpy(rgb_tensor, x1, y1, x2, y2):
    """CHW CUDA RGB uint8 -> HWC CPU BGR uint8 numpy for one bbox crop --
    crops on GPU first (a view, no copy yet) so only the small patch
    AppearanceExtractor needs crosses back to host memory, never the
    full-resolution frame. bbox coords are expected in rgb_tensor's own
    (full-resolution, original-frame) pixel space -- i.e. already passed
    through _rescale_xyxy_from_letterbox. Returns None for a
    degenerate/too-small crop, matching AppearanceExtractor.extract()'s
    own bounds check (duplicated here deliberately: this crop has to
    happen before extract() ever sees a frame, to know how much to
    transfer)."""
    _, h, w = rgb_tensor.shape
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    crop = rgb_tensor[:, y1:y2, x1:x2]
    hwc_rgb = crop.permute(1, 2, 0).contiguous().cpu().numpy()
    return np.ascontiguousarray(hwc_rgb[:, :, ::-1])


def _unletterbox_uint8_to_bgr_numpy(letterboxed_uint8, pad_left, pad_top, content_h, content_w):
    """Output-path counterpart to gpu_letterbox_to_uint8's forward
    transform: reuses the tensor already computed for inference (1280x960,
    already in GPU memory, already uint8 -- see gpu_decode_source.py,
    which now letterboxes at buffer time) instead of re-touching the
    full-resolution frame for drawing. Crops padding off using the same
    values the letterbox transform produced, then converts to BGR numpy.
    After cropping, image (0,0) matches original-frame (0,0) scaled by r,
    with no pad offset left to account for -- draw_detections_3d's
    scale=r alone is correct on the result (see process_tick's drawing
    block)."""
    cropped = letterboxed_uint8[:, pad_top:pad_top + content_h, pad_left:pad_left + content_w]
    hwc_rgb = cropped.permute(1, 2, 0).contiguous().cpu().numpy()
    return np.ascontiguousarray(hwc_rgb[:, :, ::-1])


def _rescale_xyxy_from_letterbox(x1, y1, x2, y2, r, pad_left, pad_top):
    """Invert _letterbox()'s transform on one box -- see the module
    docstring above for why .track() returns letterboxed-space coordinates
    when handed a tensor, instead of the original-frame coordinates
    compute_3d_detections needs."""
    return (
        (x1 - pad_left) / r,
        (y1 - pad_top) / r,
        (x2 - pad_left) / r,
        (y2 - pad_top) / r,
    )


def _scale_xyxy_to_letterbox(x1, y1, x2, y2, r, pad_left, pad_top):
    """Exact inverse of _rescale_xyxy_from_letterbox: original-frame-space
    box -> letterboxed-tensor-space box. Needed because det_2d/det_3d's
    stored bbox is in original-frame space (compute_3d_detections' camera
    projection needs it there), but the gpu_decode Re-ID crop now reads
    from the letterboxed tensor (GpuDecodeSource letterboxes at buffer
    time -- see gpu_decode_source.py), not the original-resolution frame
    -- cropping original-space coordinates out of a letterboxed tensor
    without this conversion would cut the wrong patch."""
    return (
        x1 * r + pad_left,
        y1 * r + pad_top,
        x2 * r + pad_left,
        y2 * r + pad_top,
    )


# orig_img placeholder for _NoRoundTripDetectionPredictor below. Content
# doesn't matter -- extract_detections only reads .boxes.xyxy/.conf/.cls/
# .id (Boxes.xyxy is a direct `self.data[:, :4]` passthrough, no re-
# clipping against orig_shape at that layer), and drawing uses this
# pipeline's own held frame arrays, never result.orig_img.
#
# Its SHAPE does matter, though -- found empirically, not assumed: the
# tracking callback (register_tracker's "on_predict_postprocess_end") is
# separate from this predictor's postprocess() and runs regardless,
# calling `tracker.update(det, result.orig_img, ...)` directly (see
# ultralytics/trackers/track.py's on_predict_postprocess_end). A (1,1,3)
# placeholder collapsed every returned box to ~1px -- caught by re-running
# verify_batched_letterbox.py after attaching this predictor, which is
# exactly why that harness gets re-run on every change here, not just
# once. Must match the model's actual input shape, i.e. img.shape[2:] at
# postprocess() call time -- built fresh per call since imgsz is the only
# thing that shape depends on and this dwarfs in cost what it replaces
# (a few MB zeros allocation vs. the avoided full-batch GPU->CPU
# round-trip).
def _orig_img_placeholder(letterboxed_hw):
    return np.zeros((*letterboxed_hw, 3), dtype=np.uint8)


class _NoRoundTripDetectionPredictor(DetectionPredictor):
    """DetectionPredictor whose postprocess() skips two things the normal
    list-input path needs but this pipeline's batched-tensor path does not
    -- see _letterboxed_batch_to_tensor and _rescale_xyxy_from_letterbox
    above for the tensor-input path this exists to serve.

    1. ops.convert_torch2numpy_batch(orig_imgs) -- reconstructs an "original
       image" by dragging the full GPU batch tensor back to CPU (permute +
       scale + clamp + cast + a synchronous .cpu()), every tick, at every
       achieved Hz. Confirmed by measurement to be a real cost: GPU
       utilization stayed pinned at 12-16% under Commit 3's batched
       inference despite the batch itself being correct (0.00px against
       this file's own verify_batched_letterbox.py harness). And what it
       reconstructs is useless here regardless of its cost: with a tensor
       input, DetectionPredictor.postprocess's own `orig_imgs = ... if not
       isinstance(orig_imgs, list)` branch has no reference to the true
       original frame, only to the letterboxed one -- this pipeline already
       rescales box coordinates itself via _rescale_xyxy_from_letterbox,
       so that reconstructed value was never read.

    2. ops.scale_boxes(img.shape[2:], boxes, orig_img.shape) -- rescales
       boxes from the model's input shape to orig_img's shape. Skipped
       entirely (not fed a shape stand-in): it would be a no-op here
       regardless, since our own rescale happens downstream of this
       method, not inside it, and a (C, H, W) tensor shape fed to code
       expecting (H, W, C) numpy would otherwise silently corrupt every
       box -- confirmed safe to drop against Boxes.xyxy, see
       _orig_img_placeholder's comment above. That comment also covers why
       orig_img still needs the real letterboxed (H, W), just not the
       full-cost GPU round-trip: the separate tracking callback
       (on_predict_postprocess_end) reads result.orig_img directly.

    NMS and Results construction are unchanged from the base class.
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
        )
        placeholder = _orig_img_placeholder(img.shape[2:])
        paths = self.batch[0]
        return [
            Results(placeholder, path=img_path, names=self.model.names, boxes=pred[:, :6])
            for pred, img_path in zip(preds, paths)
        ]


def _attach_no_roundtrip_predictor(model):
    """Force `model` (a shared ultralytics.YOLO instance) to use
    _NoRoundTripDetectionPredictor, constructed and attached the same way
    Model.predict() builds its own default predictor -- see
    engine/model.py's `if not self.predictor: self.predictor = (predictor
    or self._smart_load("predictor"))(...); self.predictor.setup_model(...)`.

    Pre-assigning model.predictor directly (rather than passing
    predictor=_NoRoundTripDetectionPredictor through .track()'s **kwargs,
    which _would_ also reach Model.predict()'s predictor= parameter, just
    less certainly) means that `if not self.predictor:` check is already
    false on the very first .track() call, so Ultralytics never has a
    chance to construct its own default DetectionPredictor first. Verified
    by construction, not assumed -- the assert below fails loudly if this
    doesn't hold, rather than silently falling back to the round-trip path.
    """
    predictor = _NoRoundTripDetectionPredictor(overrides={}, _callbacks=model.callbacks)
    predictor.setup_model(model=model.model, verbose=False)
    model.predictor = predictor
    assert isinstance(model.predictor, _NoRoundTripDetectionPredictor), (
        "model.predictor is not _NoRoundTripDetectionPredictor after attaching -- "
        "the GPU round-trip removal did not land"
    )


# TEMPORARY, decode-merge cost investigation -- remove after reporting.
_STAGE_MS = {
    "preprocess": [], "inference": [], "postprocess": [], "projection": [], "dedup_upload": [],
    # Whole-tick and draw+broadcast timers, added because the per-stage
    # numbers above summed to ~2ms while an actual tick was taking ~900ms
    # at 1.1Hz -- nothing above measured where that gap actually went.
    "tick_total": [], "draw_broadcast": [],
}
# Skip this long after process start before collecting -- otherwise the
# 500-sample window lands on CUDA kernel autotune (first calls at a new
# imgsz) and decode's post-restart buffer catch-up burst, not steady
# state, which is what Commit 2's report needs to be honest.
_TIMING_WARMUP_SEC = 90.0
_timing_start = time.monotonic()


def _record_tick_timing(warmed_up, t_draw_broadcast_total, t_tick_start):
    # TEMPORARY -- called from both of process_tick's return points, so a
    # 'q'-press early exit still gets recorded, not just the normal path.
    if not warmed_up:
        return
    if len(_STAGE_MS["draw_broadcast"]) < 500:
        _STAGE_MS["draw_broadcast"].append(t_draw_broadcast_total * 1000)
    if len(_STAGE_MS["tick_total"]) < 500:
        _STAGE_MS["tick_total"].append((time.monotonic() - t_tick_start) * 1000)


_stage_timing_reported = False  # TEMPORARY -- module-level, so the report below fires exactly once


def _report_stage_timing_once():
    # "projection" is gated on i==0 (channel 0 specifically having a
    # result this tick), unlike the other keys here which are all tick-
    # level (unconditional on any one channel's presence) since Commit 3's
    # batching change -- they can no longer be assumed to reach 500 in
    # lockstep. A channel-0 absence is routine now (session resets, see
    # MultiCameraPipeline), so checking only preprocess's length crashed
    # with IndexError the first time this ran against gpu_decode live
    # traffic: preprocess hit 500 while projection was still short.
    #
    # And once every key DOES reach 500, each one's own `< 500` append
    # guard means none of them grows any further -- so a length-only check
    # here would stay satisfied forever and print on every subsequent
    # tick, not just once, which is exactly what happened the first time
    # this ran long enough to find out (the log kept repeating this report
    # once a second until the deploying agent's own log-tailing monitor
    # got killed for excessive output). The module-level flag is what
    # actually makes this a one-time report.
    global _stage_timing_reported
    if _stage_timing_reported or any(len(samples) != 500 for samples in _STAGE_MS.values()):
        return
    _stage_timing_reported = True
    for stage, samples in _STAGE_MS.items():
        s = sorted(samples)
        print(f"[TIMING] {stage} median={s[250]:.2f}ms p90={s[450]:.2f}ms over 500 samples")


# TEMPORARY, achieved-Hz measurement -- remove after reporting. Ticks
# (process_tick calls, i.e. global cross-channel ticks, not per-channel
# frames) per wall-clock second, reported periodically post-warmup so
# Commit 2's Hz number reflects steady state, not the startup catch-up
# burst through decode's buffered backlog.
_HZ_REPORT_INTERVAL_SEC = 30.0
_hz_last_report_time = None
_hz_last_report_count = 0

def xy_to_gps(X, Z, origin_lat, origin_lon, heading_deg):
        """
        Convert local camera XZ coordinates (meters) to GPS lat/lon.
        Uses a simple flat-earth approximation (accurate within ~10km).

        Args:
            X: Right offset in meters from camera
            Z: Forward offset in meters from camera
            origin_lat: Camera GPS latitude
            origin_lon: Camera GPS longitude

        Returns:
            (latitude, longitude)
        """
        heading_rad = radians(heading_deg)
        easting = Z * sin(heading_rad) + X * cos(heading_rad)
        northing = Z * cos(heading_rad) - X * sin(heading_rad)

        METERS_PER_DEG_LAT = 111_320.0
        meters_per_deg_lon = 111_320.0 * cos(radians(origin_lat))#np.cos(np.radians(origin_lat))

        lat = origin_lat + (northing / METERS_PER_DEG_LAT)
        lon = origin_lon + (easting / meters_per_deg_lon)

        return float(lat), float(lon)

def compute_geohash(lat, lon, precision=5):
    """
    Encode lat/lon to a geohash string
    
    Args:
        lat: Latitude
        lon: Longitude
        precision: Geohash length (5 = ~5km x 5km cell)

    Returns:
        Geohash string
    """
    BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit_idx = 0
    char_val = 0
    is_lon = True

    while len(geohash) < precision:
        if is_lon:
            mid = (lon_range[0] + lon_range[1]) / 2
            if lon >= mid:
                char_val |= bits[bit_idx]
                lon_range[0] = mid
            else:
                lon_range[1] = mid
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat >= mid:
                char_val |= bits[bit_idx]
                lat_range[0] = mid
            else:
                lat_range[1] = mid

        is_lon = not is_lon
        if bit_idx < 4:
            bit_idx += 1
        else:
            geohash.append(BASE32[char_val])
            bit_idx = 0
            char_val = 0

    return "".join(geohash)

class MultiCameraPipeline:
    def __init__(self, detectors, model, conf, imgsz, device="cuda"):
        """
        Initialize the MultiCameraPipeline.

        Args:
            detectors: List of VideoObjectDetector instances (camera geometry/
                projection only as of Commit 3 -- inference itself is batched
                across all of them through the shared `model` below, not
                called per-detector; see process_tick).
            model: Single shared ultralytics.YOLO instance used for the
                batched .track() call. Must be shared (not one per detector)
                so its internal per-stream tracker state stays correctly
                indexed 0..3 to channel across ticks -- see process_tick's
                fixed-batch-index handling.
            conf: Detection confidence threshold, applied to the whole batch
                (all cameras currently share one global value -- see
                config.py).
            imgsz: Long-side target for the batched letterbox (see
                gpu_image_ops.letterbox_target_shape) -- same tunable as
                Commit 2's imgsz, now consumed here instead of per-call.
            device: torch device the batched inference tensor is built on.

        Returns:
            None
        """
        self.detectors = detectors
        self.model = model
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.all_clean_detections = []
        self.global_tracks = {} # Store global tracks
        self.local_to_global = {} # "device_id_local_track_id" -> global_id
        self.next_global_id = 0
        self.extractor = AppearanceExtractor()

        # Commit 3 batching state, lazily computed from the first real frame
        # seen (see process_tick) -- all 4 channels share one camera model's
        # native resolution (same K/cx/cy in pipeline.yaml), so one shared
        # target shape and one cached black-frame placeholder is correct,
        # not per-channel.
        self._letterbox_shape = None
        self._black_frame = None
        # gpu_decode only: (r, pad_left, pad_top, content_h, content_w)
        # from the letterbox transform every channel's GpuDecodeSource
        # already applied at buffer time -- read once from the first
        # present channel's own letterbox_params (see process_tick),
        # instead of recomputing it here every tick.
        self._gpu_letterbox_params = None

    @staticmethod
    def haversine_distance_meters(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance in meters between two GPS points.
        
        Args:
            lat1: Latitude of the first point.
            lon1: Longitude of the first point.
            lat2: Latitude of the second point.
            lon2: Longitude of the second point.
            
        Returns:
            Distance in meters between the two points.
        """
        R = 6371000.0  # Earth radius in meters
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)

        a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
        c = 2 * asin(sqrt(a))
        return R * c
    
    def deduplicate(self, raw_buffer, current_time_epoch, merge_radius_meters=1.5):
        """
        Takes a list of V2X JSON records and removes duplicates that are 
        physically too close together (overlapping camera seams).
        
        Args:
            raw_buffer: List of raw detection records.
            current_time_epoch: Current time in epoch seconds.
            merge_radius_meters: Radius in meters to consider detections as duplicates.
            
        Returns:
            List of deduplicated and tracked detection records.
        """
        clean_buffer = []

        for new_det in raw_buffer:
            is_duplicate = False
            
            for existing_det in clean_buffer:
                if new_det['object_type'] != existing_det['object_type']:
                    continue
                
                if new_det['device_id'] == existing_det['device_id']:
                    continue
                    
                dist = self.haversine_distance_meters(
                    new_det['gps_location']['latitude'], 
                    new_det['gps_location']['longitude'],
                    existing_det['gps_location']['latitude'], 
                    existing_det['gps_location']['longitude']
                )

                radius = 8.0 if new_det['object_type'] in {'car', 'truck', 'bus'} else 1.5

                if dist < radius:
                    is_duplicate = True
                    if new_det['confidence_score'] > existing_det['confidence_score']:
                        existing_det['confidence_score'] = new_det['confidence_score']
                        existing_det['gps_location'] = new_det['gps_location']
                        existing_det['device_id'] = new_det['device_id']
                        existing_det['camera_data'] = new_det['camera_data'] 
                    break
                    
            if not is_duplicate:
                clean_buffer.append(new_det)

        # 2. Temporal Tracking (Cross frames)
        tracked_buffer = []
        claimed_gids = set() # Prevent multiple detections in the same frame from claiming the same track
        vehicle_classes = {'car', 'truck', 'bus'}
        for det in clean_buffer:
            best_match_id = None
            min_dist = float('inf')
            local_key = f"{det['device_id']}_{det['track_id']}"
            
            # 1. Fast Path: Use visual local tracker ID
            if local_key in self.local_to_global:
                gid = self.local_to_global[local_key]
                if gid in self.global_tracks and gid not in claimed_gids:
                    if current_time_epoch - self.global_tracks[gid]['last_seen'] <= 40.0:
                        best_match_id = gid
            
            # 2. Slow Path: Spatial Math Search
            if best_match_id is None:
                for gid, track in self.global_tracks.items():
                    if gid in claimed_gids:
                        continue
                        
                    t_type = track['type']
                    d_type = det['object_type']
                    if t_type != d_type:
                        # Allow matches between vehicle types
                        if not (t_type in vehicle_classes and d_type in vehicle_classes):
                            continue
                            
                    dt = current_time_epoch - track['last_seen']
                    if dt > 40.0:
                        continue
                        
                    pred_lat, pred_lon = track['kf'].get_prediction(dt=dt if dt > 0 else 0.1)
                    last_lat, last_lon = track['kf'].x[0], track['kf'].x[1]
                    
                    dist_pred = self.haversine_distance_meters(
                        det['gps_location']['latitude'], det['gps_location']['longitude'],
                        pred_lat, pred_lon
                    )
                    dist_last = self.haversine_distance_meters(
                        det['gps_location']['latitude'], det['gps_location']['longitude'],
                        last_lat, last_lon
                    )
                    dist = min(dist_pred, dist_last)
                    
                    emb_sim = 0.0
                    if track.get('embedding') is not None and det.get('embedding') is not None:
                        emb_sim = np.dot(track['embedding'], det['embedding'])
                    
                    # Match to track if within 40m
                    if dist < 40.0 and dist < min_dist:
                        # Allow match if very close physically OR if visually similar
                        if dist < 30.0 or emb_sim > 0.50:
                            best_match_id = gid
                            min_dist = dist
                    
            if best_match_id is not None:
                claimed_gids.add(best_match_id)
                dt = current_time_epoch - self.global_tracks[best_match_id]['last_seen']
                self.global_tracks[best_match_id]['kf'].predict(dt=dt if dt > 0 else 0.1)
                self.global_tracks[best_match_id]['kf'].update([det['gps_location']['latitude'], det['gps_location']['longitude']])
                
                if det.get('embedding') is not None:
                    old_emb = self.global_tracks[best_match_id].get('embedding')
                    if old_emb is not None:
                        new_emb = 0.8 * old_emb + 0.2 * det['embedding']
                        self.global_tracks[best_match_id]['embedding'] = new_emb / np.linalg.norm(new_emb)
                    else:
                        self.global_tracks[best_match_id]['embedding'] = det['embedding']
                        
                self.global_tracks[best_match_id]['last_seen'] = current_time_epoch
                det['object_id'] = f"global_{self.global_tracks[best_match_id]['type']}_{best_match_id}"
                det['object_type'] = self.global_tracks[best_match_id]['type'] # Enforce stable class
                self.local_to_global[local_key] = best_match_id
            else:
                self.next_global_id += 1
                new_gid = self.next_global_id
                self.global_tracks[new_gid] = {
                    'type': det['object_type'],
                    'kf': KalmanTracker(det['gps_location']['latitude'], det['gps_location']['longitude']),
                    'embedding': det.get('embedding'),
                    'last_seen': current_time_epoch
                }
                det['object_id'] = f"global_{det['object_type']}_{new_gid}"
                self.local_to_global[local_key] = new_gid

            tracked_buffer.append(det)

        return tracked_buffer
    
    def process_streams(self, sources, show_live=True, upload=False, output_json=None, output_video=None,
                         output_image=None, output_validate=False, broadcast_sink=None,
                         target_fps=None, nominal_fps=30, max_consecutive_skips=5, sync_margin_sec=1.0):
        """
        Processes multiple videos in parallel, running YOLO, 3D math, and deduplication.

        Args:
            sources: List of frame_sources.FrameSource instances (one per detector,
                same order) -- see ingest/frame_sources.py for local-socket/
                local-file/AWS-KVS implementations, and config.py for how
                ingestion.mode picks between them.
            show_live: Boolean to display the live processing grid.
            upload: Boolean to upload detections to V2X API.
            output_json: Path to save the detections JSON.
            output_video: Path to save the annotated output video.
            output_image: Path to save a final annotated image frame.
            output_validate: Boolean to enable validation output.
            broadcast_sink: Optional output.broadcast_sink.BroadcastSink -- if set,
                each channel's annotated frame is JPEG-broadcast locally every tick
                for ws_broadcast_server.py to relay to browsers.
            target_fps: Live sources only -- process at this rate (config.py's
                ingestion.target_fps). Must be <= nominal_fps. Defaults to
                nominal_fps (no downsampling) if not given.
            nominal_fps: Live sources only -- the camera's real source rate
                (config.py's ingestion.nominal_fps), used for the channel-
                stability check the sync loop's real-time basis locks
                against.
            max_consecutive_skips: Live sources only -- how many consecutive
                ticks one channel may miss before the sync loop halts,
                waits for all four to be stable again, and re-locks the
                basis (config.py's ingestion.max_consecutive_skips).
            sync_margin_sec: Live sources only -- how far basis_0 is locked
                behind the freshest jointly-available timestamp at (re)lock
                time (config.py's ingestion.sync_margin_sec). A jitter-
                buffer margin: wide enough to absorb real delivery bursts
                (measured up to ~290ms on one channel), not just
                1/target_fps.

        Returns:
            None
        """
        if len(self.detectors) != len(sources):
            print("Error: Number of detectors must match number of sources.")
            return

        caps = sources
        frame_count = 0
        all_live = all(getattr(src, "is_live", False) for src in caps)

        if all_live:
            target_fps = target_fps or nominal_fps
            if target_fps > nominal_fps:
                raise ValueError(f"target_fps ({target_fps}) cannot exceed nominal_fps ({nominal_fps})")
            if target_fps <= 0:
                raise ValueError(f"target_fps must be positive, got {target_fps}")

        global_start_time = datetime.now(timezone.utc)
        global_start_epoch = time.time()
        fps = target_fps if all_live else 30
        if not all_live and len(caps) > 0 and hasattr(caps[0], "fps"):
            fps = int(caps[0].fps) or 30

        num_cams = len(caps)
        if num_cams == 1:
            out_size = (640, 480)
        elif num_cams == 4:
            out_size = (1280, 960) # 2x2 grid
        else:
            # Default horizontal concatenation for 2 or 3 cameras
            out_size = (640 * num_cams, 480)

        # --- Initialize the Video Writer ---
        writer = None
        if output_video and len(caps) > 0:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video, fourcc, max(1, int(fps)), out_size)

        print(f"Starting Multi-Stream Pipeline for {len(caps)} cameras...")

        last_valid_frames = [None] * len(caps)

        def process_tick(frames_to_process, current_utc_str, current_epoch, channels_present=None):
            """One synchronized batch: detect+track on whatever channels
            have an aligned frame this tick, dedupe/upload/write/broadcast
            the result. Returns False if the caller should stop (q pressed
            in the live-preview window).

            channels_present: indices into self.detectors/frames_to_process
            that actually contributed this tick, or None (file/KVS sources,
            where every channel is always present by construction) to mean
            "all of them." A live channel absent from this set has None in
            frames_to_process -- see the live sync loop below for why that
            happens routinely now, not just during an outage."""
            t_tick_start = time.monotonic()  # TEMPORARY, see top of file
            nonlocal frame_count
            frame_count += 1
            raw_buffer = []
            annotated_frames = []
            warmed_up = (time.monotonic() - _timing_start) >= _TIMING_WARMUP_SEC  # TEMPORARY, see top of file

            global _hz_last_report_time, _hz_last_report_count  # TEMPORARY
            now_mono = time.monotonic()
            if warmed_up:
                if _hz_last_report_time is None:
                    _hz_last_report_time = now_mono
                    _hz_last_report_count = frame_count
                elif now_mono - _hz_last_report_time >= _HZ_REPORT_INTERVAL_SEC:
                    dt = now_mono - _hz_last_report_time
                    dcount = frame_count - _hz_last_report_count
                    print(f"[TIMING] achieved Hz over last {dt:.1f}s: {dcount / dt:.2f}")
                    _hz_last_report_time = now_mono
                    _hz_last_report_count = frame_count

            active_channels = sorted(
                self.detectors[i].device_id
                for i in (channels_present if channels_present is not None else range(len(frames_to_process)))
            )

            # Commit 3: one batched .track() call across all 4 channels
            # instead of four sequential per-channel calls -- see the
            # Commit 3 comment block near the top of this file for why
            # every step here has to replicate Ultralytics' own
            # preprocessing exactly.
            #
            # Two frame representations flow through here depending on
            # ingestion.mode: local_socket gives BGR HWC numpy frames,
            # gpu_decode gives (3, H, W) CUDA RGB uint8 tensors (frames born
            # in VRAM, see gpu_decode_source.py). Branched on via
            # isinstance() against the first present frame this tick rather
            # than a stored config flag -- all present channels share one
            # ingestion.mode, so this is unambiguous, and it means nothing
            # here needs to know about config.py at all. The numpy path
            # below is untouched from before this branch existed, so
            # local_socket keeps working exactly as it did and gpu_decode
            # can be reverted independently of it.
            first_frame = next((f for f in frames_to_process if f is not None), None)
            is_gpu_tensor = isinstance(first_frame, torch.Tensor)
            if first_frame is not None and self._letterbox_shape is None:
                if is_gpu_tensor:
                    # Frames arrive already letterboxed now -- GpuDecodeSource
                    # letterboxes at buffer time (see gpu_decode_source.py),
                    # so the incoming shape already IS the inference target
                    # shape, not a source resolution to derive one from.
                    _, h, w = first_frame.shape  # CHW, already the target shape
                    self._letterbox_shape = (h, w)
                    self._black_frame = torch.zeros((3, h, w), dtype=torch.uint8, device=self.device)
                    first_idx = next(i for i, f in enumerate(frames_to_process) if f is not None)
                    params = caps[first_idx].letterbox_params
                    assert params is not None, f"channel {first_idx} produced a frame but has no letterbox_params yet"
                    self._gpu_letterbox_params = params
                else:
                    h, w = first_frame.shape[:2]  # HWC
                    self._black_frame = np.zeros((h, w, 3), dtype=np.uint8)
                    self._letterbox_shape = letterbox_target_shape(h, w, self.imgsz)
                print(
                    f"letterbox target shape: {self._letterbox_shape} (source {h}x{w}, "
                    f"imgsz={self.imgsz}, gpu_tensor={is_gpu_tensor})"
                )

            batch_results = [None] * len(frames_to_process)
            letterbox_r = letterbox_pad_left = letterbox_pad_top = None
            content_h = content_w = None
            letterboxed = None  # gpu path only: kept per-channel post-stack so drawing can reuse it
            if self._letterbox_shape is not None:
                # Fixed batch index: channel 0 is always slot 0, regardless
                # of which channels are actually present this tick. .track()
                # holds per-index tracker state (one BoT-SORT tracker per
                # stream index) -- a shrinking batch would silently
                # reassign every track to the wrong camera. A missing
                # channel (RTSP reset, tolerance mismatch -- routine now
                # with the relaxed sync loop, see channels_present above)
                # gets the cached black frame instead of being omitted.
                is_black = [f is None for f in frames_to_process]
                batch_src = [f if f is not None else self._black_frame for f in frames_to_process]
                if is_gpu_tensor:
                    # Already letterboxed uint8 (GpuDecodeSource does this
                    # at buffer time now) -- no per-frame letterbox call
                    # here any more, just stack and defer the float
                    # normalize to the small stacked batch.
                    letterboxed = batch_src
                    # A channel-resolution mismatch should fail with a
                    # clear message here, not a cryptic torch.stack error
                    # -- newly meaningful now that every channel is
                    # expected to already agree on one letterboxed shape
                    # at buffer time (see gpu_image_ops.LetterboxShapeResolver).
                    assert all(t.shape == letterboxed[0].shape for t in letterboxed), (
                        f"channel letterboxed shapes disagree: {[tuple(t.shape) for t in letterboxed]}"
                    )
                    stacked_uint8 = torch.stack(letterboxed, dim=0)
                    tensor = track_input_from_uint8_batch(stacked_uint8)
                    letterbox_r, letterbox_pad_left, letterbox_pad_top, content_h, content_w = self._gpu_letterbox_params
                else:
                    letterboxed = []
                    for f in batch_src:
                        padded, letterbox_r, letterbox_pad_left, letterbox_pad_top = _letterbox(f, self._letterbox_shape)
                        letterboxed.append(padded)
                    tensor = _letterboxed_batch_to_tensor(letterboxed, self.device)

                t_track0 = time.monotonic()  # TEMPORARY, see top of file
                raw_results = self.model.track(
                    tensor, persist=True, conf=self.conf,
                    imgsz=list(self._letterbox_shape), tracker="botsort.yaml", verbose=False,
                )
                track_ms = (time.monotonic() - t_track0) * 1000  # TEMPORARY

                if warmed_up and len(_STAGE_MS["preprocess"]) < 500:  # TEMPORARY, see top of file
                    sp = raw_results[0].speed
                    _STAGE_MS["preprocess"].append(sp["preprocess"])
                    _STAGE_MS["inference"].append(sp["inference"])
                    _STAGE_MS["postprocess"].append(sp["postprocess"])

                for idx in range(len(frames_to_process)):
                    if is_black[idx]:
                        # A black-frame slot must produce nothing -- assert
                        # it as a sanity check, but never rely on the model
                        # alone: drop the slot's results unconditionally
                        # (assertions can be compiled out with -O).
                        n_det = len(raw_results[idx].boxes)
                        assert n_det == 0, (
                            f"black-frame batch slot {idx} produced {n_det} detections -- "
                            "letterbox padding or normalization is wrong"
                        )
                        continue
                    batch_results[idx] = raw_results[idx]

            t_draw_broadcast_total = 0.0  # TEMPORARY, see top of file
            for i, frame in enumerate(frames_to_process):
                detector = self.detectors[i]
                if frame is None or batch_results[i] is None:
                    if last_valid_frames[i] is not None:
                        if isinstance(last_valid_frames[i], torch.Tensor):
                            annotated_frames.append(_gpu_frame_to_bgr_numpy(last_valid_frames[i], target_hw=(480, 640)))
                        else:
                            annotated_frames.append(cv2.resize(last_valid_frames[i], (640, 480)))
                    continue

                last_valid_frames[i] = frame.clone() if is_gpu_tensor else frame.copy()
                result = batch_results[i]

                t_proj0 = time.monotonic()  # TEMPORARY, see top of file
                det_2d = detector.extract_detections(
                    result, frame_count, letterbox_params=(letterbox_r, letterbox_pad_left, letterbox_pad_top)
                )
                det_3d = detector.compute_3d_detections(det_2d, current_utc_str, current_epoch)

                if i == 0 and warmed_up and len(_STAGE_MS["projection"]) < 500:  # TEMPORARY
                    _STAGE_MS["projection"].append((time.monotonic() - t_proj0) * 1000)

                for det in det_3d:
                    # So a consumer can tell partial coverage (a channel
                    # mid-session-reset, or genuinely down) apart from an
                    # empty sector actually being empty of objects -- see
                    # the live sync loop below, which is what makes
                    # active_channels a strict subset of all channels
                    # routinely now, not just during an outage.
                    det['active_channels'] = active_channels
                    if det['object_type'] == 'person':
                        bbox = det['camera_data']['bifocal_metadata']['bbox']
                        if is_gpu_tensor:
                            # Crop on GPU first -- only the small patch
                            # AppearanceExtractor needs crosses back to
                            # host memory, never the full-resolution
                            # frame (see _gpu_crop_to_bgr_numpy). bbox is
                            # in original-frame space (compute_3d_detections
                            # needs it there); `frame` here is the
                            # letterboxed tensor GpuDecodeSource buffers now
                            # -- must convert before cropping, or this cuts
                            # the wrong patch (see _scale_xyxy_to_letterbox).
                            lb_x1, lb_y1, lb_x2, lb_y2 = _scale_xyxy_to_letterbox(
                                bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'],
                                letterbox_r, letterbox_pad_left, letterbox_pad_top,
                            )
                            crop_bgr = _gpu_crop_to_bgr_numpy(frame, lb_x1, lb_y1, lb_x2, lb_y2)
                            if crop_bgr is not None:
                                ch, cw = crop_bgr.shape[:2]
                                emb = self.extractor.extract(crop_bgr, {'x1': 0, 'y1': 0, 'x2': cw, 'y2': ch})
                            else:
                                emb = None
                        else:
                            emb = self.extractor.extract(frame, bbox)
                        det['embedding'] = emb
                    else:
                        det['embedding'] = None

                raw_buffer.extend(det_3d)

                if show_live or writer or output_image or broadcast_sink:
                    t_draw0 = time.monotonic()  # TEMPORARY, see top of file
                    if is_gpu_tensor:
                        # Reuse the letterboxed tensor already buffered by
                        # GpuDecodeSource (1280x960, already in GPU memory,
                        # already uint8) instead of touching the full
                        # 2560x1920 frame -- that full-res round-trip was
                        # measured taking real GPU->CPU bandwidth every
                        # tick for a result that just got downscaled and
                        # discarded anyway. Strip padding using the SAME
                        # r/pad_left/pad_top/content_h/content_w every
                        # channel's letterbox transform produced (identical
                        # for every channel, since all 4 share one native
                        # resolution and target shape) -- see
                        # _unletterbox_uint8_to_bgr_numpy. After that crop,
                        # scale=r alone (no additive pad offset) is correct
                        # for draw_detections_3d, matching how
                        # extract_detections' rescale already put boxes in
                        # original-frame space.
                        annotated = _unletterbox_uint8_to_bgr_numpy(
                            letterboxed[i], letterbox_pad_left, letterbox_pad_top, content_h, content_w
                        )
                        annotated = detector.draw_detections_3d(annotated, det_3d, scale=letterbox_r)
                    else:
                        annotated = detector.draw_detections_3d(frame, det_3d)
                        annotated = cv2.resize(annotated, (640, 480))
                    annotated_frames.append(annotated)
                    if broadcast_sink:
                        # Tagged by the real channel index i, not by
                        # position in annotated_frames -- that list can
                        # be sparse (a camera with no frame yet and no
                        # last_valid_frames[i] contributes nothing to it).
                        broadcast_sink.send_frame(i, annotated)
                    t_draw_broadcast_total += time.monotonic() - t_draw0  # TEMPORARY

            t_dedup0 = time.monotonic()  # TEMPORARY, see top of file
            # Deduplicate objects crossing the seams
            # Using a smaller radius (1.5m) so we don't accidentally merge multiple people in the same frame
            clean_batch = self.deduplicate(raw_buffer, current_epoch, merge_radius_meters=3.0)
            self.all_clean_detections.extend(clean_batch)

            # Batch Upload
            if upload and clean_batch:
                self.detectors[0].upload_batch(clean_batch)
                print(f"Frame {frame_count}: Uploaded {len(clean_batch)} unique objects (merged from {len(raw_buffer)} raw detections).")
            if warmed_up and len(_STAGE_MS["dedup_upload"]) < 500:  # TEMPORARY
                _STAGE_MS["dedup_upload"].append((time.monotonic() - t_dedup0) * 1000)
            if warmed_up:  # TEMPORARY -- called every tick, not gated on any
                # one counter, so a slower-filling key (projection, gated on
                # channel 0 specifically having a result -- see its own
                # comment above) still gets checked and reported once it
                # eventually reaches 500, rather than the report being
                # skipped forever because dedup_upload's own counter
                # (unconditional, hits 500 first) stopped calling this.
                _report_stage_timing_once()

            if annotated_frames:
                if len(annotated_frames) == 1:
                    grid = annotated_frames[0]
                elif len(annotated_frames) == 4:
                    top_row = cv2.hconcat([annotated_frames[0], annotated_frames[1]])
                    bottom_row = cv2.hconcat([annotated_frames[2], annotated_frames[3]])
                    grid = cv2.vconcat([top_row, bottom_row])
                else:
                    grid = cv2.hconcat(annotated_frames)

                if writer:
                    writer.write(grid)
                if output_image:
                    cv2.imwrite(output_image, grid)
                if show_live:
                    cv2.imshow('V2X Multi-Camera Feed', grid)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        _record_tick_timing(warmed_up, t_draw_broadcast_total, t_tick_start)  # TEMPORARY
                        return False
            _record_tick_timing(warmed_up, t_draw_broadcast_total, t_tick_start)  # TEMPORARY
            return True

        try:
            if all_live:
                # Live sources: real-time-basis sync loop. Four earlier
                # versions of this loop were each wrong in a different way:
                #   1. A carried-forward "is this channel within 35ms of the
                #      others" gate that could permanently exclude a channel
                #      the instant it first drifted out of range, since
                #      nothing ever refreshed its timestamp after that.
                #   2. A median-based version that fixed (1) but silently
                #      processed whichever *subset* of channels happened to
                #      align each tick, quietly excluding stragglers rather
                #      than ever requiring all 4 together.
                #   3. An "oldest unconsumed frame per channel" version that
                #      fixed (2) (all 4 or nothing) but deadlocked: a
                #      channel's oldest-buffered item only advances when
                #      something pops it, and nothing gets popped until
                #      aligned -- a real, permanent deadlock the moment all
                #      4 channels have a *persistent* mutual offset
                #      (confirmed directly: with each channel independently
                #      1-2s off from the others at all times, no front ever
                #      moved).
                #   4. A "target_t = min of everyone's newest, discard just
                #      the superseded entries, process whatever aligns and
                #      mark the rest unavailable" version that fixed (3) but
                #      had its own failure mode: target_t was *derived from
                #      channel state* (the slowest channel's own newest
                #      arrival), so a channel stalled at some frozen value
                #      could pin target_t to that stale point, and the other
                #      channels' own multi-second buffers could keep
                #      satisfying matches against it -- the loop would
                #      silently crawl through buffered history instead of
                #      tracking real time. Diagnosed after a batched-
                #      inference run measured ~2ms/tick of actual stage
                #      work (preprocess+inference+postprocess) against an
                #      observed ~900ms/tick at 1.1Hz: nothing in the per-
                #      stage numbers explained the gap, because the gap
                #      wasn't in any stage -- it was in how target_t itself
                #      was chosen.
                #
                # This version's `basis` is a clock, not a derived value:
                # it advances by exactly 1/target_fps every tick, on a wall-
                # clock schedule set once at lock time and never touched
                # again based on channel state. It cannot be pinned by a
                # stalled channel, slowed by a laggard, or sped up by a fast
                # one. A channel that can't supply a frame within tolerance
                # of the current basis produces a skipped tick for the
                # *whole* pipeline (no partial output -- BoT-SORT's per-
                # index tracker state needs every tick's batch to mean the
                # same 4 cameras, not a shrinking/growing set), not a
                # stalled one: real time is preserved unconditionally.
                tolerance_sec = 1.0 / target_fps
                # "Stable" for the boot/reset-recovery wait below: each
                # channel producing at roughly nominal_fps, not just having
                # produced one frame (the first frame back from a reset is
                # the start of recovery, not evidence of it). 20/s, not
                # 30/s or 25/s: observed directly (both at startup and
                # after a reset) that a stricter bar took a very long time
                # to satisfy -- decode's own counters confirmed the real
                # rate was a steady 30/s the whole time (450 frames over a
                # 15s HEALTH interval, exactly 30.0/s), so the slowness was
                # this stability check's own polling measurement (a 50Hz
                # peek_newest() loop across 4 channels, each behind its own
                # lock, contending with the reader threads) undercounting
                # the true rate, not channels actually being unstable. 20/s
                # still rejects a channel that's genuinely not producing (a
                # stalled channel reads near 0, not high-teens/low-20s), so
                # this doesn't weaken the "actually recovered" check, it
                # just stops the check's own measurement noise from being
                # pickier than target_fps=10 actually needs.
                min_stable_frames_per_sec = round(nominal_fps * 20 / 30)

                def wait_for_stable_and_lock_basis():
                    """Block until every channel clears
                    min_stable_frames_per_sec for two CONSECUTIVE 1-second
                    windows, then lock basis_0 from the minimum newest
                    abs_time across all four (the only point every channel
                    can already serve) minus sync_margin_sec, so the first
                    real basis point is unambiguously in the past
                    everywhere -- and comfortably behind it, not just by
                    one tick, so a delivery burst that's already resolved
                    by lock time doesn't get re-triggered by starting the
                    very next tick right back at the edge of "now". Runs
                    at startup and again after any channel reset -- there's
                    nothing better to do than wait when channels genuinely
                    aren't producing frames, so this has no timeout.

                    No basis is locked while this runs, so nothing prunes
                    (the tick loop, where pruning happens, isn't running
                    either) -- buffers just keep accumulating normally,
                    bounded by their existing maxlen. That's fine now:
                    buffered frames are letterboxed uint8 at buffer time
                    (see gpu_decode_source.py), ~4x smaller than the old
                    full-res buffer, and sync_margin_sec needs real rolling
                    history near the lock point for every channel to match
                    against -- not just the single newest frame."""
                    consecutive_ok = 0
                    while consecutive_ok < 2:
                        seen = [set() for _ in caps]
                        window_start = time.monotonic()
                        while time.monotonic() - window_start < 1.0:
                            for i, src in enumerate(caps):
                                n = src.peek_newest()
                                if n is not None:
                                    seen[i].add(n[1])
                            time.sleep(0.02)
                        counts = [len(s) for s in seen]
                        consecutive_ok = consecutive_ok + 1 if all(c >= min_stable_frames_per_sec for c in counts) else 0
                        print(
                            f"[SYNC] stability check: counts={counts} "
                            f"(need >={min_stable_frames_per_sec}/s x2 consecutive) "
                            f"consecutive_ok={consecutive_ok}/2"
                        )
                    newests = [src.peek_newest() for src in caps]
                    locked = min(n[1] for n in newests) - sync_margin_sec
                    print(f"[SYNC] basis locked: basis_0={locked:.3f}")
                    return locked

                basis_0 = wait_for_stable_and_lock_basis()
                t_start = time.monotonic()
                n_tick = 0
                channel_miss_count = [0] * num_cams  # consecutive per channel -- drives the reset trigger
                channel_miss_total = [0] * num_cams  # cumulative, for rate reporting only
                overrun_count = 0
                ticks_attempted = 0
                last_log = time.monotonic()

                while True:
                    n_tick += 1
                    basis = basis_0 + n_tick / target_fps
                    # Absolute deadline, not sleep(1/target_fps): relative
                    # sleeps accumulate drift from their own overhead and
                    # from whatever process_tick costs each iteration;
                    # basis and the loop then advance in lockstep by
                    # construction instead.
                    deadline = t_start + n_tick / target_fps
                    now = time.monotonic()
                    if now < deadline:
                        time.sleep(deadline - now)
                    else:
                        # Previous tick (or this deadline check itself) ran
                        # long. Never queue -- proceed immediately with
                        # whatever basis this firing owns. A throughput
                        # signal only; must never trigger a channel reset.
                        overrun_count += 1

                    ticks_attempted += 1
                    matches = [src.find_closest(basis, tolerance_sec) for src in caps]
                    missed = [i for i, m in enumerate(matches) if m is None]

                    # Prune every tick, hit or miss -- basis advanced
                    # either way, so anything behind it is dead either
                    # way. Not to basis - 1/target_fps through the match
                    # itself -- a frame just before that point may still
                    # be the *next* tick's closest match. Pruning only on
                    # a hit let buffers grow toward maxlen (90 frames =
                    # 1.32GB/channel, 5.3GB across four) for as long as
                    # any channel kept missing, since a miss used to skip
                    # this entirely.
                    prune_before = basis - 1.0 / target_fps
                    for src in caps:
                        src.discard_older_than(prune_before)

                    if missed:
                        needs_reset = False
                        for i in missed:
                            channel_miss_count[i] += 1
                            channel_miss_total[i] += 1
                            if channel_miss_count[i] >= max_consecutive_skips:
                                print(
                                    f"[SYNC] ch{i} missed {channel_miss_count[i]} consecutive "
                                    "ticks -- resetting: halting, waiting for all 4 stable, re-locking basis"
                                )
                                needs_reset = True
                        if needs_reset:
                            basis_0 = wait_for_stable_and_lock_basis()
                            t_start = time.monotonic()
                            n_tick = 0
                            channel_miss_count = [0] * num_cams
                            # channel_miss_total NOT reset -- lifetime counter for rate reporting.
                        continue  # skip this tick entirely -- no partial output, see module comment above

                    channel_miss_count = [0] * num_cams  # every channel matched -- clear all consecutive counters

                    frames_to_process = [m[1] for m in matches]
                    actual_t = min(m[2] for m in matches)
                    current_epoch = global_start_epoch + actual_t
                    current_time = global_start_time + timedelta(seconds=actual_t)
                    current_utc_str = current_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                    if not process_tick(frames_to_process, current_utc_str, current_epoch):
                        break

                    if time.monotonic() - last_log >= 1.0:
                        newests = [src.peek_newest() for src in caps]
                        if all(n is not None for n in newests):
                            # The camera-pipeline delay: how far the
                            # freshest available frame, across all 4
                            # channels, trails wall clock. Expected around
                            # -3s (matches decode's own observed lag) and
                            # expected to be FLAT -- a trend means the
                            # camera-side clock is drifting relative to
                            # ours and the basis will eventually diverge.
                            camera_delay = (global_start_epoch + min(n[1] for n in newests)) - time.time()
                            print(
                                f"[SYNC] camera-pipeline delay={camera_delay:.3f}s (expect ~-3s, flat) | "
                                f"overrun={overrun_count}/{ticks_attempted} | "
                                f"channel_miss_total={channel_miss_total} of {ticks_attempted} ticks"
                            )
                        last_log = time.monotonic()
            else:
                # File/KVS sources: original pull-based, windowed-catchup
                # pacing -- unchanged from before live-source support
                # existed. These are pulled sequentially by this same loop
                # rather than pushed independently, so keeping them within a
                # shared window is both meaningful and safe here, unlike for
                # live sources above.
                buffered_frames = [None] * len(caps)
                buffered_msecs = [-1.0] * len(caps)
                for i, src in enumerate(caps):
                    ret, frame, msec = src.read()
                    if ret:
                        buffered_frames[i] = frame
                        buffered_msecs[i] = msec

                for i, f in enumerate(buffered_frames):
                    if f is not None:
                        last_valid_frames[i] = f.copy()

                raw_tick_count = 0
                while True:
                    valid_msecs = [m for m in buffered_msecs if m >= 0]
                    if not valid_msecs:
                        break
                    global_msec = min(valid_msecs)

                    frames_to_process = [None] * len(caps)
                    for i in range(len(caps)):
                        if buffered_msecs[i] >= 0 and buffered_msecs[i] <= global_msec + 35.0:
                            frames_to_process[i] = buffered_frames[i]
                            ret, frame, msec = caps[i].read()
                            if ret:
                                buffered_frames[i] = frame
                                buffered_msecs[i] = msec
                            else:
                                buffered_frames[i] = None
                                buffered_msecs[i] = -1.0

                    raw_tick_count += 1
                    if raw_tick_count != 1 and raw_tick_count % 2 != 0:
                        continue

                    current_offset = global_msec / 1000.0
                    current_time = global_start_time + timedelta(seconds=current_offset)
                    current_epoch = global_start_epoch + current_offset
                    current_utc_str = current_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

                    if not process_tick(frames_to_process, current_utc_str, current_epoch):
                        break

        finally:
            for src in caps:
                src.close()
            cv2.destroyAllWindows()
            print(f"Multi-Stream complete. Processed {frame_count} frames, found {len(self.all_clean_detections)} total unique objects.")

            if writer:
                writer.release()
                print(f"Video saved to: {output_video}")

            if output_image:
                print(f"Image saved to: {output_image}")
                
            if output_json:
                for det in self.all_clean_detections:
                    if 'embedding' in det:
                        del det['embedding']
                with open(output_json, 'w') as f:
                    json.dump(self.all_clean_detections, f, indent=2)
                print(f"JSON saved to: {output_json}")
            
            if output_validate:
                first_person=None
                for det in self.all_clean_detections:
                    if det.get('object_type') == 'person':
                        first_person = det
                        break
                
                if first_person:
                    metadata = first_person['camera_data']['bifocal_metadata']
                    u_val = metadata['pixel_centroid']['x']
                    v_val = metadata['bbox']['y2']

                    validation_output = {
                        "u": u_val,
                        "v": v_val
                    }
                    print(json.dumps(validation_output, indent=2))

class VideoObjectDetector:
    def __init__(self, model, K=np.eye(3,3), dist_coeffs=None, camera_height=5.0, pitch_deg=0.0, yaw_deg=0.0, heading_deg=0.0, device_id="cam-001", origin_lat=0.0, origin_lon=0.0,
                 city="", state="", country=""):

        """
        Args:
            model:           Shared ultralytics.YOLO instance (used here only for
                              class_names -- as of Commit 3, inference itself is
                              batched across all cameras through
                              MultiCameraPipeline.model, not called per-detector;
                              see process_tick). Must be the SAME instance passed
                              to MultiCameraPipeline, not a separate load.
            K:               3x3 camera intrinsic matrix
            dist_coeffs:     Lens distortion coefficients [k1,k2,p1,p2,k3]
            camera_height:   Camera height above ground in meters
            device_id:       Unique identifier for this camera device
            origin_lat/lon:  GPS coordinates of the camera (used for XZ → GPS)
            city/state/country: Global context metadata
        """

        self.class_names = model.names
        self.K = K
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        self.camera_height = camera_height
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

        self.pitch_deg = pitch_deg
        self.yaw_deg = yaw_deg
        self.heading_deg = heading_deg

        pitch = np.radians(self.pitch_deg)
        yaw = np.radians(self.yaw_deg)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])

        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])

        self.R = Ry @ Rx

        # Metadata
        self.device_id = device_id
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.city = city
        self.state = state
        self.country = country

        self.all_detections_3d = []
        print(f"Camera parameters:")
        print(f"  Intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")
        print(f"  Height: {self.camera_height}m")

    def extract_detections(self, result, frame_num, letterbox_params=None):
        """
        Extract 2D bounding boxes and track IDs from YOLO results.

        Args:
            result: YOLO inference result object.
            frame_num: Current frame number.
            letterbox_params: (r, pad_left, pad_top) from _letterbox(), or None.
                Commit 3's batched call hands .track() a pre-built tensor, which
                returns box coordinates in letterboxed-tensor space rather than
                original-frame space (see module docstring on
                _rescale_xyxy_from_letterbox) -- pass this to undo that. None
                means result.boxes is already in original-frame coordinates
                (e.g. a non-batched caller).

        Returns:
            List of 2D detection dictionaries.
        """
        detections = []

        # Check if any tracks were actually found
        if result.boxes.id is not None:
            # Get IDs as an array of integers
            track_ids = result.boxes.id.int().cpu().tolist()

            for box, track_id in zip(result.boxes, track_ids):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                if letterbox_params is not None:
                    x1, y1, x2, y2 = _rescale_xyxy_from_letterbox(x1, y1, x2, y2, *letterbox_params)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.class_names.get(cls, 'unknown')
                
                allowed_classes = {'car', 'person', 'truck'} #, 'bus', 'person', 'bike', 'bicycle', 'motor', 'motorcycle', 'rider', 'traffic light', 'traffic sign', 'train'}
                if class_name not in allowed_classes:
                    continue

                detections.append({
                    'frame': frame_num,
                    'track_id': track_id,
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': {'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)},
                    'center': {'x': float((x1 + x2) / 2), 'y': float((y1 + y2) / 2)}
                })
        return detections

    def get_class_color(self, class_id):
        """
        Get color for each class for visualization.
        
        Args:
            class_id: Integer ID of the object class.
            
        Returns:
            RGB color tuple (B, G, R).
        """
        colors = {
            0: (0, 255, 0),      # car - green
            1: (0, 255, 255),    # truck - yellow
            2: (255, 0, 255),    # bus - magenta
            3: (255, 0, 0),      # person - blue
            4: (0, 128, 255),    # bike - orange
            5: (128, 0, 255),    # motor - purple
            6: (255, 128, 0),    # rider - cyan
            7: (0, 0, 255),      # traffic light - red
            8: (128, 128, 0),    # traffic sign - teal
            9: (255, 255, 0),    # train - cyan
        }
        return colors.get(class_id, (255, 255, 255))
    
    def compute_world_coordinates(self, u, v):
        """
        Compute 3D world coordinates (X, Y, Z) from 2D pixel coordinates (u, v).
        
        Args:
            u: X pixel coordinate.
            v: Y pixel coordinate.
            
        Returns:
            Dictionary containing X, Y, Z, distance, and angle if valid, else None.
        """
        # 1. Undistort the pixel
        pixel = np.array([[u, v]], dtype=np.float32)
        undistorted = cv2.undistortPoints(pixel, self.K, self.dist_coeffs, P=self.K)
        u_u, v_u = undistorted[0][0]
        
        # 2. Create the Local Camera Ray
        ray_cam = np.array([(u_u - self.cx) / self.fx, (v_u - self.cy) / self.fy, 1.0])

        # 3. Rotate the Ray using the Extrinsics Matrix
        ray_world = self.R @ ray_cam
        dx, dy, dz = ray_world

        # 4. Intersect with the Ground
        # In OpenCV, Y points down. So the ground is at Y = camera_height.
        # If dy <= 0, the ray is pointing at or above the horizon (won't hit the ground).
        if dy <= 1e-6:
            return None
            # theta = np.arctan2(dx, dz)
            # return {
            #     "X": float(999.0 * np.sin(theta)),
            #     "Y": 0.0,
            #     "Z": float(999.0 * np.cos(theta)),
            #     "theta_rad": float(theta),
            #     "theta_deg": float(np.degrees(theta)),
            #     "distance": 999.0
            # }

        # Scaling factor to reach the ground
        t = self.camera_height / dy
        
        # Calculate final distances in meters
        X = t * dx
        Z = t * dz

        theta = np.arctan2(X, Z)
        distance = np.sqrt(X**2 + Z**2)

        pixel_plus = np.array([[u, v + 1]], dtype=np.float32)
        undistorted_plus = cv2.undistortPoints(pixel_plus, self.K, self.dist_coeffs, P=self.K)
        u_u_p, v_u_p = undistorted_plus[0][0]
        
        ray_cam_plus = np.array([(u_u_p - self.cx) / self.fx, (v_u_p - self.cy) / self.fy, 1.0])
        ray_world_plus = self.R @ ray_cam_plus
        dx_p, dy_p, dz_p = ray_world_plus
        
        if dy_p > 1e-6:
            t_p = self.camera_height / dy_p
            Z_plus = t_p * dz_p
            # The absolute difference in meters for a 1-pixel error
            uncertainty_meters = abs(Z - Z_plus)
        else:
            uncertainty_meters = 999.0 # Effectively infinite error at the horizon

        return {
            "X": float(X),
            "Y": 0.0,
            "Z": float(Z),
            "theta_rad": float(theta),
            "theta_deg": float(np.degrees(theta)),
            "distance": float(distance),
            "uncertainty_meters": float(uncertainty_meters)
        }

    def compute_3d_detections(self, detections_2d, current_utc_str=None, current_epoch=None):
        """
        Convert 2D detections to V2X-schema dicts with 3D world coordinates.
        
        Args:
            detections_2d: List of 2D detection dictionaries.
            current_utc_str: Current timestamp in UTC string format.
            current_epoch: Current time in epoch seconds.
            
        Returns:
            List of 3D detection records formatted for V2X schema.
        """
        records = []
        if current_utc_str is None or current_epoch is None:
            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            epoch_now = int(time.time())
        else:
            now_utc = current_utc_str
            epoch_now = current_epoch

        for det in detections_2d:
            # Ground-contact pixel: bottom-centre of bbox
            u = det['center']['x']
            v = det['bbox']['y2']
            world = self.compute_world_coordinates(u, v)
            if world is None:
                continue

            # Convert XZ → GPS
            lat, lon = xy_to_gps(world['X'], world['Z'], self.origin_lat, self.origin_lon, self.heading_deg)
            geohash = compute_geohash(lat, lon, precision=5)

            event_id = str(uuid.uuid4())

            record = {
                # --- V2X schema fields ---
                "event_id": event_id,
                "object_id": f"{det['class_name']}_{self.device_id}_{det['track_id']}",
                "object_type": det['class_name'],
                "timestamp_utc": now_utc, # TODO: Take a look here
                "confidence_score": round(det['confidence'], 4),
                "gps_location": {
                    "latitude": round(lat, 8),
                    "longitude": round(lon, 8)
                },
                "geohash": geohash,
                "street_name_normalized": "",
                "global_context": {
                    "city": self.city,
                    "state": self.state,
                    "country": self.country
                },
                "camera_data": {
                    "image_reference_url": "",
                    "svo2_reference_url": "",
                    "bifocal_metadata": {
                        "frame": det['frame'],
                        "bbox": det['bbox'],
                        "pixel_centroid": det['center'],
                        "world_position": world   # X, Y, Z, theta, distance
                    }
                },
                "notes": (f"theta={world['theta_deg']:.1f}deg "
                          f"dist={world['distance']:.1f}m"),
                "device_id": self.device_id,
                "ts_event": f"{now_utc}#{event_id}",
                "expires_at": epoch_now + 86400,   # expire in 24 h
                "ingested_at_epoch": epoch_now,
                "track_id": det.get('track_id')
            }
            records.append(record)
        return records

    V2X_ENDPOINT = "https://w0j9m7dgpg.execute-api.us-west-1.amazonaws.com/detections"

    def upload_detection(self, record):
        """
        POST a single V2X record to the API.
        
        Args:
            record: Dictionary containing the detection record.
            
        Returns:
            None
        """
        try:
            r = requests.post(self.V2X_ENDPOINT,
                              headers={"content-type": "application/json"},
                              data=json.dumps(record),
                              timeout=5)
            if r.status_code not in (200, 201):
                print(f"  ⚠️  Upload failed ({r.status_code}): {r.text[:120]}")
        except Exception as e:
            print(f"  ❌ Upload error: {e}")
    
    def upload_batch(self, records):
        """
        POST a list of V2X records to the API in a single request.
        
        Args:
            records: List of detection record dictionaries.
            
        Returns:
            None
        """
        if not records:
            return

        # Prepare payload: strip internal non-serializable fields (like embeddings)
        payload = []
        for r in records:
            clean_r = r.copy()
            if 'embedding' in clean_r:
                del clean_r['embedding']
            payload.append(clean_r)

        try:
            # Wrap array in the "items" object as per the API documentation
            r = requests.post(self.V2X_ENDPOINT,
                            headers={"content-type": "application/json"},
                            data=json.dumps({"items": payload}),
                            timeout=5)
            
            if r.status_code not in (200, 201):
                print(f"  ⚠️  Batch upload failed ({r.status_code}): {r.text[:120]}")
            else:
                print(f"  ✅ Uploaded batch of {len(records)} detections.")

        except Exception as e:
            print(f"  ❌ Batch upload error: {e}")

    def upload_all(self):
        """
        Upload all accumulated detections to the V2X API.
        
        Args:
            None
            
        Returns:
            None
        """
        print(f"\nUploading {len(self.all_detections_3d)} detections to V2X API...")
        for i, det in enumerate(self.all_detections_3d):
            self.upload_detection(det)
            if (i + 1) % 20 == 0:
                print(f"  Uploaded {i + 1}/{len(self.all_detections_3d)}")
        print("✅ Upload complete")
    
    def draw_detections_3d(self, frame, detections_3d, scale=1.0):
        """
        Draw 3D bounding boxes, metadata, and labels on a video frame.

        Args:
            frame: The input video frame as a NumPy array.
            detections_3d: List of 3D detection records.
            scale: Uniform multiplier applied to each detection's bbox
                coordinates before drawing. 1.0 (default) when frame is at
                the same resolution the bbox coordinates were computed at
                (the local_socket path: draw first, resize after). The
                gpu_decode path instead downscales the frame on GPU before
                any CPU transfer (see _gpu_frame_to_bgr_numpy) and draws
                directly at that smaller size, so it needs the boxes
                scaled down to match -- exact for this camera's frames
                (2560x1920, matches the 640x480 draw target's aspect ratio
                precisely), not a general-case assumption.

        Returns:
            Annotated image as a NumPy array.
        """
        annotated = frame.copy()
        for det in detections_3d:
            x1, y1 = int(det['camera_data']['bifocal_metadata']['bbox']['x1'] * scale), \
                     int(det['camera_data']['bifocal_metadata']['bbox']['y1'] * scale)
            x2, y2 = int(det['camera_data']['bifocal_metadata']['bbox']['x2'] * scale), \
                     int(det['camera_data']['bifocal_metadata']['bbox']['y2'] * scale)
            world = det['camera_data']['bifocal_metadata']['world_position']
            cls_id = next((k for k, v in self.class_names.items()
                           if v == det['object_type']), 0)
            color = self.get_class_color(cls_id)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, (int((x1 + x2) / 2), y2), 5, color, -1)

            lines = [
                f"{det['object_type']} {det['confidence_score']:.2f}",
                f"GPS: ({det['gps_location']['latitude']:.5f}, {det['gps_location']['longitude']:.5f})",
                f"Angle: {world['theta_deg']:.1f}°  Dist: {world['distance']:.1f}m"
            ]
            y_off = y1 - 10
            for i, txt in enumerate(lines):
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                yp = y_off - (len(lines) - i - 1) * (th + 5)
                cv2.rectangle(annotated, (x1, yp - th - 4), (x1 + tw + 4, yp + 2), color, -1)
                cv2.putText(annotated, txt, (x1 + 2, yp - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.putText(annotated, f"Detections: {len(detections_3d)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "config" / "pipeline.yaml")
    cfg = pipeline_config.load_config(config_path, REPO_ROOT)

    if cfg.upload.endpoint:
        VideoObjectDetector.V2X_ENDPOINT = cfg.upload.endpoint

    # One shared model for the whole pipeline (Commit 3) -- batched
    # inference needs a single instance so its per-stream tracker state
    # stays correctly indexed across ticks; see MultiCameraPipeline.
    shared_model = YOLO(cfg.resolve(cfg.model_path))
    _attach_no_roundtrip_predictor(shared_model)

    detectors = [
        VideoObjectDetector(
            model=shared_model,
            K=np.array(cam.K, dtype=np.float64),
            dist_coeffs=None,
            camera_height=cam.camera_height,
            pitch_deg=cam.pitch_deg,
            yaw_deg=cam.yaw_deg,
            heading_deg=cam.heading_deg,
            device_id=cam.device_id,
            origin_lat=cam.origin_lat,
            origin_lon=cam.origin_lon,
            city=cam.city,
            state=cam.state,
            country=cam.country,
        )
        for cam in cfg.cameras
    ]

    pipeline = MultiCameraPipeline(detectors=detectors, model=shared_model, conf=cfg.conf, imgsz=cfg.imgsz)

    # Shared t0 so every local-socket channel's msec is measured from the
    # same instant -- see frame_sources.LocalSocketSource.
    t0 = time.time()
    # gpu_decode only: shared across all 4 channels so every one letterboxes
    # to the SAME target shape, regardless of which channel's decoder
    # produces the first usable frame -- see gpu_image_ops.LetterboxShapeResolver.
    letterbox_resolver = LetterboxShapeResolver(cfg.imgsz)
    sources = [
        build_frame_source(ch, cfg.ingestion_mode, t0, cfg.nominal_fps, cfg.sync_buffer_seconds,
                            cfg.target_fps, cfg.max_buffer_ahead_sec,
                            imgsz=cfg.imgsz, letterbox_resolver=letterbox_resolver)
        for ch in cfg.channels
    ]
    print(f"ingestion sources built: {[type(s).__name__ for s in sources]} (mode={cfg.ingestion_mode!r})")
    if cfg.ingestion_mode == "gpu_decode":
        assert all(type(s).__name__ == "GpuDecodeSource" for s in sources), (
            "ingestion.mode is gpu_decode but not every source is a GpuDecodeSource -- "
            f"got {[type(s).__name__ for s in sources]}"
        )

    broadcast_sink = BroadcastSink(cfg.broadcast.socket_path) if cfg.broadcast.enabled else None

    try:
        pipeline.process_streams(
            sources=sources,
            show_live=cfg.show_live,
            upload=cfg.upload.enabled,
            output_json=cfg.resolve(cfg.save.json_path) if cfg.save.enabled else None,
            output_video=cfg.resolve(cfg.save.video_path) if cfg.save.enabled else None,
            output_image=None,
            target_fps=cfg.target_fps,
            nominal_fps=cfg.nominal_fps,
            max_consecutive_skips=cfg.max_consecutive_skips,
            sync_margin_sec=cfg.sync_margin_sec,
            output_validate=False,
            broadcast_sink=broadcast_sink,
        )
    finally:
        if broadcast_sink:
            broadcast_sink.close()

    # Or upload all at once after processing:
    # detector.upload_all()