import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ultralytics import YOLO
import cv2
import numpy as np
import json
import uuid
import time
import requests
from co_perception.perception import tracking_utils
from co_perception.ingest.frame_sources import build_frame_source
from co_perception.output.broadcast_sink import BroadcastSink
from co_perception import config as pipeline_config
from datetime import datetime, timezone, timedelta
from math import radians, cos, sin, asin, sqrt
from co_perception.perception.tracking_utils import AppearanceExtractor, KalmanTracker

# TEMPORARY, decode-merge cost investigation -- remove after reporting.
_STAGE_MS = {"preprocess": [], "inference": [], "postprocess": [], "projection": [], "dedup_upload": []}


def _report_stage_timing_once():
    if len(_STAGE_MS["preprocess"]) != 500:
        return
    for stage, samples in _STAGE_MS.items():
        s = sorted(samples)
        print(f"[TIMING] {stage} median={s[250]:.2f}ms p90={s[450]:.2f}ms over 500 samples")

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
    def __init__(self, detectors):
        """
        Initialize the MultiCameraPipeline.
        
        Args:
            detectors: List of VideoObjectDetector instances.
            
        Returns:
            None
        """
        self.detectors = detectors
        self.all_clean_detections = []
        self.global_tracks = {} # Store global tracks
        self.local_to_global = {} # "device_id_local_track_id" -> global_id
        self.next_global_id = 0
        self.extractor = AppearanceExtractor()

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
                         target_fps=None, nominal_fps=30):
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
                (config.py's ingestion.nominal_fps), used to size the
                cross-channel timestamp-alignment tolerance below.

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

        def process_tick(frames_to_process, current_utc_str, current_epoch):
            """One synchronized batch: detect+track on whatever channels
            have an aligned frame this tick, dedupe/upload/write/broadcast
            the result. Returns False if the caller should stop (q pressed
            in the live-preview window)."""
            nonlocal frame_count
            frame_count += 1
            raw_buffer = []
            annotated_frames = []

            for i, frame in enumerate(frames_to_process):
                detector = self.detectors[i]
                if frame is None:
                    if last_valid_frames[i] is not None:
                        annotated_frames.append(cv2.resize(last_valid_frames[i], (640, 480)))
                    continue

                last_valid_frames[i] = frame.copy()

                results = detector.model.track(frame, persist=True, conf=detector.conf, tracker="botsort.yaml", verbose=False)

                if i == 0 and len(_STAGE_MS["preprocess"]) < 500:  # TEMPORARY, see top of file
                    sp = results[0].speed
                    _STAGE_MS["preprocess"].append(sp["preprocess"])
                    _STAGE_MS["inference"].append(sp["inference"])
                    _STAGE_MS["postprocess"].append(sp["postprocess"])
                    t_proj0 = time.monotonic()

                det_2d = detector.extract_detections(results[0], frame_count)
                det_3d = detector.compute_3d_detections(det_2d, current_utc_str, current_epoch)

                if i == 0 and len(_STAGE_MS["projection"]) < 500:  # TEMPORARY
                    _STAGE_MS["projection"].append((time.monotonic() - t_proj0) * 1000)

                for det in det_3d:
                    if det['object_type'] == 'person':
                        emb = self.extractor.extract(frame, det['camera_data']['bifocal_metadata']['bbox'])
                        det['embedding'] = emb
                    else:
                        det['embedding'] = None

                raw_buffer.extend(det_3d)

                if show_live or writer or output_image or broadcast_sink:
                    annotated = detector.draw_detections_3d(frame, det_3d)
                    annotated = cv2.resize(annotated, (640, 480))
                    annotated_frames.append(annotated)
                    if broadcast_sink:
                        # Tagged by the real channel index i, not by
                        # position in annotated_frames -- that list can
                        # be sparse (a camera with no frame yet and no
                        # last_valid_frames[i] contributes nothing to it).
                        broadcast_sink.send_frame(i, annotated)

            t_dedup0 = time.monotonic()  # TEMPORARY, see top of file
            # Deduplicate objects crossing the seams
            # Using a smaller radius (1.5m) so we don't accidentally merge multiple people in the same frame
            clean_batch = self.deduplicate(raw_buffer, current_epoch, merge_radius_meters=3.0)
            self.all_clean_detections.extend(clean_batch)

            # Batch Upload
            if upload and clean_batch:
                self.detectors[0].upload_batch(clean_batch)
                print(f"Frame {frame_count}: Uploaded {len(clean_batch)} unique objects (merged from {len(raw_buffer)} raw detections).")
            if len(_STAGE_MS["dedup_upload"]) < 500:  # TEMPORARY
                _STAGE_MS["dedup_upload"].append((time.monotonic() - t_dedup0) * 1000)
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
                        return False
            return True

        try:
            if all_live:
                # Live sources: genuine cross-channel synchronization via
                # each channel's own bounded history buffer (see
                # ingest/frame_sources.py's LocalSocketSource), not a
                # per-tick best-effort guess. Three earlier versions of this
                # loop were each wrong in different ways:
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
                #      aligned -- fine if one channel briefly stalls while
                #      the others already agree with each other, but a
                #      real, permanent deadlock the moment all 4 channels
                #      have a *persistent* mutual offset (confirmed directly
                #      against the real pipeline: with each channel
                #      independently 1-2s off from the others at all times,
                #      no front ever moved and nothing was ever processed).
                #
                # This version keys the sync target off each channel's
                # *newest* arrival instead, which -- unlike the oldest/front
                # item -- always advances as new frames arrive regardless of
                # whether anything's been consumed. target_t tracks whichever
                # channel is currently furthest behind (the min of everyone's
                # newest), and every channel (including the laggard) looks
                # back through its own buffered history for the frame
                # closest to that target. Only once *all 4* have a match
                # within tolerance does anything get consumed -- and only
                # then, discarding just the now-superseded older entries,
                # not everything.
                alignment_tolerance_sec = 1.5 / nominal_fps
                min_gap_sec = 1.0 / target_fps
                next_target_t = None  # don't accept a set older than this

                while True:
                    newests = [src.peek_newest() for src in caps]
                    if any(n is None for n in newests):
                        # At least one channel has never produced anything
                        # yet -- nothing to target against.
                        time.sleep(0.01)
                        continue

                    target_t = min(t for _, t in newests)
                    if next_target_t is not None and target_t < next_target_t:
                        # Not enough new progress since the last processed
                        # set for the next target_fps-spaced sample yet --
                        # wait rather than re-matching the same span.
                        time.sleep(0.01)
                        continue

                    matches = [src.find_closest(target_t, alignment_tolerance_sec) for src in caps]
                    if any(m is None for m in matches):
                        # At least one channel has nothing within tolerance
                        # of target_t (yet, or possibly ever, if it outran
                        # this channel's buffer window) -- wait and retry;
                        # target_t moves forward on its own as more data
                        # arrives, so this isn't a deadlock the way (3) was.
                        time.sleep(0.01)
                        continue

                    # All 4 genuinely matched -- commit (discard superseded
                    # history) only now that every channel has confirmed a
                    # match, not as each one is found.
                    for src, (idx, _frame, _t) in zip(caps, matches):
                        src.discard_through(idx)

                    frames_to_process = [m[1] for m in matches]
                    actual_t = min(m[2] for m in matches)
                    next_target_t = target_t + min_gap_sec
                    current_epoch = global_start_epoch + actual_t
                    current_time = global_start_time + timedelta(seconds=actual_t)
                    current_utc_str = current_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                    if not process_tick(frames_to_process, current_utc_str, current_epoch):
                        break
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
    def __init__(self, model_path, conf=0.25, K=np.eye(3,3), dist_coeffs=None, camera_height=5.0, pitch_deg=0.0, yaw_deg=0.0, heading_deg=0.0, device_id="cam-001", origin_lat=0.0, origin_lon=0.0,
                 city="", state="", country=""):
        
        """
        Args:
            model_path:      Path to YOLO model weights
            conf:            Detection confidence threshold
            K:               3x3 camera intrinsic matrix
            dist_coeffs:     Lens distortion coefficients [k1,k2,p1,p2,k3]
            camera_height:   Camera height above ground in meters
            device_id:       Unique identifier for this camera device
            origin_lat/lon:  GPS coordinates of the camera (used for XZ → GPS)
            city/state/country: Global context metadata
        """
        
        self.model = YOLO(model_path)
        self.conf = conf
        self.class_names = self.model.names
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

    def extract_detections(self, result, frame_num):
        """
        Extract 2D bounding boxes and track IDs from YOLO results.
        
        Args:
            result: YOLO inference result object.
            frame_num: Current frame number.
            
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
    
    def draw_detections_3d(self, frame, detections_3d):
        """
        Draw 3D bounding boxes, metadata, and labels on a video frame.
        
        Args:
            frame: The input video frame as a NumPy array.
            detections_3d: List of 3D detection records.
            
        Returns:
            Annotated image as a NumPy array.
        """
        annotated = frame.copy()
        for det in detections_3d:
            x1, y1 = int(det['camera_data']['bifocal_metadata']['bbox']['x1']), \
                     int(det['camera_data']['bifocal_metadata']['bbox']['y1'])
            x2, y2 = int(det['camera_data']['bifocal_metadata']['bbox']['x2']), \
                     int(det['camera_data']['bifocal_metadata']['bbox']['y2'])
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

    detectors = [
        VideoObjectDetector(
            model_path=cfg.resolve(cfg.model_path),
            conf=cfg.conf,
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

    pipeline = MultiCameraPipeline(detectors=detectors)

    # Shared t0 so every local-socket channel's msec is measured from the
    # same instant -- see frame_sources.LocalSocketSource.
    t0 = time.time()
    sources = [
        build_frame_source(ch, cfg.ingestion_mode, t0, cfg.nominal_fps, cfg.sync_buffer_seconds, cfg.target_fps)
        for ch in cfg.channels
    ]

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
            output_validate=False,
            broadcast_sink=broadcast_sink,
        )
    finally:
        if broadcast_sink:
            broadcast_sink.close()

    # Or upload all at once after processing:
    # detector.upload_all()