from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
import json
import uuid
import time
import requests
from datetime import datetime, timezone, timedelta
from math import radians, cos, sin, asin, sqrt

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
    Encode lat/lon to a geohash string without external dependencies.

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
        self.detectors = detectors
        self.all_clean_detections = []
        self.global_tracks = {} # Store global tracks: global_id -> { 'type': str, 'lat': float, 'lon': float, 'last_seen': float }
        self.next_global_id = 0

    @staticmethod
    def haversine_distance_meters(lat1, lon1, lat2, lon2):
        """Calculate the great circle distance in meters between two GPS points."""
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
        """
        clean_buffer = []

        for new_det in raw_buffer:
            is_duplicate = False
            
            for existing_det in clean_buffer:
                if new_det['object_type'] != existing_det['object_type']:
                    continue
                    
                dist = self.haversine_distance_meters(
                    new_det['gps_location']['latitude'], 
                    new_det['gps_location']['longitude'],
                    existing_det['gps_location']['latitude'], 
                    existing_det['gps_location']['longitude']
                )

                if dist < merge_radius_meters:
                    is_duplicate = True
                    if new_det['confidence_score'] > existing_det['confidence_score']:
                        existing_det['confidence_score'] = new_det['confidence_score']
                        existing_det['gps_location'] = new_det['gps_location']
                        existing_det['device_id'] = new_det['device_id']
                    break
                    
            if not is_duplicate:
                clean_buffer.append(new_det)

        # 2. Temporal Tracking (Cross frames)
        tracked_buffer = []
        for det in clean_buffer:
            best_match_id = None
            min_dist = float('inf')
            
            for gid, track in self.global_tracks.items():
                if track['type'] != det['object_type']:
                    continue
                # Forget tracks that haven't been seen in > 3 seconds
                if current_time_epoch - track['last_seen'] > 3.0:
                    continue
                    
                dist = self.haversine_distance_meters(
                    det['gps_location']['latitude'], det['gps_location']['longitude'],
                    track['lat'], track['lon']
                )
                
                # Match to track if within larger tracking radius (e.g. 15m)
                if dist < 15.0 and dist < min_dist:
                    best_match_id = gid
                    min_dist = dist
                    
            if best_match_id is not None:
                self.global_tracks[best_match_id]['lat'] = det['gps_location']['latitude']
                self.global_tracks[best_match_id]['lon'] = det['gps_location']['longitude']
                self.global_tracks[best_match_id]['last_seen'] = current_time_epoch
                det['object_id'] = f"global_{det['object_type']}_{best_match_id}"
            else:
                self.next_global_id += 1
                new_gid = self.next_global_id
                self.global_tracks[new_gid] = {
                    'type': det['object_type'],
                    'lat': det['gps_location']['latitude'],
                    'lon': det['gps_location']['longitude'],
                    'last_seen': current_time_epoch
                }
                det['object_id'] = f"global_{det['object_type']}_{new_gid}"
                
            tracked_buffer.append(det)

        return tracked_buffer
    
    def process_streams(self, video_paths, show_live=True, upload=False, output_json=None, output_video=None, output_image=None, output_validate=False):
        """
        Processes multiple videos in parallel, running YOLO, 3D math, and deduplication.
        """
        if len(self.detectors) != len(video_paths):
            print("❌ Error: Number of detectors must match number of video paths.")
            return

        caps = [cv2.VideoCapture(str(path)) for path in video_paths]
        frame_count = 0
        
        global_start_time = datetime.now(timezone.utc)
        global_start_epoch = time.time()
        fps = 30
        if len(caps) > 0:
            fps = int(caps[0].get(cv2.CAP_PROP_FPS)) or 30

        # --- NEW: Initialize the Video Writer ---
        writer = None
        if output_video and len(caps) > 0:
            # We skip 9/10 frames, so adjust the output framerate so it doesn't play at 10x speed
            out_fps = max(1, fps // 10) 
            
            # Use mp4v codec for standard .mp4 output
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 2x2 grid of 640x480 frames = 1280x960 resolution
            writer = cv2.VideoWriter(output_video, fourcc, out_fps, (1280, 960))
        
        print(f"🚀 Starting Multi-Stream Pipeline for {len(caps)} cameras...")

        try:
            while True:
                # Read 1 frame from all cameras
                ret_frames = [cap.read() for cap in caps]
                frames = [f for ret, f in ret_frames if ret]
                
                # If any video ends, stop the loop
                if len(frames) != len(caps):
                    break
                    
                frame_count += 1
                
                # Speed optimization: Skip 9 out of 10 frames
                if frame_count != 1 and frame_count % 10 != 0:
                    continue

                raw_buffer = []
                annotated_frames = []

                current_offset = frame_count / fps
                current_time = global_start_time + timedelta(seconds=current_offset)
                current_epoch = int(global_start_epoch + current_offset)
                current_utc_str = current_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

                # Process each camera's frame with its specific detector
                for i, frame in enumerate(frames):
                    detector = self.detectors[i]
                    results = detector.model.track(frame, persist=True, conf=detector.conf, verbose=False)
                    
                    det_2d = detector.extract_detections(results[0], frame_count)
                    det_3d = detector.compute_3d_detections(det_2d, current_utc_str, current_epoch)
                    
                    raw_buffer.extend(det_3d)
                    
                    # Optional visualization
                    if show_live or writer or output_image:
                        annotated = detector.draw_detections_3d(frame, det_3d)
                        # Resize to fit on screen (otherwise 2560x1920 is too huge)
                        annotated = cv2.resize(annotated, (640, 480))
                        annotated_frames.append(annotated)

                # Deduplicate objects crossing the seams
                clean_batch = self.deduplicate(raw_buffer, current_epoch, merge_radius_meters=8)
                self.all_clean_detections.extend(clean_batch)

                # Batch Upload
                if upload and clean_batch:
                    self.detectors[0].upload_batch(clean_batch) 
                    print(f"Frame {frame_count}: Uploaded {len(clean_batch)} unique objects (merged from {len(raw_buffer)} raw detections).")

                # --- NEW: Build the 2x2 Grid View and Save ---
                if annotated_frames:
                    if len(annotated_frames) == 1:
                        grid = annotated_frames[0]
                    elif len(annotated_frames) == 4:
                        top_row = cv2.hconcat([annotated_frames[0], annotated_frames[1]])
                        bottom_row = cv2.hconcat([annotated_frames[2], annotated_frames[3]])
                        grid = cv2.vconcat([top_row, bottom_row])
                    else:
                        grid = cv2.hconcat(annotated_frames)
                    
                    # Save to file if output_video was provided
                    if writer:
                        writer.write(grid)

                    if output_image:
                        cv2.imwrite(output_image, grid)
                    
                    # Show on screen if requested
                    if show_live:
                        cv2.imshow('V2X Multi-Camera Feed', grid)
                        # wait key was 1
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

        finally:
            for cap in caps:
                cap.release()
            cv2.destroyAllWindows()
            print(f"✅ Multi-Stream complete. Processed {frame_count} frames, found {len(self.all_clean_detections)} total unique objects.")
            
            # --- NEW: Close the Video Writer cleanly ---
            if writer:
                writer.release()
                print(f"🎬 Video saved to: {output_video}")

            if output_image:
                print(f"🖼️ Image saved to: {output_image}")
                
            if output_json:
                with open(output_json, 'w') as f:
                    json.dump(self.all_clean_detections, f, indent=2)
                print(f"📁 JSON saved to: {output_json}")
            
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


    
    def process_streams_old(self, video_paths, show_live=True, upload=False, output_json=None):
        """
        Processes multiple videos in parallel, running YOLO, 3D math, and deduplication.
        """
        if len(self.detectors) != len(video_paths):
            print("❌ Error: Number of detectors must match number of video paths.")
            return

        caps = [cv2.VideoCapture(str(path)) for path in video_paths]
        frame_count = 0
        
        global_start_time = datetime.now(timezone.utc)
        global_start_epoch = time.time()
        fps = 30
        if len(caps) > 0:
            fps = int(caps[0].get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"🚀 Starting Multi-Stream Pipeline for {len(caps)} cameras...")

        try:
            while True:
                # Read 1 frame from all cameras
                ret_frames = [cap.read() for cap in caps]
                frames = [f for ret, f in ret_frames if ret]
                
                # If any video ends, stop the loop
                if len(frames) != len(caps):
                    break
                    
                frame_count += 1
                
                # Speed optimization: Skip 9 out of 10 frames
                if frame_count != 1 and frame_count % 10 != 0:
                    continue

                raw_buffer = []
                annotated_frames = []

                current_offset = frame_count / fps
                current_time = global_start_time + timedelta(seconds=current_offset)
                current_epoch = int(global_start_epoch + current_offset)
                current_utc_str = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")

                # Process each camera's frame with its specific detector
                for i, frame in enumerate(frames):
                    detector = self.detectors[i]
                    results = detector.model(frame, conf=detector.conf, verbose=False)
                    
                    det_2d = detector.extract_detections(results[0], frame_count)
                    det_3d = detector.compute_3d_detections(det_2d, current_utc_str, current_epoch)
                    
                    raw_buffer.extend(det_3d)
                    
                    # Optional visualization
                    if show_live:
                        annotated = detector.draw_detections_3d(frame, det_3d)
                        # Resize to fit on screen (otherwise 2560x1920 is too huge)
                        annotated = cv2.resize(annotated, (640, 480))
                        annotated_frames.append(annotated)

                # Deduplicate objects crossing the seams
                clean_batch = self.deduplicate(raw_buffer, current_epoch, merge_radius_meters=3.0)
                self.all_clean_detections.extend(clean_batch)

                # Batch Upload
                if upload and clean_batch:
                    # You can call the upload_batch from any of the detectors
                    self.detectors[0].upload_batch(clean_batch) 
                    print(f"Frame {frame_count}: Uploaded {len(clean_batch)} unique objects (merged from {len(raw_buffer)} raw detections).")

                # Build the 2x2 Grid View
                if show_live and len(annotated_frames) == 4:
                    top_row = cv2.hconcat([annotated_frames[0], annotated_frames[1]])
                    bottom_row = cv2.hconcat([annotated_frames[2], annotated_frames[3]])
                    grid = cv2.vconcat([top_row, bottom_row])
                    
                    cv2.imshow('V2X Multi-Camera Feed', grid)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            for cap in caps:
                cap.release()
            cv2.destroyAllWindows()
            print(f"✅ Multi-Stream complete. Processed {frame_count} frames, found {len(self.all_clean_detections)} total unique objects.")
            if output_json:
                with open(output_json, 'w') as f:
                    json.dump(self.all_clean_detections, f, indent=2)
                print(f"JSON saved to: {output_json}")

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
    
    def process_video(self, video_path, output_path=None, output_json=None, show_live=True, upload=False):
        
        """
        Process video, collect 3D detections, and optionally save/upload.

        Args:
            video_path:   Input video path
            output_path:  Save annotated video here (optional)
            output_json:  Save V2X JSON here (optional)
            show_live:    Show OpenCV preview window
            upload:       POST each detection to the V2X API in real time
        """
        
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {width} x {height} @ {fps}fps, {total_frames} frames")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps // 10,
                (width, height)
            )

        global_start_time = datetime.now(timezone.utc)
        global_start_epoch = time.time()

        frame = 0
        try:
            while cap.isOpened():
                ret, f = cap.read()
                if not ret:
                    break
                frame += 1
                if frame != 1 and frame % 10 != 0:
                    continue
                
                current_offset = frame / fps
                current_time = global_start_time + timedelta(seconds=current_offset)
                current_epoch = int(global_start_epoch + current_offset)
                current_utc_str = current_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

                results = self.model(f, conf=self.conf, verbose=False)
                detections_2d = self.extract_detections(results[0], frame)
                detections_3d = self.compute_3d_detections(detections_2d, current_utc_str, current_epoch)
                self.all_detections_3d.extend(detections_3d)

                # if upload:
                #     for det in detections_3d:
                #         self.upload_detection(det)

                if upload and detections_3d:
                    self.upload_batch(detections_3d)

                annotated_frame = self.draw_detections_3d(f, detections_3d)
                if writer:
                    writer.write(annotated_frame)
                if show_live:
                    cv2.imshow('YOLO Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if frame % 30 == 0:
                    print(f"Processed {frame}/{total_frames} frames ({frame/total_frames*100:.1f}%)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        print(f"\n✅ Processed {frame} frames, {len(self.all_detections_3d)} total detections")
        if output_json:
            self.save_detections_json(output_json)
            print(f"JSON saved to: {output_json}")
        if output_path:
            print(f"Output saved to: {output_path}")

    def extract_detections(self, result, frame_num):
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
                
                if class_name != 'car':
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

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        annotated = frame.copy()
        
        for det in detections:
            # Extract data
            x1 = int(det['bbox']['x1'])
            y1 = int(det['bbox']['y1'])
            x2 = int(det['bbox']['x2'])
            y2 = int(det['bbox']['y2'])
            conf = det['confidence']
            label = det['class_name']
            
            # Color based on class
            color = self.get_class_color(det['class_id'])
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            text = f"{label} {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(annotated, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add frame info
        cv2.putText(annotated, f"Frame: {detections[0]['frame'] if detections else 0}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Detections: {len(detections)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated

    def get_class_color(self, class_id):
        """Get color for each class"""
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


    # def pixel_to_ray(self, u, v):
    #     pixel = np.array([[u, v]], dtype=np.float32)
    #     undistorted = cv2.undistortPoints(pixel, self.K, self.dist_coeffs, P=self.K)
    #     u_u, v_u = undistorted[0][0]
    #     ray = np.array([(u_u - self.cx) / self.fx, (v_u - self.cy) / self.fy, 1.0])
    #     return ray / np.linalg.norm(ray)
    
    def compute_world_coordinates(self, u, v):
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
        """Convert 2D detections to V2X-schema dicts with 3D world coords."""
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
                "ingested_at_epoch": epoch_now
            }
            records.append(record)
        return records

    V2X_ENDPOINT = "https://qxacv7wah0.execute-api.us-west-1.amazonaws.com/detections"

    def upload_detection(self, record):
        """POST a single V2X record to the API."""
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
        """POST a list of V2X records to the API in a single request."""
        if not records:
            return

        try:
            # Note: We send 'records' (a list) directly, not a single 'record'
            r = requests.post(self.V2X_ENDPOINT,
                            headers={"content-type": "application/json"},
                            data=json.dumps(records),
                            timeout=5)
            
            if r.status_code not in (200, 201):
                print(f"  ⚠️  Batch upload failed ({r.status_code}): {r.text[:120]}")
            else:
                print(f"  ✅ Uploaded batch of {len(records)} detections.")

        except Exception as e:
            print(f"  ❌ Batch upload error: {e}")

    def upload_all(self):
        """Upload all accumulated detections to the V2X API."""
        print(f"\nUploading {len(self.all_detections_3d)} detections to V2X API...")
        for i, det in enumerate(self.all_detections_3d):
            self.upload_detection(det)
            if (i + 1) % 20 == 0:
                print(f"  Uploaded {i + 1}/{len(self.all_detections_3d)}")
        print("✅ Upload complete")
    
    def draw_detections_3d(self, frame, detections_3d):
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
    
    def save_detections_json(self, output_path):
        """Save all detections as a JSON array in V2X schema format."""
        with open(output_path, 'w') as f:
            json.dump(self.all_detections_3d, f, indent=2)

if __name__ == "__main__":
    K = np.array([
        [1325.4,      0, 1280.0],  # fx=1325.4, cx=1280
        [     0, 1325.4,  960.0],  # fy=1325.4, cy=960
        [     0,      0,      1]
    ], dtype=np.float64)

    # K = np.array([
    #     [1005.0,      0, 1920.0],
    #     [     0, 1076.0, 1080.0],
    #     [     0,      0,      1]
    # ], dtype=np.float64)

    base_lat = 37.91560117034595
    base_lon = -122.33478756387032

    #DONE
    cam4 = VideoObjectDetector('yolov8n.pt', 0.3, K, None, 7.0, -43.48, -22.63, 260.0, "cam-001-ch4", base_lat, base_lon, "Richmond", "CA", "USA")
    cam1 = VideoObjectDetector('yolov8n.pt', 0.3, K, None, 7.0, -39.20, -46.06, 200.0, "cam-001-ch1", base_lat, base_lon, "Richmond", "CA", "USA")
    cam3 = VideoObjectDetector('yolov8n.pt', 0.3, K, None, 7.0, -30.42, 14.58, 315.0, "cam-001-ch3", base_lat, base_lon, "Richmond", "CA", "USA")
    cam2 = VideoObjectDetector('yolov8n.pt', 0.3, K, None, 7.0, -40.52, 71.25, 300.0,"cam-001-ch2", base_lat, base_lon, "Richmond", "CA", "USA")
    
    
    #cam1 = VideoObjectDetector('yolov8n.pt', 0.3, K, None, 7.0, -39.20, -46.06, 200.0, "cam-001-ch1", base_lat, base_lon, "Richmond", "CA", "USA")
    #cam1 = VideoObjectDetector('yolov8n.pt', 0.3, K, None, 7.0, -39.20, -46.06, 200.0, "cam-001-ch1", base_lat, base_lon, "Richmond", "CA", "USA")
    #cam2 = VideoObjectDetector('yolov8n.pt', 0.3, K, None, 7.0, -40.52, 71.25, 300.0,"cam-001-ch2", base_lat, base_lon, "Richmond", "CA", "USA")
    #cam3 = VideoObjectDetector('yolov8n.pt', 0.3, K, None, 7.0, -32.63, 9.53, 315.0, "cam-001-ch3", base_lat, base_lon, "Richmond", "CA", "USA")
    #cam4 = VideoObjectDetector('yolov8n.pt', 0.3, K, None, 7.0, -43.48, -22.63, 260.0, "cam-001-ch4", base_lat, base_lon, "Richmond", "CA", "USA")
    #video_path = 'camera_views/ch4/Centerline_NE-SW_16m_ch4.png'

    #cam1.process_video(video_path=video_path, output_json='multi_cam_detections.json', show_live=True, upload=False)
    # cam2 = VideoObjectDetector('yolov8n.pt', 0.3, K, None, 7.0, -17.21, 88.68, "cam-001-ch2", base_lat, base_lon, "Richmond", "CA", "USA")
    # cam3 = VideoObjectDetector('yolov8n.pt', 0.3, K, None, 7.0, -26.32, 50.44, "cam-001-ch3", base_lat, base_lon, "Richmond", "CA", "USA")
    # cam4 = VideoObjectDetector('yolov8n.pt', 0.3, K, None, 7.0, -43.67, -39.49, "cam-001-ch4", base_lat, base_lon, "Richmond", "CA", "USA")
    #cam4.process_video(video_path=video_path, output_json='multi_cam_detections.json', show_live=True, upload=False)
    #pipeline = MultiCameraPipeline(detectors=[cam1, cam2, cam3, cam4])
    pipeline = MultiCameraPipeline(detectors=[cam1,cam2])

    video_paths = [
        #'camera_views/ch1/event3/sensor_0_20260302_123255.ts'
        #'camera_views/ch1/NE-SE_5m_ch1.png'
        #'camera_views/ch1/center/EastRoad_center_0_ch1.png',
        #'camera_views/ch4/NE-SE_5m_ch4.png'
        'camera_views/ch1/event1/sensor_0_20260302_122940.ts',
        'camera_views/ch2/event1/sensor_1_20260302_122940.ts',
        # 'camera_views/ch3/event2/sensor_2_20260302_123039.ts',
        # 'camera_views/ch4/event2/sensor_3_20260302_123039.ts'
        #'camera_views/ch4/event3/sensor_3_20260302_123255.ts'
        #'camera_views/ch3/event3/sensor_2_20260302_123255.ts'
    ]

    pipeline.process_streams(
        video_paths=video_paths, 
        show_live=True, 
        upload=False,
        output_json='multi_cam_detections.json',
        output_video=None,#'output.mp4',
        output_image=None, #'annotated_output.jpg',
        output_validate=False
    )

    # Or upload all at once after processing:
    # detector.upload_all()