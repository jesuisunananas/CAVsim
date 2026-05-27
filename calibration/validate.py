from process_video import VideoObjectDetector, MultiCameraPipeline
from pathlib import Path
import numpy as np

folder_path = Path('camera_views/ch1/center')

files = [item for item in folder_path.iterdir() if item.is_file()]

calibration_pts = []

K = np.array([
        [1325.4,      0, 1280.0],  # fx=1325.4, cx=1280
        [     0, 1325.4,  960.0],  # fy=1325.4, cy=960
        [     0,      0,      1]
    ], dtype=np.float64)

base_lat = 37.91560117034595
base_lon = -122.33478756387032

#cam1 = VideoObjectDetector('yolov8n.pt', 0.3, K, None, 7.0, -103.63, -166.80, 200.0, "cam-001-ch1", base_lat, base_lon, "Richmond", "CA", "USA")
#cam2 = VideoObjectDetector('yolov8n.pt', 0.3, K, None, 7.0, -43.48, -22.63, 200.0,"cam-001-ch2", base_lat, base_lon, "Richmond", "CA", "USA")
#cam4 = VideoObjectDetector('yolov8n.pt', 0.2, K, None, 7.0, -43.48, -22.63, 260.0, "cam-001-ch4", base_lat, base_lon, "Richmond", "CA", "USA")

#pipeline = MultiCameraPipeline(detectors=[cam4])
for file in files:
    cam1 = VideoObjectDetector('yolov8n.pt', 0.3, K, None, 7.0, -103.63, -166.80, 200.0, "cam-001-ch1", base_lat, base_lon, "Richmond", "CA", "USA")
    pipeline = MultiCameraPipeline(detectors=[cam1])
    video_paths = [file]
    pipeline.all_clean_detections = []
    print(f"\n--- Processing: {file.name} ---")
    pipeline.process_streams(
        video_paths=video_paths, 
        show_live=True, 
        upload=False,
        output_json=None,
        output_image=None,
        output_video=None,
        output_validate=True
    )