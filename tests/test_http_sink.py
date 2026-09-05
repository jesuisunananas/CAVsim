import json
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from co_perception.output.http_sink import HttpSink


def _record(track_id, object_type, confidence, lat, lon):
    return {
        "object_id": track_id,
        "object_type": object_type,
        "confidence_score": confidence,
        "gps_location": {"latitude": lat, "longitude": lon},
        "camera_data": {
            "bifocal_metadata": {
                "bbox": {"x1": 10, "y1": 20, "x2": 30, "y2": 40}
            }
        },
    }


def _get_json(url):
    with urlopen(url, timeout=2) as response:
        assert response.headers["Content-Type"] == "application/json"
        return json.load(response)


def test_latest_detections_and_health_map_zero_based_channels():
    sink = HttpSink("127.0.0.1", 0)
    try:
        sink.update_frame(0, 1_800_000_000.25, [
            _record("car-camera-1-42", "car", 0.91, 37.91, -122.33)
        ])
        sink.update_frame(3, 1_800_000_001.5, [
            _record("person-camera-4-7", "person", 0.83, 37.92, -122.34)
        ])

        base = f"http://127.0.0.1:{sink.port}"
        latest = _get_json(f"{base}/detections/latest")
        assert set(latest) == {"cameras"}
        assert set(latest["cameras"]) == {"ch1", "ch4"}
        assert latest["cameras"]["ch1"] == {
            "ts": 1_800_000_000.25,
            "detections": [{
                "object_id": "car-camera-1-42",
                "object_type": "car",
                "confidence": 0.91,
                "gps_location": {"lat": 37.91, "lon": -122.33},
                "camera": "ch1",
                "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
            }],
        }
        assert latest["cameras"]["ch4"]["detections"][0]["camera"] == "ch4"

        health = _get_json(f"{base}/health")
        assert health["ok"] is True
        assert health["cameras"]["ch1"]["count"] == 1
        assert health["cameras"]["ch4"]["ts"] == 1_800_000_001.5
        assert health["cameras"]["ch4"]["age_s"] >= 0
    finally:
        sink.close()
