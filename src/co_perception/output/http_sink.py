"""Non-blocking local HTTP view of the latest detections per camera."""
from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_LOG = logging.getLogger(__name__)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class HttpSink:
    """Keeps one frame per channel in memory and exposes JSON snapshots."""

    def __init__(self, host: str, port: int):
        self._lock = threading.Lock()
        self._cameras: dict[str, dict] = {}
        sink = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/detections/latest":
                    sink._send_json(self, sink.latest())
                elif self.path == "/health":
                    sink._send_json(self, sink.health())
                else:
                    sink._send_json(self, {"error": "not found"}, status=404)

            def log_message(self, _format, *_args):
                return

        self._server = _Server((host, port), Handler)
        self._thread = threading.Thread(
            target=self._serve,
            name="co-perception-http-sink",
            daemon=True,
        )
        self._thread.start()

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def _serve(self) -> None:
        try:
            self._server.serve_forever(poll_interval=0.2)
        except Exception:
            _LOG.exception("HTTP detections server stopped unexpectedly")

    @staticmethod
    def _send_json(handler: BaseHTTPRequestHandler, body: dict, status: int = 200) -> None:
        try:
            payload = json.dumps(body, separators=(",", ":"), allow_nan=False).encode("utf-8")
            handler.send_response(status)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            _LOG.exception("HTTP detections response failed")

    def update_frame(self, channel: int, ts: float, records: list[dict]) -> None:
        """Replace one channel's snapshot without propagating sink failures."""
        try:
            camera = f"ch{int(channel) + 1}"
            detections = [self._to_detection(record, camera) for record in records]
            snapshot = {"ts": float(ts), "detections": detections}
            with self._lock:
                self._cameras[camera] = snapshot
        except Exception:
            _LOG.exception("HTTP detections update failed for channel %r", channel)

    @staticmethod
    def _to_detection(record: dict, camera: str) -> dict:
        gps = record["gps_location"]
        detection = {
            "object_id": str(record["object_id"]),
            "object_type": record["object_type"],
            "confidence": float(record["confidence_score"]),
            "gps_location": {
                "lat": float(gps["latitude"]),
                "lon": float(gps["longitude"]),
            },
            "camera": camera,
        }
        bbox = record.get("camera_data", {}).get("bifocal_metadata", {}).get("bbox")
        if bbox is not None:
            detection["bbox"] = {
                key: float(bbox[key]) for key in ("x1", "y1", "x2", "y2")
            }
        return detection

    def latest(self) -> dict:
        with self._lock:
            cameras = dict(self._cameras)
        return {"cameras": cameras}

    def health(self) -> dict:
        now = time.time()
        with self._lock:
            cameras = dict(self._cameras)
        return {
            "ok": True,
            "cameras": {
                camera: {
                    "ts": snapshot["ts"],
                    "age_s": max(0.0, now - snapshot["ts"]),
                    "count": len(snapshot["detections"]),
                }
                for camera, snapshot in cameras.items()
            },
        }

    def close(self) -> None:
        try:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=1.0)
        except Exception:
            _LOG.exception("HTTP detections server shutdown failed")
