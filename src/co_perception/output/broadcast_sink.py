"""Broadcast annotated frames locally for the WebSocket viewer bridge."""
import cv2

from co_perception.common.broadcast import BroadcastServer
from co_perception.common.protocol import MSG_ANNOTATED_FRAME, pack_annotated_frame

_JPEG_QUALITY = 80


class BroadcastSink:
    def __init__(self, socket_path):
        self._server = BroadcastServer(socket_path)

    def send_frame(self, channel, frame):
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        if not ok:
            return
        self._server.send(MSG_ANNOTATED_FRAME, pack_annotated_frame(channel, encoded.tobytes()))

    def close(self):
        self._server.close()
