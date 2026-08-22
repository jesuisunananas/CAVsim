"""Message framing for co-perception's own output broadcast (process_video.py
-> ws_broadcast_server.py, see common/broadcast.py for the transport).

This is a *different* wire format from camera/stream's decode broadcast --
that one carries raw decoded pixels, this one carries an already-annotated
JPEG-encoded frame per channel, since its only consumer is a WebSocket
bridge that just needs to hand bytes to a browser <img>.
"""
import struct

MSG_ANNOTATED_FRAME = 1

# channel (uint8) + jpeg byte length is implicit (rest of payload)
_FRAME_HEADER = struct.Struct(">B")


def pack_annotated_frame(channel, jpeg_bytes):
    return _FRAME_HEADER.pack(channel) + jpeg_bytes


def unpack_annotated_frame(payload):
    (channel,) = _FRAME_HEADER.unpack_from(payload, 0)
    return channel, payload[_FRAME_HEADER.size:]


if __name__ == "__main__":
    assert unpack_annotated_frame(pack_annotated_frame(2, b"fake-jpeg-bytes")) == (2, b"fake-jpeg-bytes")
    print("all self-tests passed")
