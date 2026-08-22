"""Read-only, minimal reimplementation of camera/stream/common/protocol.py's
MSG_FRAME framing -- co-perception only ever consumes decode's broadcast,
never produces it, so only unpack_frame is needed here. Duplicated rather
than imported cross-repo for the same reason as common/broadcast.py: this
project should still run on a machine that doesn't have camera/stream
checked out (with ingestion.mode set to local_file or aws_kvs instead).

Wire format must match camera/stream/common/protocol.py exactly -- verified
against that file directly, not re-derived. Header: height, width, channels
(uint32/uint32/uint8) + abs_time (float64, NaN if uncalibrated) + raw_rtp_ts
(uint32), big-endian, followed by raw pixel bytes.
"""
import struct

MSG_FRAME = 3

_FRAME_HEADER = struct.Struct(">IIBdI")


def unpack_frame(payload):
    height, width, channels, abs_time_val, raw_rtp_ts = _FRAME_HEADER.unpack_from(payload, 0)
    abs_time = None if abs_time_val != abs_time_val else abs_time_val  # NaN check
    return height, width, channels, abs_time, raw_rtp_ts, payload[_FRAME_HEADER.size:]


if __name__ == "__main__":
    assert unpack_frame(_FRAME_HEADER.pack(1920, 2560, 3, 1786508148.309, 3074545224) + b"fake-pixels") == (
        1920, 2560, 3, 1786508148.309, 3074545224, b"fake-pixels",
    )
    h, w, c, at, rt, _ = unpack_frame(_FRAME_HEADER.pack(1920, 2560, 3, float("nan"), 3074545224) + b"x")
    assert at is None
    print("all self-tests passed")
