"""Read-only, minimal reimplementation of camera/stream/common/protocol.py's
MSG_VIDEO/MSG_SR_ANCHOR framing -- gpu_decode_source.py subscribes directly
to demux's broadcast (compressed H.264 access units + RTCP SR anchors)
instead of decode's, so it needs this half of the protocol too. Duplicated
rather than imported cross-repo for the same reason as decode_protocol.py:
this project should still run on a machine that doesn't have camera/stream
checked out.

Wire format must match camera/stream/common/protocol.py exactly -- verified
against that file directly, not re-derived.
"""
import struct

MSG_VIDEO = 1
MSG_SR_ANCHOR = 2

# pts_ns (uint64, demux's own pipeline-relative running time -- not used
# here, see gpu_decode_source.py for why) + raw_rtp_ts (uint32), big-endian,
# followed by the raw H.264 access unit bytes.
_VIDEO_HEADER = struct.Struct(">QI")

# raw_rtp_ts (uint32) + ntp_time (float64), big-endian.
_SR_ANCHOR = struct.Struct(">Id")


def unpack_video(payload):
    pts_ns, raw_rtp_ts = _VIDEO_HEADER.unpack_from(payload, 0)
    return pts_ns, raw_rtp_ts, payload[_VIDEO_HEADER.size:]


def unpack_sr_anchor(payload):
    return _SR_ANCHOR.unpack(payload)


if __name__ == "__main__":
    assert unpack_video(_VIDEO_HEADER.pack(12345, 3074545224) + b"fake-h264-bytes") == (
        12345, 3074545224, b"fake-h264-bytes",
    )
    assert unpack_sr_anchor(_SR_ANCHOR.pack(3074545224, 1786508148.309)) == (3074545224, 1786508148.309)
    print("all self-tests passed")
