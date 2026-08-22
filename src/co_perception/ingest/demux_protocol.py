"""Read-only, minimal reimplementation of camera/stream/common/protocol.py's
MSG_VIDEO/MSG_SR_ANCHOR/MSG_SESSION_RESET framing -- gpu_decode_source.py
subscribes directly to demux's broadcast (compressed H.264 access units +
RTCP SR anchors + explicit session-reset signals) instead of decode's, so
it needs this half of the protocol too. Duplicated rather than imported
cross-repo for the same reason as decode_protocol.py: this project should
still run on a machine that doesn't have camera/stream checked out.

Wire format must match camera/stream/common/protocol.py exactly -- verified
against that file directly, not re-derived. This is one of three things
that now have to stay in sync by hand across the two repos (the others:
rtp_time_calibration.py's math, and decode/main.py's SANITY_LAG_BOUND_SEC
vs frame_sources.py's MAX_ABS_TIME_SKEW_SEC) -- see project memory.
"""
import struct

MSG_VIDEO = 1
MSG_SR_ANCHOR = 2
MSG_SESSION_RESET = 4

# pts_ns (uint64, demux's own pipeline-relative running time -- not used
# here, see gpu_decode_source.py for why) + raw_rtp_ts (uint32), big-endian,
# followed by the raw H.264 access unit bytes.
_VIDEO_HEADER = struct.Struct(">QI")

# raw_rtp_ts (uint32) + ntp_time (float64), big-endian.
_SR_ANCHOR = struct.Struct(">Id")

# raw_rtp_ts (uint32), the new session's first raw wire timestamp. Emitted
# by demux the instant it reconnects to the camera, before resuming
# MSG_VIDEO delivery for the new session -- see demux/main.py's
# _on_rtp_buffer. Replaces guessing a session boundary from a large tick
# delta downstream (the previous approach: a threshold guess, wrong in
# both directions -- a genuine long stall reads as a reconnect, and a
# reconnect whose new random origin lands close to the old one is missed).
_SESSION_RESET = struct.Struct(">I")


def unpack_video(payload):
    pts_ns, raw_rtp_ts = _VIDEO_HEADER.unpack_from(payload, 0)
    return pts_ns, raw_rtp_ts, payload[_VIDEO_HEADER.size:]


def unpack_sr_anchor(payload):
    return _SR_ANCHOR.unpack(payload)


def unpack_session_reset(payload):
    return _SESSION_RESET.unpack(payload)[0]


if __name__ == "__main__":
    assert unpack_video(_VIDEO_HEADER.pack(12345, 3074545224) + b"fake-h264-bytes") == (
        12345, 3074545224, b"fake-h264-bytes",
    )
    assert unpack_sr_anchor(_SR_ANCHOR.pack(3074545224, 1786508148.309)) == (3074545224, 1786508148.309)
    assert unpack_session_reset(_SESSION_RESET.pack(3074545224)) == 3074545224
    print("all self-tests passed")
