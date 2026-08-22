#!/usr/bin/env python3
"""Pure RTP-timestamp-to-absolute-time calibration math -- no GStreamer/gi
dependency, so it's testable in isolation from the camera/pipeline.

Ported as-is from camera/stream/decode/rtp_time_calibration_helper.py --
this file's logic is unchanged; it just lives here too so co-perception's
GPU decode source (see gpu_decode_source.py) doesn't depend on the camera
repo. Keep the two in sync if either changes.

Each camera channel's RTP clock increments at RTP_CLOCK_RATE_HZ from an
arbitrary, per-session origin. RTCP Sender Reports give periodic anchor pairs
(rtp_ts, ntp_time) mapping that clock to real time. This offset-only model
(no rate correction) was validated in this project's earlier SRT-based
design: <0.15% clock-rate-vs-real-time drift across all 4 channels over a
~28s steady-state window, so re-anchoring on every new SR is sufficient --
it self-corrects rather than needing to accumulate a rate estimate.
"""

RTP_CLOCK_RATE_HZ = 90000
RTP_MOD = 1 << 32


def wrapped_diff(a, b, mod=RTP_MOD):
    """Signed distance from b to a on a wrapping mod-`mod` counter.
    Assumes the true delta magnitude is under half the modulus (~6.6 hours
    at 90kHz) -- always true here given ~5s anchor spacing."""
    d = (a - b) % mod
    if d > mod // 2:
        d -= mod
    return d


class ChannelClockCalibrator:
    """Tracks the latest RTCP SR anchor for one channel and converts raw
    RTP timestamps to absolute (Unix epoch) time."""

    def __init__(self, clock_rate_hz=RTP_CLOCK_RATE_HZ):
        self.clock_rate_hz = clock_rate_hz
        self._anchor_rtp_ts = None
        self._anchor_ntp_time = None
        self.anchor_count = 0

    @property
    def has_anchor(self):
        return self._anchor_rtp_ts is not None

    def update_anchor(self, rtp_ts, ntp_time):
        self._anchor_rtp_ts = rtp_ts
        self._anchor_ntp_time = ntp_time
        self.anchor_count += 1

    def to_abs_time(self, frame_rtp_ts):
        """Convert a raw RTP timestamp to absolute Unix time, or None if no
        anchor has arrived yet."""
        if self._anchor_rtp_ts is None:
            return None
        delta_ticks = wrapped_diff(frame_rtp_ts, self._anchor_rtp_ts)
        return self._anchor_ntp_time + delta_ticks / self.clock_rate_hz


if __name__ == "__main__":
    # normal in-range delta
    assert wrapped_diff(1090000, 1000000) == 90000
    assert wrapped_diff(1000000, 1090000) == -90000

    # forward wraparound: a is just past the wrap point from b
    assert wrapped_diff(1000, 2**32 - 1000) == 2000

    # backward wraparound: b is just past the wrap point from a
    assert wrapped_diff(2**32 - 1000, 1000) == -2000

    # no anchor yet
    calib = ChannelClockCalibrator()
    assert not calib.has_anchor
    assert calib.to_abs_time(12345) is None

    # first anchor
    calib.update_anchor(rtp_ts=1_000_000, ntp_time=1_700_000_000.0)
    assert calib.has_anchor
    assert calib.anchor_count == 1
    # exactly 1 second of RTP ticks later
    assert calib.to_abs_time(1_000_000 + RTP_CLOCK_RATE_HZ) == 1_700_000_000.0 + 1.0
    # exactly 1 second before the anchor
    assert calib.to_abs_time(1_000_000 - RTP_CLOCK_RATE_HZ) == 1_700_000_000.0 - 1.0

    # re-anchoring updates the offset (self-correcting, not accumulating)
    calib.update_anchor(rtp_ts=2_000_000, ntp_time=1_700_000_012.0)
    assert calib.anchor_count == 2
    assert calib.to_abs_time(2_000_000) == 1_700_000_012.0

    print("all self-tests passed")
