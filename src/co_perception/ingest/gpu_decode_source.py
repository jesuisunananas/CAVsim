"""GPU-resident decode source for co-perception's live ingestion.

Subscribes directly to camera/stream's demux broadcast (compressed H.264
access units, see demux_protocol.py) instead of decode's (raw BGR frames,
see frame_sources.py's LocalSocketSource) -- decodes on GPU via
PyNvVideoCodec instead of going through camera/stream's decode process at
all, so frames stay in GPU memory from decode through inference instead of
crossing back to host memory and a socket.

Ported from camera/stream/decode/main.py:
- RTCP SR anchoring -> abs_time (rtp_time_calibration.py, moved as-is)
- raw_rtp_ts tick unwrapping and MAX_PLAUSIBLE_TICK_JUMP session-boundary
  detection (same logic, copied below)
- the reader-thread pattern (demux drops slow clients, same as decode's own
  reconnect loop)
- _contains_idr (same byte-scanning, no GStreamer dependency to begin with)

Deliberately NOT ported: decode/main.py's MAX_LAG_SEC-triggered flush/skip-
until-IDR logic. That existed because avdec_h264's own internal buffering
could accumulate an unbounded backlog under CPU contention with nothing to
cap it. PyNvVideoCodec.Decode() is a direct, synchronous call -- there's no
comparable internal queue that can silently grow, so there's nothing
equivalent to detect or flush. What IS kept is a same-shaped HEALTH log
line, for visibility.

Decimation happens strictly after decode, never before it: dropping H.264
access units before feeding them to the decoder corrupts every subsequent
P-frame in the same GOP, since this camera's stream is a standard IPPP
chain (confirmed via ffprobe: one IDR then 60 consecutive P-frames, no
B-frames) where each P-frame references the immediately preceding decoded
frame, not just the last IDR. NVDEC conceals this rather than erroring --
silently corrupted/smeared output that still "decodes successfully" --
exactly the failure mode this whole project is trying to avoid. So every
access unit gets decoded, and only the decoded frames needed to hit
target_fps are kept.

Buffered frame representation: NV12, not BGR. DecodedFrame.cuda() returns
TWO separate CUDA Array Interface views for NV12 -- a full-resolution Y
(luma) plane and a half-resolution interleaved UV (chroma) plane, confirmed
directly (they are NOT one combined view; taking only the first silently
discards all colour). Both are buffered. Colour-space conversion to
whatever format inference needs happens downstream, not here -- see
gpu_decode_source.nv12_to_bgr_numpy for the temporary bridge used by this
commit's verification harness only; the real production conversion (exact
Ultralytics letterbox/RGB/CHW/normalize, done on-GPU) is Commit 3's job.
"""
from __future__ import annotations

import ctypes
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
import PyNvVideoCodec as nvc

from co_perception.common.broadcast import BroadcastClient
from co_perception.ingest.demux_protocol import MSG_SR_ANCHOR, MSG_VIDEO, unpack_sr_anchor, unpack_video
from co_perception.ingest.rtp_time_calibration import ChannelClockCalibrator, wrapped_diff

RTP_CLOCK_HZ = 90000
# Same rationale as camera/stream/decode/main.py: bigger than any real
# frame-to-frame gap could ever be, so a jump this large can only mean the
# underlying RTP session was replaced (demux restarted), not normal wrap.
MAX_PLAUSIBLE_TICK_JUMP = RTP_CLOCK_HZ * 10

# See _handle_access_unit's liveness guard: how many consecutive decoded
# frames may share an unchanging PTS before that's treated as a stalled
# clock rather than coincidence. ~1s at 30fps -- long enough that ordinary
# jitter never trips it, short enough to catch a stall fast.
STUCK_OUTPUT_THRESHOLD = 30


def _contains_idr(h264_bytes):
    """True if this byte-stream-format access unit contains an IDR slice
    NAL (type 5). Ported verbatim from camera/stream/decode/main.py --
    pure byte scanning, no GStreamer dependency to begin with."""
    i = 0
    n = len(h264_bytes)
    while i < n - 3:
        if h264_bytes[i] == 0 and h264_bytes[i + 1] == 0:
            if h264_bytes[i + 2] == 1:
                start = i + 3
            elif i < n - 4 and h264_bytes[i + 2] == 0 and h264_bytes[i + 3] == 1:
                start = i + 4
            else:
                i += 1
                continue
            if start < n and (h264_bytes[start] & 0x1F) == 5:
                return True
            i = start
            continue
        i += 1
    return False


def nv12_to_bgr_numpy(y_view, uv_view):
    """TEMPORARY bridge for this commit's verification only -- converts a
    buffered (y_tensor, uv_tensor) pair back to a CPU BGR numpy array so it
    can run through the existing, unmodified detector.model.track() call
    and be compared against the old LocalSocketSource path. Not the
    production data path: Commit 3 does colour conversion on-GPU as part
    of the batched letterbox, never through a CPU numpy round-trip."""
    y = y_view.cpu().numpy()[:, :, 0]
    uv = uv_view.cpu().numpy()
    # (960, 1280, 2) interleaved U/V -> (960, 2560) raw bytes: merge the
    # pair-count and channel dims, not double the row count -- OpenCV wants
    # a flat (h/2, w) byte plane below the Y plane, matching the shape
    # DecodedFrame.shape itself reports for the whole NV12 buffer.
    uv_bytes = uv.reshape(uv.shape[0], uv.shape[1] * uv.shape[2])
    nv12 = np.concatenate([y, uv_bytes], axis=0)
    return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)


class GpuDecodeSource:
    """Same interface as LocalSocketSource (frame_sources.py) -- peek_newest
    / find_closest / discard_through / close -- so process_streams() can use
    either interchangeably based on config. Internally: decodes on GPU via
    PyNvVideoCodec instead of subscribing to already-decoded frames.

    Buffered items are ((y_tensor, uv_tensor), t_seconds), NV12, not the
    BGR numpy frames LocalSocketSource buffers -- see module docstring."""

    is_live = True

    def __init__(self, socket_path, t0, buffer_duration_sec, nominal_fps, target_fps, gpu_id=0):
        self._socket_path = socket_path
        self._t0 = t0
        self._gpu_id = gpu_id
        # Sized by target_fps, not nominal_fps like LocalSocketSource:
        # decimation happens before buffering here (post-decode, see module
        # docstring), so the buffer only ever holds target_fps-rate frames
        # to begin with -- sizing it any larger would just waste GPU memory.
        maxlen = max(1, int(buffer_duration_sec * target_fps))
        self._buffer = deque(maxlen=maxlen)  # [((y, uv), t_seconds), ...], oldest first
        self._lock = threading.Lock()
        self._stopped = False

        self.calibrator = ChannelClockCalibrator()
        self._unwrapped_ticks = None
        self._last_wire_ts = None

        # Decoded-frame decimation: keep a frame only once at least
        # 1/target_fps seconds (in RTP ticks) have passed since the last
        # kept one. Gates what gets *kept*, never what gets *decoded* --
        # see module docstring for why that distinction is load-bearing.
        self._min_gap_ticks = RTP_CLOCK_HZ / target_fps
        self._last_kept_ticks = None

        # HEALTH log counters (decoded vs. kept access units), reset each
        # log interval -- see _maybe_log_health.
        self._decoded_count = 0
        self._kept_count = 0
        self._last_health_log = time.monotonic()

        # Liveness guard on decoded PTS -- see _handle_access_unit. Not
        # known to be reachable given NVDEC echoes pkt.pts back per-frame
        # rather than deriving it from internal running state (unlike the
        # GStreamer bug this guards against, see project history), but a
        # stalled clock should be loud regardless of whether this specific
        # trigger can still happen after the fix below.
        self._last_output_ticks_seen = None
        self._stuck_output_count = 0

        self._decoder = self._create_decoder()

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _create_decoder(self):
        return nvc.CreateDecoder(
            gpuid=self._gpu_id,
            codec=nvc.cudaVideoCodec.H264,
            usedevicememory=1,
            # Minimize NVDEC's own internal reorder latency -- default
            # (NATIVE) buffers 4 frames for display-order reordering, which
            # this stream has no use for: it has no B-frames (confirmed via
            # ffprobe), so there's nothing to reorder, and a live pipeline
            # wants each decoded frame as soon as it's ready.
            latency=nvc.DisplayDecodeLatencyType.LOW,
        )

    def _recv_loop(self):
        while not self._stopped:
            try:
                client = BroadcastClient(self._socket_path)
            except (ConnectionError, OSError):
                time.sleep(1.0)
                continue
            try:
                for msg_type, payload in client:
                    if self._stopped:
                        break
                    if msg_type == MSG_SR_ANCHOR:
                        rtp_ts, ntp_time = unpack_sr_anchor(payload)
                        self.calibrator.update_anchor(rtp_ts, ntp_time)
                        continue
                    if msg_type != MSG_VIDEO:
                        continue
                    _pts_ns, raw_rtp_ts, h264_bytes = unpack_video(payload)
                    self._handle_access_unit(raw_rtp_ts, h264_bytes)
            except ConnectionError:
                pass
            client.close()
            if not self._stopped:
                time.sleep(1.0)  # demux restarted or briefly gone -- retry

    def _handle_access_unit(self, raw_rtp_ts, h264_bytes):
        # Tick unwrapping -- identical logic to decode/main.py's
        # push_video(); see there for the full rationale. The unwrapped
        # value is what gets fed to the decoder as pkt.pts (NVDEC echoes it
        # back on the matching output frame via getPTS(), so there's no
        # separate ns<->ticks conversion needed the way GStreamer's buffer
        # PTS required) and what decimation paces against.
        if self._last_wire_ts is None:
            self._unwrapped_ticks = raw_rtp_ts
        else:
            delta = wrapped_diff(raw_rtp_ts, self._last_wire_ts)
            if abs(delta) > MAX_PLAUSIBLE_TICK_JUMP:
                print(
                    f"gpu_decode {self._socket_path}: raw_rtp_ts jumped by {delta} ticks -- "
                    "treating as a new RTP session: flushing decoder and resetting calibrator"
                )
                # A large PTS discontinuity is a real production incident
                # in camera/stream/decode/main.py, root-caused during this
                # commit's work: it fed the new session's ticks straight
                # into a still-running decoder, which lost input/output PTS
                # association after the jump and silently kept publishing
                # frames with a frozen, stale timestamp -- undetected for
                # ~19 hours on the live pipeline because nothing there
                # checked for a clock that stopped advancing, only for one
                # running fast. Not repeatable here even in principle: the
                # decoder is recreated outright rather than patched (this
                # API exposes no flush/reset call), so there's no live
                # session left to lose PTS association with, and the
                # calibrator is reset too -- its anchor belongs to the old
                # session's numeric tick range and would compute nonsense
                # abs_time against the new one otherwise. Recreating a
                # decoder has real cost, but session boundaries are rare
                # (demux restarts), not a per-frame path.
                #
                # One measured consequence worth knowing about: a freshly
                # created decoder produces no output at all for its first
                # ~48 access units (~1.6s at 30fps) -- confirmed directly
                # by feeding a real captured sequence through a fresh
                # decoder and diffing output ticks against a parallel
                # avdec_h264 decode of the same input; the mismatched
                # frames were the first ~48 by input order, nothing
                # scattered through the middle. Not a bug, just decoder
                # session warm-up latency (distinct from the LOW
                # DisplayDecodeLatencyType setting above, which governs
                # reorder latency, not initial session setup) -- but it
                # means every decoder recreation, i.e. every session
                # boundary, opens a ~1.6s blind window on this channel
                # with zero frames buffered, not just zero *new* ones.
                self._decoder = self._create_decoder()
                self.calibrator = ChannelClockCalibrator()
                self._last_output_ticks_seen = None
                self._stuck_output_count = 0
                self._unwrapped_ticks = raw_rtp_ts
            else:
                self._unwrapped_ticks += delta
        self._last_wire_ts = raw_rtp_ts

        frames = self._decode(h264_bytes, self._unwrapped_ticks)
        self._decoded_count += 1

        for frame in frames:
            output_ticks = frame.getPTS()

            # Liveness guard: a decoded PTS that stops advancing is not a
            # value to quietly keep using -- see the session-jump comment
            # above for the incident this guards against. STUCK_OUTPUT_
            # THRESHOLD frames (~1s at 30fps) of an unchanging PTS while
            # decode is otherwise running normally means something is
            # producing frames with a dead clock; drop them loudly rather
            # than buffer them with timestamps no consumer should trust.
            if output_ticks == self._last_output_ticks_seen:
                self._stuck_output_count += 1
            else:
                self._stuck_output_count = 0
            self._last_output_ticks_seen = output_ticks
            if self._stuck_output_count >= STUCK_OUTPUT_THRESHOLD:
                if self._stuck_output_count == STUCK_OUTPUT_THRESHOLD:
                    print(
                        f"gpu_decode {self._socket_path}: ERROR decoded PTS has not "
                        f"advanced in {self._stuck_output_count} frames (stuck at "
                        f"{output_ticks}) -- dropping frames rather than publishing "
                        "them with a dead clock"
                    )
                continue

            keep = (
                self._last_kept_ticks is None
                or (output_ticks - self._last_kept_ticks) >= self._min_gap_ticks
            )
            if not keep:
                continue

            abs_time = self.calibrator.to_abs_time(output_ticks)
            if abs_time is None:
                continue  # no RTCP SR anchor yet -- not calibrated, not usable

            self._last_kept_ticks = output_ticks
            self._kept_count += 1

            # MUST clone: PyNvVideoCodec recycles its surface pool, so the
            # views returned by frame.cuda() alias memory the next
            # Decode() call may overwrite. Kept frames sit in the buffer
            # for up to buffer_duration_sec -- well past "the next call" --
            # confirmed necessary directly, not a defensive guess.
            y_view, uv_view = frame.cuda()
            y_tensor = torch.as_tensor(y_view, device=f"cuda:{self._gpu_id}").clone()
            uv_tensor = torch.as_tensor(uv_view, device=f"cuda:{self._gpu_id}").clone()

            t = abs_time - self._t0
            with self._lock:
                self._buffer.append(((y_tensor, uv_tensor), t))

        self._maybe_log_health()

    def _decode(self, h264_bytes, pts_ticks):
        # bsl_data must be a raw pointer that stays valid for the duration
        # of Decode() -- ctypes.addressof() on a temporary is a use-after-
        # free (the buffer can be collected before native code reads it,
        # works fine in light testing, corrupts frames under load).
        # create_string_buffer copies h264_bytes into a ctypes-owned
        # buffer that this local variable (`buf`) keeps alive across the
        # call.
        buf = ctypes.create_string_buffer(h264_bytes, len(h264_bytes))
        pkt = nvc.PacketData()
        pkt.bsl_data = ctypes.addressof(buf)
        pkt.bsl = len(h264_bytes)
        pkt.pts = pts_ticks
        pkt.key = _contains_idr(h264_bytes)
        return self._decoder.Decode(pkt)

    def _maybe_log_health(self):
        now = time.monotonic()
        if now - self._last_health_log < 15.0:
            return
        self._last_health_log = now
        with self._lock:
            depth = len(self._buffer)
            newest_t = self._buffer[-1][1] if self._buffer else None
        lag_str = f"{(time.time() - self._t0) - newest_t:.2f}s" if newest_t is not None else "n/a"
        print(
            f"gpu_decode {self._socket_path}: [HEALTH] lag={lag_str} buffer={depth} "
            f"decoded={self._decoded_count} kept={self._kept_count} "
            f"anchors={self.calibrator.anchor_count}"
        )
        self._decoded_count = 0
        self._kept_count = 0

    def peek_newest(self):
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def find_closest(self, target_t, tolerance):
        with self._lock:
            best_idx = None
            best_diff = None
            best_item = None
            for idx, (frame, t) in enumerate(self._buffer):
                diff = abs(t - target_t)
                if best_diff is None or diff < best_diff:
                    best_diff, best_idx, best_item = diff, idx, (frame, t)
                if t > target_t:
                    break
            if best_diff is None or best_diff > tolerance:
                return None
            return best_idx, best_item[0], best_item[1]

    def discard_through(self, index):
        with self._lock:
            for _ in range(min(index + 1, len(self._buffer))):
                self._buffer.popleft()

    def close(self):
        self._stopped = True
