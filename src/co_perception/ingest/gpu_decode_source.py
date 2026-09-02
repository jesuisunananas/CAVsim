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

No decimation anywhere in this class, before or after decode. Every H.264
access unit gets decoded -- dropping any of them before feeding the
decoder corrupts every subsequent P-frame in the same GOP, since this
camera's stream is a standard IPPP chain (confirmed via ffprobe: one IDR
then 60 consecutive P-frames, no B-frames) where each P-frame references
the immediately preceding decoded frame, not just the last IDR (NVDEC
conceals this rather than erroring -- silently corrupted/smeared output
that still "decodes successfully"). And every decoded frame gets
buffered, keyed by abs_time: picking a target_fps-rate subset here was
tried and abandoned -- it meant this class's own idea of "the right frame
to keep" could disagree with what the sync loop's real-time basis
actually wants a tick later, with no way to reconcile after the fact.
Downsampling now happens exactly once, in process_video.py's sync loop,
by selecting the buffered frame closest to each tick's basis -- see
its docstring.

Buffered frames are letterboxed to the inference target shape (uint8)
immediately after decode, not the raw full-resolution decode output --
see gpu_image_ops.gpu_letterbox_to_uint8. This is a real memory decision,
not just convenience: it's what makes a wider cross-channel sync margin
(process_video.py's sync_margin_sec) affordable -- a 1-second buffer of
full-res frames costs ~442MB/channel, the same second at letterboxed
uint8 costs ~111MB/channel.

Buffered frame representation: planar RGB (outputColorType=RGBP), not NV12.
Colour conversion (BT.601/BT.709, limited/full range) happens inside
NVIDIA's decoder, not in this codebase -- the entire silent-corruption risk
of hand-rolling YUV->RGB math is avoided by construction. Confirmed
empirically, not assumed from the docs (the documented example uses
ThreadedDecoder over a file path; this module feeds packets one access
unit at a time via the low-level CreateDecoder/Decode path instead, since
frames arrive over a socket, not from a file):
- outputColorType=RGBP is accepted by CreateDecoder on this packet-fed
  path (the docs only show it on ThreadedDecoder).
- DecodedFrame.cuda() returns THREE separate CUDA Array Interface views in
  RGBP mode, one per plane (R, G, B), each a plain (H, W) uint8 view --
  not one fused (3, H, W) view. Stacked into one tensor at buffer time
  (torch.stack, dim=0) so buffered items are a single (3, H, W) CUDA
  tensor, not a tuple.
- Plane order is R, G, B (not reversed) -- verified by decoding a real
  captured frame both ways (RGBP and the old NATIVE/NV12 mode) from
  identical input access units, converting the NV12 copy to BGR via
  OpenCV as a known-correct reference, and confirming the RGBP planes
  stacked as (H, W, 3) and reversed to BGR visually match (natural colours
  -- yellow lane paint, tan dry grass, gray asphalt -- not a channel swap
  or inversion).
"""
from __future__ import annotations

import ctypes
import threading
import time
from collections import deque

import torch
import PyNvVideoCodec as nvc

from co_perception.common.broadcast import BroadcastClient
from co_perception.ingest.demux_protocol import (
    MSG_SESSION_RESET,
    MSG_SR_ANCHOR,
    MSG_VIDEO,
    unpack_session_reset,
    unpack_sr_anchor,
    unpack_video,
)
from co_perception.ingest.frame_sources import MAX_ABS_TIME_SKEW_SEC
from co_perception.ingest.gpu_image_ops import gpu_letterbox_to_uint8
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


class GpuDecodeSource:
    """Same interface as LocalSocketSource (frame_sources.py) -- peek_newest
    / find_closest / discard_older_than / close -- so process_streams() can
    use either interchangeably based on config. Internally: decodes on GPU
    via PyNvVideoCodec instead of subscribing to already-decoded frames.

    Buffered items are (letterboxed_tensor, t_seconds) -- letterboxed_tensor
    a single (3, H, W) CUDA uint8 tensor already resized+padded to the
    inference target shape, not the raw full-resolution decode output, not
    the BGR numpy frames LocalSocketSource buffers, and not a (y, uv) tuple
    either -- see module docstring."""

    is_live = True

    def __init__(self, socket_path, t0, max_buffer_ahead_sec, nominal_fps, imgsz, letterbox_resolver, gpu_id=0):
        self._socket_path = socket_path
        self._t0 = t0
        self._gpu_id = gpu_id
        self._imgsz = imgsz
        self._letterbox_resolver = letterbox_resolver
        # (r, pad_left, pad_top, content_h, content_w) -- constant for this
        # channel's lifetime (its native resolution doesn't change),
        # cached on first frame so process_video.py can read it once
        # instead of recomputing every tick. See letterbox_params property.
        self._letterbox_params = None
        # No decimation on the way in any more -- every decoded frame is
        # buffered, keyed by abs_time (see module docstring and
        # process_video.py's real-time-basis sync loop, which does its own
        # closest-match selection against a fixed clock rather than
        # relying on a pre-decimated stream). maxlen is purely an OOM
        # guard against a channel getting ahead of consumption (the sync
        # loop prunes every tick in steady state, so this is a ceiling,
        # not the expected working size) -- explicitly a config knob
        # (ingestion.max_buffer_ahead_sec), not tuned here.
        maxlen = max(1, int(max_buffer_ahead_sec * nominal_fps))
        self._buffer = deque(maxlen=maxlen)  # [(letterboxed_tensor, t_seconds), ...], oldest first
        self._lock = threading.Lock()
        self._stopped = False

        self.calibrator = ChannelClockCalibrator()
        self._unwrapped_ticks = None
        self._last_wire_ts = None

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

        # Set by handle_session_reset(), cleared once the first IDR after a
        # reset is seen. Feeding a fresh decoder (no reference frames) a
        # non-IDR access unit is invalid input, not just suboptimal -- see
        # handle_session_reset() for why the decoder is fresh after a reset.
        self._waiting_for_idr = False

        self._decoder = self._create_decoder()

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _create_decoder(self):
        return nvc.CreateDecoder(
            gpuid=self._gpu_id,
            codec=nvc.cudaVideoCodec.H264,
            usedevicememory=1,
            # Planar RGB, decoded on-GPU by NVIDIA's own colour-conversion
            # code rather than hand-rolled YUV math downstream -- see module
            # docstring for the empirical checks this was verified with
            # (accepted on this packet-fed path, 3 separate (H,W) plane
            # views in R,G,B order).
            outputColorType=nvc.OutputColorType.RGBP,
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
                    if msg_type == MSG_SESSION_RESET:
                        new_raw_rtp_ts = unpack_session_reset(payload)
                        self.handle_session_reset(new_raw_rtp_ts)
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

    def handle_session_reset(self, new_raw_rtp_ts):
        """Demux has reconnected to the camera (RTSP session boundary) --
        an authoritative signal, not a guess. Ordering: demux emits
        MSG_SESSION_RESET from a pad probe on depay's sink pad, upstream of
        appsink, so it always precedes the new session's first MSG_VIDEO on
        the wire -- see demux/main.py's _on_rtp_buffer.

        Unlike decode/main.py's handle_session_reset(), no stale-frame
        session tagging is needed here: _handle_access_unit() runs
        synchronously start-to-finish on this object's single receiver
        thread (Decode() hands back whatever frames are ready in the same
        call that fed it -- no separate GStreamer-style streaming thread
        with its own internal buffering), so there is no async pipeline
        stage that could still be draining the old session's frames by the
        time this method returns. Recreating self._decoder below is
        therefore a hard cutover, not a race: once it happens, no later
        call can produce old-session output.

        Old reference frames must not decode new-session frames, and this
        API exposes no flush/reset call, so recreation is the only way to
        guarantee clean decoder state. This has a real, measured cost: a
        freshly created decoder produces no output at all for its first
        ~48 access units (~1.6s at 30fps) -- confirmed by feeding a
        captured sequence through a fresh decoder and diffing output ticks
        against a parallel avdec_h264 decode of the same input. Combined
        with waiting for the next IDR (below, in _handle_access_unit),
        every session boundary opens a blind window on this channel of at
        least that long -- expected to be routine now (up to one GOP,
        ~2s, per reset), not just an outage-time cost.
        """
        print(
            f"gpu_decode {self._socket_path}: session reset (demux reconnected) "
            f"-- new origin {new_raw_rtp_ts}"
        )
        self._unwrapped_ticks = new_raw_rtp_ts
        self._last_wire_ts = new_raw_rtp_ts
        self.calibrator = ChannelClockCalibrator()
        self._last_output_ticks_seen = None
        self._stuck_output_count = 0
        # Everything buffered belongs to the old session -- discarding it
        # is correct, not a loss: an RTSP reset is a genuine network
        # discontinuity, so there's no continuity across it worth
        # preserving (see module history).
        with self._lock:
            self._buffer.clear()
        self._decoder = self._create_decoder()
        self._waiting_for_idr = True

    def _handle_access_unit(self, raw_rtp_ts, h264_bytes):
        # Tick unwrapping -- identical logic to decode/main.py's
        # push_video(); see there for the full rationale. The unwrapped
        # value is what gets fed to the decoder as pkt.pts (NVDEC echoes it
        # back on the matching output frame via getPTS(), so there's no
        # separate ns<->ticks conversion needed the way GStreamer's buffer
        # PTS required).
        if self._last_wire_ts is None:
            self._unwrapped_ticks = raw_rtp_ts
        else:
            delta = wrapped_diff(raw_rtp_ts, self._last_wire_ts)
            if abs(delta) > MAX_PLAUSIBLE_TICK_JUMP:
                # MONITOR ONLY -- must not act on this. Session boundaries
                # are decided exclusively by handle_session_reset(), driven
                # by demux's authoritative MSG_SESSION_RESET, not by
                # guessing from tick deltas (a threshold guess is wrong in
                # both directions: a genuine long stall reads as a
                # reconnect, and a reconnect whose new random origin lands
                # close to the old one is missed). A jump with no preceding
                # reset message means demux missed signaling a reconnect,
                # or this process's socket dropped the message.
                print(
                    f"gpu_decode {self._socket_path}: WARNING raw_rtp_ts jumped by {delta} "
                    "ticks with no preceding MSG_SESSION_RESET -- demux may have missed "
                    "signaling a reconnect, or this process's socket dropped the reset message"
                )
            self._unwrapped_ticks += delta
        self._last_wire_ts = raw_rtp_ts

        if self._waiting_for_idr:
            if not _contains_idr(h264_bytes):
                return
            self._waiting_for_idr = False

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

            abs_time = self.calibrator.to_abs_time(output_ticks)
            if abs_time is None:
                continue  # no RTCP SR anchor yet -- not calibrated, not usable

            # Same guard as LocalSocketSource (frame_sources.py) -- and the
            # same reason it exists there: a frame with a badly wrong
            # abs_time doesn't just mis-time itself, it can stall every
            # channel's output, since process_streams' cross-channel sync
            # requires all channels to agree within a tight tolerance.
            skew = abs_time - time.time()
            if abs(skew) > MAX_ABS_TIME_SKEW_SEC:
                print(
                    f"gpu_decode {self._socket_path}: REJECTED frame, abs_time is "
                    f"{skew:+.1f}s from wall clock (limit {MAX_ABS_TIME_SKEW_SEC}s) "
                    "-- dropping rather than stalling the sync loop"
                )
                continue

            self._kept_count += 1

            # MUST clone: PyNvVideoCodec recycles its surface pool, so the
            # views returned by frame.cuda() alias memory the next
            # Decode() call may overwrite. Kept frames sit in the buffer
            # for up to the configured max_buffer_ahead_sec -- well past
            # "the next call" --
            # confirmed necessary directly, not a defensive guess.
            r_view, g_view, b_view = frame.cuda()
            device = f"cuda:{self._gpu_id}"
            rgb_tensor = torch.stack(
                [torch.as_tensor(v, device=device) for v in (r_view, g_view, b_view)],
                dim=0,
            ).clone()  # (3, H, W) uint8, full resolution

            # Letterbox to the inference target shape immediately, before
            # buffering -- not the raw full-res frame. This is what makes
            # a wider sync_margin_sec buffer affordable (~4x smaller per
            # frame; see module docstring). Must return uint8, not the
            # float32-in-[0,1] scripts/process_video.py's own _gpu_letterbox
            # returns -- that would cost the same bytes as full-res and
            # defeat the point (see gpu_letterbox_to_uint8's docstring).
            target_shape = self._letterbox_resolver.resolve(rgb_tensor.shape[1], rgb_tensor.shape[2])
            letterboxed, r, pad_left, pad_top, content_h, content_w = gpu_letterbox_to_uint8(rgb_tensor, target_shape)
            if self._letterbox_params is None:
                self._letterbox_params = (r, pad_left, pad_top, content_h, content_w)

            t = abs_time - self._t0
            with self._lock:
                self._buffer.append((letterboxed, t))

        self._maybe_log_health()

    def to_abs_time(self, timestamp):
        return self._t0 + timestamp

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

    def discard_older_than(self, threshold_t):
        """Drop buffered frames with t < threshold_t, keeping anything at
        or after it -- including the frame just matched this tick, which
        the sync loop's basis keeps by design (see process_video.py:
        pruning to basis - 1/target_fps, not through the match itself, in
        case a frame just before that point turns out to be the *next*
        tick's closest match)."""
        with self._lock:
            while self._buffer and self._buffer[0][1] < threshold_t:
                self._buffer.popleft()

    @property
    def letterbox_params(self):
        """(r, pad_left, pad_top, content_h, content_w) from this channel's
        letterbox transform, or None if no frame has been decoded yet.
        Constant for the channel's lifetime -- process_video.py reads this
        once instead of recomputing it every tick."""
        return self._letterbox_params

    def close(self):
        self._stopped = True
