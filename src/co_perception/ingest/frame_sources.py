"""Uniform frame-source abstraction so MultiCameraPipeline.process_streams()
can poll a local camera-decode broadcast socket, a local video file, or an
AWS KVS HLS pull through the same interface. config.py's ingestion.mode
selects which one gets built for each channel -- see build_frame_source().
"""
from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np

from co_perception.common.broadcast import BroadcastClient
from co_perception.ingest import kinesis_utils
from co_perception.ingest.decode_protocol import MSG_FRAME, unpack_frame

# TEMPORARY, decode-merge cost investigation -- remove after reporting.
_SOCKET_READ_MS = []

# How far a frame's abs_time may disagree with wall-clock before it's
# rejected outright rather than buffered. Generous relative to normal
# operation (steady-state lag is well under a second, decode's own
# worst-case GOP-drop bounds it to ~3s) -- this exists to catch a decode
# process broadcasting a badly wrong timestamp (confirmed happening in
# production: a session-tracking bug left one channel's frames tagged
# hours in the future), not to police ordinary jitter. Silently buffering
# a value like that doesn't just mis-time one frame -- process_streams'
# cross-channel sync requires all channels to agree within a tight
# tolerance before processing anything, so one bad channel stalls every
# channel's output, not just its own.
MAX_ABS_TIME_SKEW_SEC = 60.0


class FrameSource:
    is_live = False  # True means "no new frame yet" != "stream ended"

    def read(self):
        """Returns (ok, frame, msec). msec must be directly comparable
        across every source used together in one pipeline run -- see
        process_streams' global_msec synchronization."""
        raise NotImplementedError

    def to_abs_time(self, timestamp):
        """Convert this source's relative timestamp to epoch seconds."""
        raise NotImplementedError

    def close(self):
        pass


class VideoCaptureSource(FrameSource):
    """Wraps cv2.VideoCapture for both local files and AWS KVS HLS pulls.
    !ret means genuine end-of-stream (or, for kvs, one reconnect attempt
    also failed) -- matches this project's original file/KVS behavior."""

    def __init__(self, path, kvs_stream_name=None):
        self._kvs_stream_name = kvs_stream_name
        self._cap = cv2.VideoCapture(path)

    @property
    def fps(self):
        return self._cap.get(cv2.CAP_PROP_FPS)

    def read(self):
        ret, frame = self._cap.read()
        if not ret and self._kvs_stream_name:
            url = kinesis_utils.get_kvs_hls_url(self._kvs_stream_name)
            self._cap = cv2.VideoCapture(url)
            ret, frame = self._cap.read()
        if not ret:
            return False, None, -1.0
        return True, frame, self._cap.get(cv2.CAP_PROP_POS_MSEC)

    def close(self):
        self._cap.release()


class LocalSocketSource(FrameSource):
    """Subscribes to camera/stream's decode-stage broadcast (already-decoded
    frames + calibrated abs_time, see decode_protocol.py) on a background
    thread, buffering the last `buffer_duration_sec` worth of frames in
    arrival order.

    This is deliberately a bounded FIFO, not a single "latest" slot: proper
    cross-channel synchronization needs to compare each channel's *oldest
    not-yet-used* frame against the others (see process_streams' live
    branch), which means a channel that's running ahead has to hold onto
    its older frames until a lagging channel catches up to them -- a
    single-slot design throws those away the instant a newer frame arrives,
    which is exactly what made real alignment impossible before this.

    The bound (`maxlen`, computed from buffer_duration_sec * nominal_fps)
    keeps memory bounded rather than growing forever if a channel is never
    consumed from -- once full, each new arrival evicts the channel's own
    oldest frame automatically (collections.deque's standard maxlen
    behavior), so a channel that's been waiting a long time naturally has
    its usable history "walk forward" rather than accumulating without
    limit. Sized to comfortably exceed decode's own worst-case backlog
    (bounded to ~3s by its GOP-aware drop, see camera/stream/decode/main.py)
    -- see config.py's ingestion.sync_buffer_seconds for the actual value
    and its memory-cost tradeoff.
    """

    is_live = True

    def __init__(self, socket_path, t0, buffer_duration_sec, nominal_fps):
        self._socket_path = socket_path
        self._t0 = t0
        maxlen = max(1, int(buffer_duration_sec * nominal_fps))
        self._buffer = deque(maxlen=maxlen)  # [(frame, t_seconds), ...], oldest first
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        while not self._stopped:
            try:
                client = BroadcastClient(self._socket_path)
            except (ConnectionError, OSError):
                time.sleep(1.0)
                continue
            try:
                t_prev = time.monotonic()  # TEMPORARY, see _SOCKET_READ_MS above
                for msg_type, payload in client:
                    if self._stopped:
                        break
                    if msg_type != MSG_FRAME:
                        continue
                    if "ch0" in self._socket_path and len(_SOCKET_READ_MS) < 500:
                        _SOCKET_READ_MS.append((time.monotonic() - t_prev) * 1000)
                        if len(_SOCKET_READ_MS) == 500:
                            s = sorted(_SOCKET_READ_MS)
                            print(f"[TIMING] socket_read median={s[250]:.2f}ms p90={s[450]:.2f}ms over 500 samples")
                    t_prev = time.monotonic()
                    height, width, channels, abs_time, _raw_rtp_ts, pixels = unpack_frame(payload)
                    if abs_time is None:
                        continue  # no RTCP SR anchor yet -- not calibrated, not usable
                    skew = abs_time - time.time()
                    if abs(skew) > MAX_ABS_TIME_SKEW_SEC:
                        print(
                            f"LocalSocketSource {self._socket_path}: REJECTED frame, "
                            f"abs_time is {skew:+.1f}s from wall clock (limit "
                            f"{MAX_ABS_TIME_SKEW_SEC}s) -- decode is broadcasting a bad "
                            "timestamp; dropping rather than stalling the sync loop"
                        )
                        continue
                    frame = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, channels)
                    t = abs_time - self._t0
                    with self._lock:
                        self._buffer.append((frame, t))

            except ConnectionError:
                pass
            client.close()
            if not self._stopped:
                time.sleep(1.0)  # decode restarted or briefly gone -- retry

    def to_abs_time(self, timestamp):
        return self._t0 + timestamp

    def peek_newest(self):
        """Returns (frame, t_seconds) for the most recently arrived frame,
        without removing it, or None if nothing is buffered yet. Used to
        find how far this channel has progressed *right now* -- unlike the
        oldest/front item, this always advances as new frames arrive,
        regardless of whether anything has been consumed. See
        find_closest()'s docstring for why that distinction matters."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def find_closest(self, target_t, tolerance):
        """Non-destructive: returns (index, frame, t) for the buffered
        frame closest to target_t, or None if nothing is within tolerance.
        Does not modify the buffer -- see discard_older_than() for the
        follow-up commit step, kept separate so a caller comparing multiple
        channels can confirm *all* of them have a match before discarding
        anything from *any* of them (a match on channel 0 shouldn't mutate
        its buffer if channel 2 turns out to have no match at all).

        Deliberately keyed off peek_newest() rather than the oldest/front
        item when picking target_t upstream: if every channel's front were
        used as the sync signal instead, a channel would only ever advance
        past a misaligned front once it got explicitly consumed -- but nothing
        gets consumed until alignment is found, which is a deadlock the
        instant multiple channels have a persistent (not transient) offset
        from each other, confirmed directly against the real pipeline: with
        4 channels each independently 1-2s off from the others at all
        times (not one channel briefly stalling while the rest agree), no
        front ever moved and nothing was ever processed. The newest item
        always progresses on its own as frames keep arriving, which is what
        target_t needs to do too."""
        with self._lock:
            best_idx = None
            best_diff = None
            best_item = None
            for idx, (frame, t) in enumerate(self._buffer):
                diff = abs(t - target_t)
                if best_diff is None or diff < best_diff:
                    best_diff, best_idx, best_item = diff, idx, (frame, t)
                if t > target_t:
                    # Buffer is time-ordered oldest-first; once entries pass
                    # target_t, later ones can only be farther away.
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

    def close(self):
        self._stopped = True


def build_frame_source(channel_cfg, mode, t0, nominal_fps=30.0, sync_buffer_seconds=8.0,
                        target_fps=None, max_buffer_ahead_sec=3.0, imgsz=None, letterbox_resolver=None):
    if mode == "local_socket":
        return LocalSocketSource(channel_cfg.socket_path, t0, sync_buffer_seconds, nominal_fps)
    if mode == "gpu_decode":
        from co_perception.ingest.gpu_decode_source import GpuDecodeSource

        # channel_cfg.socket_path here points at demux's broadcast
        # (compressed H.264), not decode's -- see gpu_decode_source.py and
        # config.py's validation for this mode. max_buffer_ahead_sec, not
        # sync_buffer_seconds/target_fps: GpuDecodeSource buffers every
        # decoded frame now (no decimation), sized as an OOM guard against
        # a channel getting ahead of consumption -- see its own docstring.
        # imgsz/letterbox_resolver: gpu_decode letterboxes at buffer time
        # now, see gpu_decode_source.py's module docstring.
        return GpuDecodeSource(
            channel_cfg.socket_path, t0, max_buffer_ahead_sec, nominal_fps, imgsz, letterbox_resolver
        )
    if mode == "local_file":
        return VideoCaptureSource(channel_cfg.file_path)
    if mode == "aws_kvs":
        url = kinesis_utils.get_kvs_hls_url(channel_cfg.kvs_stream_name)
        return VideoCaptureSource(url, kvs_stream_name=channel_cfg.kvs_stream_name)
    raise ValueError(f"unknown ingestion mode: {mode!r}")
