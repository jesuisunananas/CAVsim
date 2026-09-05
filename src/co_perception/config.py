"""Loader for the per-machine pipeline config (config/pipeline.yaml).

Exists so the same process_video.py runs unmodified on different machines --
what used to be literal values in its __main__ block (which video source,
which model/camera calibration, where results go) now lives in a YAML file
instead, one per deployment target.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

VALID_INGESTION_MODES = {"local_socket", "local_file", "aws_kvs", "gpu_decode"}


@dataclass
class ChannelSource:
    channel: int
    socket_path: str | None = None
    file_path: str | None = None
    kvs_stream_name: str | None = None


@dataclass
class CameraConfig:
    channel: int
    device_id: str
    K: list[list[float]]
    camera_height: float
    pitch_deg: float
    yaw_deg: float
    heading_deg: float
    origin_lat: float
    origin_lon: float
    city: str = ""
    state: str = ""
    country: str = ""


@dataclass
class SaveOutputConfig:
    enabled: bool = True
    json_path: str | None = None
    video_path: str | None = None


@dataclass
class UploadOutputConfig:
    enabled: bool = False
    endpoint: str | None = None


@dataclass
class BroadcastOutputConfig:
    enabled: bool = False
    socket_path: str | None = None


@dataclass
class HttpOutputConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8091

@dataclass
class PipelineConfig:
    repo_root: Path
    show_live: bool
    ingestion_mode: str
    channels: list[ChannelSource]
    nominal_fps: float
    target_fps: float
    sync_buffer_seconds: float
    max_buffer_ahead_sec: float
    max_consecutive_skips: int
    sync_margin_sec: float
    model_path: str
    conf: float
    imgsz: int
    cameras: list[CameraConfig]
    save: SaveOutputConfig
    upload: UploadOutputConfig
    broadcast: BroadcastOutputConfig
    http: HttpOutputConfig

    def resolve(self, path: str | None) -> str | None:
        """Resolve a config-relative path against the repo root, so the
        config file (and the paths inside it) work regardless of CWD."""
        if path is None:
            return None
        p = Path(path)
        return str(p if p.is_absolute() else self.repo_root / p)


def load_config(config_path: str | Path, repo_root: str | Path) -> PipelineConfig:
    repo_root = Path(repo_root)
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    ingestion = raw["ingestion"]
    mode = ingestion["mode"]
    if mode not in VALID_INGESTION_MODES:
        raise ValueError(f"ingestion.mode must be one of {VALID_INGESTION_MODES}, got {mode!r}")

    channels = [ChannelSource(**c) for c in ingestion["channels"]]
    for c in channels:
        if mode == "local_socket" and not c.socket_path:
            raise ValueError(f"ingestion.mode=local_socket but channel {c.channel} has no socket_path")
        if mode == "gpu_decode" and not c.socket_path:
            # Same field as local_socket, different meaning: this one
            # points at camera/stream's demux broadcast (compressed H.264),
            # not decode's (already-decoded frames) -- see
            # ingest/gpu_decode_source.py.
            raise ValueError(f"ingestion.mode=gpu_decode but channel {c.channel} has no socket_path")
        if mode == "local_file" and not c.file_path:
            raise ValueError(f"ingestion.mode=local_file but channel {c.channel} has no file_path")
        if mode == "aws_kvs" and not c.kvs_stream_name:
            raise ValueError(f"ingestion.mode=aws_kvs but channel {c.channel} has no kvs_stream_name")

    # nominal_fps is the camera's real source rate (used to size the
    # cross-channel timestamp-alignment tolerance for live sources --
    # see process_video.py's process_streams). target_fps is how often to
    # actually run detection; downsampling happens after decode, never
    # before it (pre-decode frame-skipping would break H.264 P/B-frame
    # references -- see project history), and can never exceed the real
    # source rate.
    nominal_fps = float(ingestion.get("nominal_fps", 30.0))
    target_fps = float(ingestion.get("target_fps", nominal_fps))
    if nominal_fps <= 0:
        raise ValueError(f"ingestion.nominal_fps must be positive, got {nominal_fps}")
    if target_fps <= 0:
        raise ValueError(f"ingestion.target_fps must be positive, got {target_fps}")
    if target_fps > nominal_fps:
        raise ValueError(
            f"ingestion.target_fps ({target_fps}) cannot exceed ingestion.nominal_fps ({nominal_fps})"
        )

    # How much per-channel history to buffer for cross-channel alignment
    # (see ingest/frame_sources.py's LocalSocketSource) -- sized in seconds
    # rather than frame count so it scales sensibly if nominal_fps changes.
    # Memory cost is real and roughly linear: at this camera's resolution
    # (~14MB/frame), 8s * 30fps * 4 channels is on the order of 13GB
    # worst-case (buffers this full only happens under sustained
    # misalignment, not steady-state operation).
    sync_buffer_seconds = float(ingestion.get("sync_buffer_seconds", 8.0))
    if sync_buffer_seconds <= 0:
        raise ValueError(f"ingestion.sync_buffer_seconds must be positive, got {sync_buffer_seconds}")

    # gpu_decode only: GpuDecodeSource buffers every decoded frame (no
    # decimation -- see its own docstring), so this caps how far a channel
    # may get ahead of consumption before frames start being evicted. An
    # OOM guard, not a tuning knob -- the sync loop prunes every tick in
    # steady state, so actual usage should stay far below this ceiling.
    # Frames are letterboxed to the inference resolution before buffering
    # (~3.69MB/frame at imgsz=1280 on this camera's resolution, not the
    # ~14.7MB/frame full-res figure this comment used to cite) -- worst
    # case is ~1.3GB across 4 channels at the default below, not ~5.3GB.
    max_buffer_ahead_sec = float(ingestion.get("max_buffer_ahead_sec", 3.0))
    if max_buffer_ahead_sec <= 0:
        raise ValueError(f"ingestion.max_buffer_ahead_sec must be positive, got {max_buffer_ahead_sec}")

    # How many consecutive ticks one channel may miss before the sync loop
    # treats it as needing an explicit reset (halt, wait for all four
    # stable again, re-lock the basis) rather than just skipping ticks --
    # see process_video.py's real-time-basis sync loop.
    max_consecutive_skips = int(ingestion.get("max_consecutive_skips", 5))
    if max_consecutive_skips <= 0:
        raise ValueError(f"ingestion.max_consecutive_skips must be positive, got {max_consecutive_skips}")

    # Live sources only: how far basis_0 is locked BEHIND the freshest
    # jointly-available timestamp at (re)lock time -- a jitter-buffer
    # margin trading added end-to-end latency for tolerance of delivery
    # bursts (see process_video.py's wait_for_stable_and_lock_basis).
    # Replaces what used to be a hardcoded 1/target_fps (100ms), which
    # doesn't survive real delivery bursts -- measured directly this
    # session: one channel showed recurring ~240-290ms delivery gaps every
    # ~2s despite perfectly smooth 33ms-spaced content timestamps
    # underneath (a delivery-side stall-then-burst, not real frame drops
    # or a timestamp problem). 1.0s was chosen as >3x that worst case.
    # Must be comfortably less than max_buffer_ahead_sec -- a margin wider
    # than the buffer's own retention window asks for history that's
    # already been evicted.
    sync_margin_sec = float(ingestion.get("sync_margin_sec", 1.0))
    if sync_margin_sec <= 0:
        raise ValueError(f"ingestion.sync_margin_sec must be positive, got {sync_margin_sec}")
    if sync_margin_sec >= max_buffer_ahead_sec:
        raise ValueError(
            f"ingestion.sync_margin_sec ({sync_margin_sec}) must be less than "
            f"ingestion.max_buffer_ahead_sec ({max_buffer_ahead_sec}) -- a margin this "
            "wide needs history the buffer wouldn't retain"
        )

    detection = raw["detection"]
    cameras = [CameraConfig(**cam) for cam in detection["cameras"]]

    # Side model.track() letterboxes the longest edge to. Defaults to
    # Ultralytics' own default (640) if unset, matching this repo's
    # previous unconfigured behaviour -- see process_video.py for why
    # that default is wrong for this camera's resolution. Must be a
    # multiple of the model stride (32 for YOLOv8); Ultralytics silently
    # rounds up otherwise, which would make the config value a lie.
    imgsz = int(detection.get("imgsz", 640))
    if imgsz <= 0:
        raise ValueError(f"detection.imgsz must be positive, got {imgsz}")
    if imgsz % 32 != 0:
        raise ValueError(f"detection.imgsz must be a multiple of 32 (model stride), got {imgsz}")

    out = raw["output"]
    save = SaveOutputConfig(**out.get("save", {}))
    upload = UploadOutputConfig(**out.get("upload", {}))
    broadcast = BroadcastOutputConfig(**out.get("broadcast", {}))
    http = HttpOutputConfig(**out.get("http", {}))
    if upload.enabled and not upload.endpoint:
        raise ValueError("output.upload.enabled is true but no endpoint is set")
    if broadcast.enabled and not broadcast.socket_path:
        raise ValueError("output.broadcast.enabled is true but no socket_path is set")
    if http.enabled and not (1 <= http.port <= 65535):
        raise ValueError(f"output.http.port must be between 1 and 65535, got {http.port}")

    return PipelineConfig(
        repo_root=repo_root,
        show_live=bool(raw.get("show_live", False)),
        ingestion_mode=mode,
        channels=channels,
        nominal_fps=nominal_fps,
        target_fps=target_fps,
        sync_buffer_seconds=sync_buffer_seconds,
        max_buffer_ahead_sec=max_buffer_ahead_sec,
        max_consecutive_skips=max_consecutive_skips,
        sync_margin_sec=sync_margin_sec,
        model_path=detection["model_path"],
        conf=float(detection.get("conf", 0.25)),
        imgsz=imgsz,
        cameras=cameras,
        save=save,
        upload=upload,
        broadcast=broadcast,
        http=http,
    )
