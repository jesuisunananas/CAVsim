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
class PipelineConfig:
    repo_root: Path
    show_live: bool
    ingestion_mode: str
    channels: list[ChannelSource]
    nominal_fps: float
    target_fps: float
    sync_buffer_seconds: float
    model_path: str
    conf: float
    cameras: list[CameraConfig]
    save: SaveOutputConfig
    upload: UploadOutputConfig
    broadcast: BroadcastOutputConfig

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

    detection = raw["detection"]
    cameras = [CameraConfig(**cam) for cam in detection["cameras"]]

    out = raw["output"]
    save = SaveOutputConfig(**out.get("save", {}))
    upload = UploadOutputConfig(**out.get("upload", {}))
    broadcast = BroadcastOutputConfig(**out.get("broadcast", {}))
    if upload.enabled and not upload.endpoint:
        raise ValueError("output.upload.enabled is true but no endpoint is set")
    if broadcast.enabled and not broadcast.socket_path:
        raise ValueError("output.broadcast.enabled is true but no socket_path is set")

    return PipelineConfig(
        repo_root=repo_root,
        show_live=bool(raw.get("show_live", False)),
        ingestion_mode=mode,
        channels=channels,
        nominal_fps=nominal_fps,
        target_fps=target_fps,
        sync_buffer_seconds=sync_buffer_seconds,
        model_path=detection["model_path"],
        conf=float(detection.get("conf", 0.25)),
        cameras=cameras,
        save=save,
        upload=upload,
        broadcast=broadcast,
    )
