"""GPU-native letterbox helpers shared between the ingest side
(GpuDecodeSource, which now letterboxes every decoded frame immediately
so the buffered representation is small) and process_video.py (which
still needs the target shape and the deferred float32 conversion right
before inference).

Kept separate from scripts/process_video.py because it needs to be
importable from the library side (src/co_perception/ingest/) -- scripts
import from src, not the reverse.
"""
from __future__ import annotations

import math
import threading

import torch
import torch.nn.functional as F

LETTERBOX_PAD_VALUE = 114
LETTERBOX_STRIDE = 32


def letterbox_target_shape(src_h, src_w, long_side, stride=LETTERBOX_STRIDE):
    """Target (h, w) scaling src's long side to `long_side`, each dimension
    then rounded up to a multiple of stride -- mirrors Ultralytics' own
    check_imgsz(..., min_dim=2) + LetterBox sizing. See
    scripts/process_video.py's original _letterbox_target_shape for the
    full rationale (passing imgsz as a bare scalar to .track() forces a
    square target instead of this long-side scaling)."""
    scale = long_side / max(src_h, src_w)
    new_h = math.ceil((src_h * scale) / stride) * stride
    new_w = math.ceil((src_w * scale) / stride) * stride
    return new_h, new_w


def gpu_letterbox_to_uint8(rgb_tensor, new_shape):
    """GPU-native resize+pad, returning uint8 -- NOT the float32-in-[0,1]
    that scripts/process_video.py's _gpu_letterbox returns. That
    distinction matters: this is called once per DECODED frame (buffer
    time), not once per inference tick, so returning float32 here would
    make a letterboxed frame cost the same 14.75MB as today's full-res
    uint8 buffer (1280x960x3x4 bytes == 2560x1920x3x1 byte) -- no VRAM
    savings at all, silently defeating the point of buffering smaller
    frames. The deferred normalize-to-float happens once, cheaply, on the
    small already-stacked (4,3,H,W) batch right before inference instead
    -- see track_input_from_uint8_batch.

    Same algorithm as _gpu_letterbox: same r computation, same round()
    convention for split padding, same pad value, same center-padding.
    Returns (padded_uint8_tensor, r, pad_left, pad_top, content_h,
    content_w) -- same 6-tuple shape _gpu_letterbox returns, so existing
    callers (_rescale_xyxy_from_letterbox, _unletterbox_uint8_to_bgr_numpy)
    work unchanged regardless of which letterbox path produced a given
    frame's boxes.

    rgb_tensor: (3, H, W) CUDA uint8. Returns (3, new_h, new_w) CUDA uint8.
    """
    _, h, w = rgb_tensor.shape
    new_h, new_w = new_shape
    r = min(new_h / h, new_w / w)
    new_unpad_w, new_unpad_h = round(w * r), round(h * r)
    dw, dh = new_w - new_unpad_w, new_h - new_unpad_h
    dw /= 2
    dh /= 2

    x = rgb_tensor.unsqueeze(0).float()  # (1, 3, H, W) -- float only for interpolate; not normalized
    if (new_unpad_h, new_unpad_w) != (h, w):
        x = F.interpolate(x, size=(new_unpad_h, new_unpad_w), mode="bilinear", align_corners=False)

    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    x = F.pad(x, (left, right, top, bottom), mode="constant", value=float(LETTERBOX_PAD_VALUE))

    x = x.round().clamp(0, 255).to(torch.uint8)
    return x.squeeze(0), r, left, top, new_unpad_h, new_unpad_w


def track_input_from_uint8_batch(stacked_uint8):
    """(N, 3, H, W) CUDA uint8 -> (N, 3, H, W) CUDA float32 in [0, 1].
    The normalize step deferred from gpu_letterbox_to_uint8 -- call this
    once per tick on the small stacked inference batch, not once per
    decoded frame."""
    return stacked_uint8.float() / 255.0


class LetterboxShapeResolver:
    """Thread-safe, memoize-on-first-call target-shape resolver, shared
    across all 4 GpuDecodeSource instances so every channel letterboxes
    to the SAME target shape regardless of which channel's decoder
    produces the first usable frame. Without this, each channel's reader
    thread would independently derive its own target shape the moment it
    starts decoding -- fine if all 4 channels share one native resolution
    (true today), but silently wrong the moment that stops being true,
    since nothing would otherwise catch a mismatch before torch.stack
    throws a generic shape error deep in the tick loop."""

    def __init__(self, imgsz):
        self._imgsz = imgsz
        self._shape = None
        self._lock = threading.Lock()

    def resolve(self, h, w):
        with self._lock:
            if self._shape is None:
                self._shape = letterbox_target_shape(h, w, self._imgsz)
            return self._shape
