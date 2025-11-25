#!/usr/bin/env python3
"""
Street View -> Depth (MiDaS ONNX on CPU) -> Point Cloud (PLY)
- Crawls pano centers along a path via Street View Metadata 'links'
- Downloads per-pano perspective images (Static API)
- Runs MiDaS-small (ONNX) on CPU (fast on Apple Silicon)
- Back-projects to 3D with your FOV, georeferences to ENU, fuses, saves PLY

Usage:
  python streetview_to_pointcloud.py

Tune the CONFIG section for your start/end, headings, FOV, etc.
"""

import os
import io
import math
import time
import json
import zipfile
import urllib.request
from collections import deque
from typing import List, Tuple

import numpy as np
import requests
import onnxruntime as ort
from PIL import Image
#import cv2
import open3d as o3d
from dotenv import load_dotenv

# =========================
# CONFIG (edit these)
# =========================
START_POINT = (37.86693610933274, -122.26582960532846)  # College & Bancroft
END_POINT   = (37.867554398886526, -122.2612591212542)  # Telegraph & Bancroft

OUTPUT_DIR = "sv_pointcloud_out"
IMAGES_PER_PANO_HEADINGS = [0, 90, 180, 270]   # headings for perspective crops
IMAGE_SIZE = (640, 640)                        # Static API max is typically 640x640; we use scale=2
SCALE = 2                                      # server-side upsample
PITCH = 0
FOV_DEG = 90.0                                 # pinhole FOV for back-projection
CAMERA_HEIGHT_M = 1.6                          # approximate camera height above ground
MAX_STEPS = 80                                 # BFS crawl budget
RATE_LIMIT_S = 0.05

# Depth scaling: MiDaS depth is relative; multiply to get "meters-ish".
# You can calibrate this later by matching inter-pano spacing or known heights.
DEPTH_SCALE = 12.0

# =========================
# API setup
# =========================
load_dotenv()
API_KEY = os.getenv("GOOGLE_STREET_VIEW_API_KEY", "")
META_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMG_URL  = "https://maps.googleapis.com/maps/api/streetview"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# MiDaS-small ONNX (download once if absent)
# =========================
MODEL_URL = "https://github.com/isl-org/MiDaS/releases/download/v3_1_small/model-small.onnx"
MODEL_PATH = os.path.join(OUTPUT_DIR, "midas_small.onnx")

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[download] MiDaS-small ONNX → {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Preprocess for MiDaS-small (expected 256x256, NCHW, float32, normalized)
# def midas_preprocess(img_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int]]:
#     # convert to RGB
#     img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     h0, w0 = img.shape[:2]
#     # resize shortest side to 256, keep aspect (then letterbox to square)
#     target = 256
#     scale = target / min(h0, w0)
#     nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
#     img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

#     # center-crop or pad to 256x256
#     top = (nh - target) // 2
#     left = (nw - target) // 2
#     if nh >= target and nw >= target:
#         img = img[top:top+target, left:left+target]
#     else:
#         canvas = np.zeros((target, target, 3), dtype=img.dtype)
#         ty = max(0, -top); tx = max(0, -left)
#         sy = max(0, top);  sx = max(0, left)
#         canvas[ty:ty+img.shape[0], tx:tx+img.shape[1]] = img[max(0,-top):, max(0,-left):]
#         img = canvas

#     img = img.astype(np.float32) / 255.0
#     # Common MiDaS small normalization (approx):
#     img = (img - 0.5) / 0.5   # scale to [-1,1]
#     img_chw = np.transpose(img, (2,0,1))[None, ...]  # NCHW
#     return img_chw.astype(np.float32), (w0, h0)

def midas_preprocess(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int,int]]:
    # rgb: HxWx3 uint8
    h0, w0 = rgb.shape[:2]
    target = 256

    # resize shortest side to 256 with aspect
    scale = target / min(h0, w0)
    nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
    img = Image.fromarray(rgb).resize((nw, nh), resample=Image.BICUBIC)

    # center-crop or pad to 256x256
    top = (nh - target) // 2
    left = (nw - target) // 2
    if nh >= target and nw >= target:
        img = img.crop((left, top, left + target, top + target))
    else:
        canvas = Image.new("RGB", (target, target))
        canvas.paste(img, (max(0, -left), max(0, -top)))
        img = canvas

    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]
    x = np.transpose(arr, (2, 0, 1))[None, ...]  # NCHW
    return x.astype(np.float32), (w0, h0)


def run_midas_onnx(sess: ort.InferenceSession, rgb: np.ndarray) -> np.ndarray:
    x, (w0, h0) = midas_preprocess(rgb)
    input_name = sess.get_inputs()[0].name
    y = sess.run(None, {input_name: x})[0]  # (1,1,256,256)
    y = y[0, 0]

    # resize back to original with PIL
    y_img = Image.fromarray(y.astype(np.float32))
    y_resized = y_img.resize((rgb.shape[1], rgb.shape[0]), resample=Image.BICUBIC)
    y = np.asarray(y_resized).astype(np.float32)

    # normalize positive
    y = y - y.min() + 1e-6
    y = y / y.mean()
    return y


# =========================
# Street View helpers
# =========================
def meta_by_location(lat, lng):
    r = requests.get(META_URL, params={"location": f"{lat},{lng}", "key": API_KEY})
    return r.json()

def meta_by_pano(pano_id):
    r = requests.get(META_URL, params={"pano": pano_id, "key": API_KEY})
    return r.json()

def crawl_path(start_latlng, end_latlng, max_steps=MAX_STEPS):
    sm = meta_by_location(*start_latlng)
    em = meta_by_location(*end_latlng)
    if sm.get("status") != "OK":
        raise RuntimeError(f"No imagery at start: {sm}")
    if em.get("status") != "OK":
        raise RuntimeError(f"No imagery at end: {em}")
    start_pano = sm["pano_id"]
    end_pano   = em["pano_id"]

    visited = set()
    parents = {start_pano: None}
    q = deque([start_pano])
    found = False

    while q and len(visited) < max_steps:
        p = q.popleft()
        if p in visited:
            continue
        visited.add(p)
        if p == end_pano:
            found = True
            break
        m = meta_by_pano(p)
        for link in m.get("links", []):
            nxt = link.get("pano_id")
            if nxt and nxt not in visited and nxt not in q:
                parents[nxt] = p
                q.append(nxt)
        time.sleep(RATE_LIMIT_S)

    chain = []
    if found:
        cur = end_pano
        while cur is not None:
            chain.append(cur)
            cur = parents.get(cur)
        chain.reverse()
    else:
        # fallback: linearize visited order
        chain = list(visited)
    return chain

def fetch_image_by_pano(pano_id, heading_deg):
    params = {
        "pano": pano_id,
        "size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        "scale": SCALE,
        "heading": heading_deg,
        "pitch": PITCH,
        "fov": FOV_DEG,
        "source": "outdoor",
        "key": API_KEY
    }
    r = requests.get(IMG_URL, params=params)
    r.raise_for_status()
    # Decode with PIL
    img = Image.open(io.BytesIO(r.content)).convert("RGB")  # PIL image in RGB
    # Convert to NumPy array in RGB; if you really want BGR, reverse channels later
    rgb = np.asarray(img)
    return rgb

# =========================
# Geo transforms (WGS84 → ECEF → ENU)
# =========================
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3

def llh_to_ecef(lat_deg, lon_deg, h_m):
    lat = math.radians(lat_deg); lon = math.radians(lon_deg)
    a = WGS84_A; e2 = WGS84_E2
    N = a / math.sqrt(1 - e2 * (math.sin(lat)**2))
    x = (N + h_m) * math.cos(lat) * math.cos(lon)
    y = (N + h_m) * math.cos(lat) * math.sin(lon)
    z = (N * (1 - e2) + h_m) * math.sin(lat)
    return np.array([x,y,z], dtype=np.float64)

def ecef_to_enu(xyz, ref_lat_deg, ref_lon_deg, ref_h_m, ref_ecef=None):
    if ref_ecef is None:
        ref_ecef = llh_to_ecef(ref_lat_deg, ref_lon_deg, ref_h_m)
    lat = math.radians(ref_lat_deg); lon = math.radians(ref_lon_deg)
    R = np.array([
        [-math.sin(lon),             math.cos(lon),              0],
        [-math.sin(lat)*math.cos(lon), -math.sin(lat)*math.sin(lon), math.cos(lat)],
        [ math.cos(lat)*math.cos(lon),  math.cos(lat)*math.sin(lon), math.sin(lat)]
    ], dtype=np.float64)
    return R @ (xyz - ref_ecef)

def enu_of_latlon(lat_deg, lon_deg, h_m, ref_lat_deg, ref_lon_deg, ref_h_m, ref_ecef=None):
    return ecef_to_enu(llh_to_ecef(lat_deg, lon_deg, h_m), ref_lat_deg, ref_lon_deg, ref_h_m, ref_ecef)

# =========================
# Back-projection (pinhole) → local camera coords
# =========================
def backproject_depth_to_points(depth: np.ndarray, fov_deg: float) -> np.ndarray:
    """
    depth: (H,W) relative depth (we'll scale outside)
    Returns Nx3 array in camera coords (x forward, y right, z down? We'll use x=forward, y=left, z=up by convention then rotate.)
    """
    H, W = depth.shape
    # Pinhole intrinsics from FOV (horizontal)
    fx = fy = (W * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    # pixel grid
    jj, ii = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    x_cam = (jj - cx) / fx
    y_cam = (ii - cy) / fy
    z = np.ones_like(x_cam)

    # direction vectors normalized
    dirs = np.stack([x_cam, y_cam, z], axis=-1)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8
    dirs = dirs / norms

    # Scale by depth
    D = depth[..., None]
    pts = dirs * D
    # Camera convention: forward +Z is more standard in CV, but here we used z=1 before norm.
    # We'll keep camera forward ~ +Z, right ~ +X, down ~ +Y. We'll fix with rotations later.
    return pts.reshape(-1, 3)

def rotate_yaw(points: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Rotate points around Z-up (ENU up) by yaw (degrees), positive CW from north → match heading convention."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[ c, -s, 0],
                  [ s,  c, 0],
                  [ 0,  0, 1]], dtype=np.float64)
    return points @ R.T

# =========================
# Main
# =========================
def main():
    if not API_KEY:
        raise SystemExit("Set GOOGLE_STREET_VIEW_API_KEY in .env")

    ensure_model()
    sess = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    print("[crawl] finding pano chain...")
    chain = crawl_path(START_POINT, END_POINT, MAX_STEPS)
    print(f"[crawl] panos found: {len(chain)}")

    # Reference origin = first pano lat/lon, zero height
    ref_lat, ref_lon = START_POINT
    ref_h = 0.0
    ref_ecef = llh_to_ecef(ref_lat, ref_lon, ref_h)

    all_pts_enu = []

    for i, pano_id in enumerate(chain):
        meta = meta_by_pano(pano_id)
        if meta.get("status") != "OK":
            print(f"[skip] bad metadata for pano {pano_id}")
            continue
        plat = meta["location"]["lat"]
        plon = meta["location"]["lng"]

        # Camera translation (ENU) at pano center; add camera height
        t_enu = enu_of_latlon(plat, plon, 0.0, ref_lat, ref_lon, ref_h, ref_ecef)
        t_enu[2] += CAMERA_HEIGHT_M

        for hidx, heading in enumerate(IMAGES_PER_PANO_HEADINGS):
            try:
                rgb = fetch_image_by_pano(pano_id, heading)
            except Exception as e:
                print(f"[warn] fetch failed for {pano_id} h={heading}: {e}")
                continue

            depth_rel = run_midas_onnx(sess, rgb)  # relative depth
            depth_m = depth_rel * DEPTH_SCALE

            pts_cam = backproject_depth_to_points(depth_m, FOV_DEG)  # camera coords

            # Re-orient camera coords to ENU:
            # Our back-projection used forward ~ +Z. We want ENU basis where +X=East, +Y=North, +Z=Up.
            # We'll treat camera local forward as +Y_ENU for a "car-style" forward; then rotate by heading.
            # First, map (x_cam,y_cam,z_cam) -> provisional ENU: (E,N,U)
            # Choose: forward(+z) -> +N, right(+x) -> +E, up(-y) -> +U
            E = pts_cam[:, 0]          # right → East
            N = pts_cam[:, 2]          # forward → North
            U = -pts_cam[:, 1]         # up     → Up
            pts_enu_local = np.stack([E, N, U], axis=1)

            # Rotate around Up by heading (degrees from North, clockwise)
            pts_enu_rot = rotate_yaw(pts_enu_local, heading)

            # Translate to pano center in ENU
            pts_enu = pts_enu_rot + t_enu[None, :]

            # Optional: simple filtering (clip extreme depths)
            zmax = np.percentile(depth_m, 99.5)
            keep = (np.linalg.norm(pts_enu_rot, axis=1) < 3*zmax)
            all_pts_enu.append(pts_enu[keep])

            print(f"[ok] pano {i+1}/{len(chain)} heading {heading} -> {keep.sum()} pts")

            time.sleep(RATE_LIMIT_S)

    if not all_pts_enu:
        raise SystemExit("No points reconstructed. Check API key / coverage.")

    P = np.concatenate(all_pts_enu, axis=0)

    # Downsample and save PLY
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd = pcd.voxel_down_sample(voxel_size=0.2)  # 20cm voxels
    out_ply = os.path.join(OUTPUT_DIR, "streetview_pointcloud.ply")
    o3d.io.write_point_cloud(out_ply, pcd)
    print(f"[done] Wrote {out_ply}  (points={np.asarray(pcd.points).shape[0]})")

    # Save a small manifest for reproducibility
    manifest = {
        "start_point": START_POINT,
        "end_point": END_POINT,
        "panos": chain,
        "headings": IMAGES_PER_PANO_HEADINGS,
        "image_size": IMAGE_SIZE,
        "scale": SCALE,
        "pitch": PITCH,
        "fov_deg": FOV_DEG,
        "camera_height_m": CAMERA_HEIGHT_M,
        "depth_scale": DEPTH_SCALE
    }
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[info] manifest saved.")
    

if __name__ == "__main__":
    main()
