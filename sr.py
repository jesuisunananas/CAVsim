#!/usr/bin/env python3
"""
Google Street View 3D Reconstruction Pipeline (Option B)
-------------------------------------------------------

FULL PIPELINE:
1. Fetch pano metadata (GPS + pano_id)
2. Fetch full-resolution equirectangular pano tiles (NOT static API)
3. Fetch depth map (GSV plane-based depth)
4. Decode spherical GSV depth → full-res depth
5. Segment road / lane markings / sidewalks using Fast-SCNN (Cityscapes)
6. RANSAC road plane fitting
7. Project road pixels onto ENU plane → build high-res road texture
8. Backproject all pixels → point cloud in ENU
9. Poisson mesh (non-road) + separate road plane mesh

Outputs in ./output/
"""

import os
import io
import math
import time
import json
import base64
import zlib
import struct
import requests
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T

import open3d as o3d
from skimage.transform import resize
from sklearn.linear_model import RANSACRegressor
import torch.nn.functional as F
import sys

# ====================================================
# CONFIG
# ====================================================
#START_LAT = 37.866936
#START_LON = -122.265829
START_LAT = 37.867265
START_LON = -122.265812
END_LAT   = 37.867554
END_LON   = -122.261259
SAMPLES   = 5

ROAD_FRAME_ROT = None

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TILE_ZOOM = 3     # 4 gives ~13k resolution pano
CAM_HEIGHT = 1.6  # meters
ROAD_RES   = 0.05 # meters per pixel for output road texture

FAST_SCNN_REPO = "/home/arjunr/dev/CAVsim/Fast-SCNN-pytorch"
FAST_SCNN_WEIGHTS = os.path.join(FAST_SCNN_REPO, "weights", "fast_scnn_citys.pth")
sys.path.append(FAST_SCNN_REPO)
from models.fast_scnn import FastSCNN as FSNet

# ====================================================
# LOCAL PANO FALLBACK
# ====================================================
USE_LOCAL_PANO = True  # toggle this on/off
LOCAL_PANO_DIR = os.path.join(OUTPUT_DIR, "local_panos")
os.makedirs(LOCAL_PANO_DIR, exist_ok=True)
# Map real Google pano IDs (for depth + geo) to local JPG names
# LOCAL_PANOS = [
#     ("jhSIAsbw05U4w6zb8Roesg", "u1"),
#     ("dPlCbF7XC9awbk5pn_2vyQ", "u2"),
# ]

LOCAL_PANOS = [
    {
        "gsv_id": "jhSIAsbw05U4w6zb8Roesg",  # full correct GSV ID if you have it
        "local_id": "u1",
        "lat": 37.867265,
        "lon": -122.265812,
    },
    {
        "gsv_id": "dPlCbF7XC9awbk5pn_2vyQ",  # full correct GSV ID if you have it
        "local_id": "u2",
        "lat": 37.867354,
        "lon": -122.265830,
    },
    {
    
        "gsv_id": "BsEfn0IKJVb086iDM9_4xg",  # full correct GSV ID if you have it
        "local_id": "u3",
        "lat": 37.867440,
        "lon": -122.265848,
    },
    {
        "gsv_id": "MjX71WdQcqGmkIzqNV_COw",  # full correct GSV ID if you have it
        "local_id": "u4",
        "lat": 37.867526,
        "lon": -122.265865,
    },
    {
        "gsv_id": "GjYss7GuKgQYYWj-Z2xNJQ",  # full correct GSV ID if you have it
        "local_id": "u5",
        "lat": 37.867610,
        "lon": -122.265881,
    },
    {
        "gsv_id": "ltVOJ4hINFWTW1WFyJGRZQ",  # full correct GSV ID if you have it
        "local_id": "u6",
        "lat": 37.867722,
        "lon": -122.265902,
    },
    {
        "gsv_id": "Xpbuk2tA9RSG-w52WcU3sg",  # full correct GSV ID if you have it
        "local_id": "u7",
        "lat": 37.867834,
        "lon": -122.265930,
    },
    {
        "gsv_id": "kD-eXLluX3G8K_QF_ZuVaA",  # full correct GSV ID if you have it
        "local_id": "u8",
        "lat": 37.867971,
        "lon": -122.265954,
    },
    {
        "gsv_id": "wVzMU5aF5Fg-WKJcQRo6qw",  # full correct GSV ID if you have it
        "local_id": "u9",
        "lat": 37.868072,
        "lon": -122.265969,
    },
]


# ====================================================
# LOAD API KEY
# ====================================================
load_dotenv()
API_KEY = os.getenv("GOOGLE_STREET_VIEW_API_KEY")

if not API_KEY:
    raise SystemExit("Missing GOOGLE_STREET_VIEW_API_KEY in .env")


def create_sv_session():
    url = "https://tile.googleapis.com/v1/createSession"

    body = {
        "mapType": "streetview",
    }

    r = requests.post(
        url,
        params={"key": API_KEY},
        json=body
    )
    if r.status_code != 200:
        raise RuntimeError(f"Failed to create session: {r.text}")

    session = r.json()#["session"]
    d = session.get("session")
    if isinstance(d, dict):
        return d.get("session")
    return d #session["session"]   # string session token

def fetch_sv_metadata(pano_id, session_token):
    url = f"https://tile.googleapis.com/v1/streetview/metadata"

    r = requests.get(
        url,
        params={
            "session": session_token,
            "panoId": pano_id,
            "key": API_KEY
        }
    )
    if r.status_code != 200:
        raise RuntimeError(f"streetview/metadata error: {r.text}")

    return r.json()

def meta_by_pano(pano_id):
    r = requests.get(META_URL, params={"pano": pano_id, "key": API_KEY})
    return r.json()


# ====================================================
# 0. Interpolation helper
# ====================================================
def interpolate_points(start, end, n):
    (lat1, lon1), (lat2, lon2) = start, end
    pts = []
    for i in range(n):
        t = i / max(1,n-1)
        pts.append((lat1 + (lat2-lat1)*t,
                    lon1 + (lon2-lon1)*t))
    return pts

# ====================================================
# 1. Metadata lookup
# ====================================================
META_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"

def meta_by_location(lat, lon):
    r = requests.get(META_URL, params={
        "location": f"{lat},{lon}",
        "key": API_KEY
    })
    return r.json()

def fetch_pano_chain():
    pts = interpolate_points((START_LAT,START_LON),
                             (END_LAT,END_LON), SAMPLES)
    out = []
    last = None
    for (lat,lon) in pts:
        m = meta_by_location(lat,lon)
        if m.get("status") == "OK":
            pid = m["pano_id"]
            if pid != last:
                out.append(pid)
                last = pid
    print(f"[meta] collected {len(out)} unique panos")
    return out

# ====================================================
# 2. TILE API FETCH (REAL EQUIRECT PANORAMA)
# ====================================================
def fetch_tile(session_token, pano_id, zoom, x, y):
    url = (
        f"https://tile.googleapis.com/v1/streetview/tiles/"
        f"{zoom}/{x}/{y}"
    )

    r = requests.get(
        url,
        params={
            "session": session_token,
            "panoId": pano_id,
            "key": API_KEY
        }
    )

    if r.status_code != 200:
        print(f"[tile] HTTP {r.status_code} for {url}")
        print(f"[tile] response text (first 200 chars): {r.text[:200]}")
        return None

    im = Image.open(io.BytesIO(r.content)).convert("RGB")
    arr = np.asarray(im)
    print(f"[tile] pano={pano_id} z={zoom} x={x} y={y} tile max={arr.max()}")
    return im


# def fetch_full_pano(pano_id, zoom=TILE_ZOOM):
#     session_token = create_sv_session()
#     meta = fetch_sv_metadata(pano_id, session_token)

#     W = meta["imageWidth"]
#     H = meta["imageHeight"]
#     tw = meta["tileWidth"]
#     th = meta["tileHeight"]

#     tiles_x = W // tw
#     tiles_y = H // th

#     print(f"[pano] Fetch {pano_id} at zoom={zoom} → tiles {tiles_x}×{tiles_y}")

#     pano = Image.new("RGB", (W, H))
#     any_tile = False

#     for ty in range(tiles_y):
#         for tx in range(tiles_x):
#             tile = fetch_tile(session_token, pano_id, zoom, tx, ty)
#             if tile is None:
#                 # leave blank, but note the failure
#                 continue
#             any_tile = True
#             pano.paste(tile, (tx * tw, ty * th))

#     rgb = np.asarray(pano)
#     print(f"[pano] stitched rgb stats: min={rgb.min()}, max={rgb.max()}")

#     if (not any_tile) or rgb.max() == 0:
#         raise RuntimeError(
#             "[pano] ERROR: Stitched panorama is all black. "
#             "Tile API is not returning imagery for this key/session."
#         )

#     out_path = os.path.join(OUTPUT_DIR, f"{pano_id}.jpg")
#     pano.save(out_path)
#     print(f"[pano] saved {out_path}")

#     return pano

def fetch_full_pano(pano_id, zoom=TILE_ZOOM):
    """
    If USE_LOCAL_PANO is True, load pano from disk:
        output/local_panos/<pano_id>.jpg
    Otherwise, fall back to the tile API (same as before).
    """

    # ---------- 1. Local fallback ----------
    if USE_LOCAL_PANO:
        local_path = os.path.join(LOCAL_PANO_DIR, f"{pano_id}.jpg")
        if not os.path.exists(local_path):
            raise RuntimeError(
                f"[pano/local] expected local pano at {local_path} "
                f"but file does not exist. "
                f"Create it manually (e.g., save a pano JPG there)."
            )

        pano = Image.open(local_path).convert("RGB")
        rgb = np.asarray(pano)
        print(f"[pano/local] {pano_id} rgb stats: "
              f"min={rgb.min()}, max={rgb.max()}, shape={rgb.shape}, dtype={rgb.dtype}")
        return pano

    # ---------- 2. Remote Tile API (original behavior) ----------
    session_token = create_sv_session()
    meta = fetch_sv_metadata(pano_id, session_token)

    W = meta["imageWidth"]
    H = meta["imageHeight"]
    tw = meta["tileWidth"]
    th = meta["tileHeight"]

    tiles_x = W // tw
    tiles_y = H // th

    print(f"[pano] Fetch {pano_id} at zoom={zoom} → tiles {tiles_x}×{tiles_y}")

    pano = Image.new("RGB", (W, H))

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile = fetch_tile(session_token, pano_id, zoom, tx, ty)
            if tile:
                pano.paste(tile, (tx * tw, ty * th))

    rgb = np.asarray(pano)
    print(f"[pano] stitched rgb stats: min={rgb.min()}, max={rgb.max()}")

    if rgb.max() == 0:
        raise RuntimeError(
            "[pano] ERROR: Stitched panorama is all black. "
            "Tile API likely returned errors (e.g., quota exceeded)."
        )

    out_path = os.path.join(OUTPUT_DIR, f"{pano_id}.jpg")
    pano.save(out_path)
    print(f"[pano] saved {out_path}")

    return pano




# ====================================================
# 3. GSV DEPTH MAP DECODER
#    (Your StackOverflow code, wrapped)
# ====================================================

def decode_depth_base64(b64_string):
    b64_string += "="*((4 - len(b64_string)%4)%4)
    data = b64_string.replace("-","+").replace("_","/")
    raw = base64.b64decode(data)
    raw = zlib.decompress(raw)
    return np.frombuffer(raw, dtype=np.uint8)

def get_bin(a):
    ba = bin(a)[2:]
    return "0"*(8-len(ba)) + ba

def getUInt16(arr, i):
    return int(get_bin(arr[i+1]) + get_bin(arr[i]), 2)

def bin_to_float(b):
    return struct.unpack("!f", struct.pack("!I", int(b,2)))[0]

def getFloat32(arr, i):
    bits = "".join(get_bin(x) for x in arr[i:i+4][::-1])
    return bin_to_float(bits)

def parse_depth_bytes(b64_string: str) -> np.ndarray:
    """Base64 → bytes array (no assumptions about compression here)."""
    b64_string += "=" * ((4 - len(b64_string) % 4) % 4)
    data = b64_string.replace("-", "+").replace("_", "/")
    raw = base64.b64decode(data)
    # Some sources decompress here; for the photometa endpoint used in SO,
    # the bytes are *already* the plane-encoded array, so don't zlib.decompress.
    # If you ever see zlib.error, flip this flag.
    # raw = zlib.decompress(raw)
    return np.frombuffer(raw, dtype=np.uint8)


def parse_header(depth_map_bytes: np.ndarray) -> dict:
    return {
        "headerSize": depth_map_bytes[0],
        "numberOfPlanes": getUInt16(depth_map_bytes, 1),
        "width": getUInt16(depth_map_bytes, 3),
        "height": getUInt16(depth_map_bytes, 5),
        "offset": getUInt16(depth_map_bytes, 7),
    }


def parse_planes(header: dict, depth_map_bytes: np.ndarray) -> dict:
    indices = []
    planes = []

    w = header["width"]
    h = header["height"]
    offset = header["offset"]

    # pixel → plane index
    for i in range(w * h):
        indices.append(depth_map_bytes[offset + i])

    # planes: 3×normal + 1×d, each float32
    base = offset + w * h
    for i in range(header["numberOfPlanes"]):
        byteOffset = base + i * 4 * 4
        n = [
            getFloat32(depth_map_bytes, byteOffset),
            getFloat32(depth_map_bytes, byteOffset + 4),
            getFloat32(depth_map_bytes, byteOffset + 8),
        ]
        d = getFloat32(depth_map_bytes, byteOffset + 12)
        planes.append({"n": n, "d": d})

    return {"planes": planes, "indices": np.array(indices, dtype=np.int32)}


def compute_depth_map(header: dict,
                      indices: np.ndarray,
                      planes: list[dict]) -> dict:
    """
    Exact Python port of the StackOverflow / GSVPanoDepth.js logic.
    Returns dict with width, height, and flattened depth array.
    """
    w = header["width"]
    h = header["height"]

    depthMap = np.empty(w * h, dtype=np.float32)

    sin_theta = np.empty(h, dtype=np.float32)
    cos_theta = np.empty(h, dtype=np.float32)
    sin_phi   = np.empty(w, dtype=np.float32)
    cos_phi   = np.empty(w, dtype=np.float32)

    for y in range(h):
        theta = (h - y - 0.5) / h * np.pi
        sin_theta[y] = np.sin(theta)
        cos_theta[y] = np.cos(theta)

    for x in range(w):
        phi = (w - x - 0.5) / w * 2 * np.pi + np.pi / 2.0
        sin_phi[x] = np.sin(phi)
        cos_phi[x] = np.cos(phi)

    for y in range(h):
        for x in range(w):
            planeIdx = indices[y * w + x]

            v0 = sin_theta[y] * cos_phi[x]
            v1 = sin_theta[y] * sin_phi[x]
            v2 = cos_theta[y]

            if planeIdx > 0:
                plane = planes[planeIdx]
                denom = (
                    v0 * plane["n"][0]
                    + v1 * plane["n"][1]
                    + v2 * plane["n"][2]
                )
                t = abs(plane["d"] / denom)
                depthMap[y * w + (w - x - 1)] = t
            else:
                depthMap[y * w + (w - x - 1)] = 1e10  # "infinite"

    return {"width": w, "height": h, "depthMap": depthMap}


def decode_depth_map_json(pano_id: str) -> tuple[dict, np.ndarray, list[dict]]:
    """
    Use the exact Google Maps photometa pattern from the StackOverflow answer.
    """
    endpoint = "https://www.google.com/maps/photometa/v1"

    params = {
        "authuser": "0",
        "hl": "en",
        "gl": "us",
        "pb": (
            "!1m4!1smaps_sv.tactile!11m2!2m1!1b1!"
            "!2m2!1sen!2suk!"
            "!3m3!1m2!1e2!2s" + pano_id +
            "!4m57!"
            "1e1!1e2!1e3!1e4!1e5!1e6!1e8!1e12!"
            "2m1!1e1!"
            "4m1!1i48!"
            "5m1!1e1!"
            "5m1!1e2!"
            "6m1!1e1!"
            "6m1!1e2!"
            "9m36!"
            "1m3!1e2!2b1!3e2!"
            "1m3!1e2!2b0!3e3!"
            "1m3!1e3!2b1!3e2!"
            "1m3!1e3!2b0!3e3!"
            "1m3!1e8!2b0!3e3!"
            "1m3!1e1!2b0!3e3!"
            "1m3!1e4!2b0!3e3!"
            "1m3!1e10!2b1!3e2!"
            "1m3!1e10!2b0!3e3"
        )
    }

    r = requests.get(endpoint, params=params)
    if r.status_code != 200:
        print("URL:", r.url)
        print("Status:", r.status_code)
        print("First 300 chars:", r.text[:300])
        raise RuntimeError(
            f"photometa depth request failed for pano {pano_id}: "
            f"HTTP {r.status_code} – first 200 chars:\n{r.text[:200]}"
        )

    # Response has 4 junk bytes, then JSON
    try:
        j = json.loads(r.content[4:])
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse photometa JSON for {pano_id}: {e}\n"
            f"First 200 chars: {r.text[:200]}"
        )

    # This indexing is exactly what the SO / GSVPanoDepth.js flow uses
    try:
        s = j[1][0][5][0][5][1][2]
    except Exception as e:
        raise RuntimeError(
            f"Could not find depth_map field in photometa JSON for {pano_id}: {e}"
        )

    depth_bytes = parse_depth_bytes(s)
    header = parse_header(depth_bytes)
    plane_data = parse_planes(header, depth_bytes)
    indices = plane_data["indices"]
    planes = plane_data["planes"]

    return header, indices, planes



def compute_spherical_depth(header, indices, planes):
    W, H = header["width"], header["height"]
    out = np.zeros(W*H, dtype=np.float32)

    sin_theta = np.zeros(H)
    cos_theta = np.zeros(H)
    sin_phi   = np.zeros(W)
    cos_phi   = np.zeros(W)

    for y in range(H):
        theta = (H - y - 0.5)/H * math.pi
        sin_theta[y] = math.sin(theta)
        cos_theta[y] = math.cos(theta)

    for x in range(W):
        phi = (W - x - 0.5)/W * 2*math.pi + math.pi/2
        sin_phi[x] = math.sin(phi)
        cos_phi[x] = math.cos(phi)

    for y in range(H):
        for x in range(W):
            pi = indices[y*W + x]
            if pi <= 0:
                out[y*W + (W-x-1)] = 9999
                continue
            plane = planes[pi]
            v0 = sin_theta[y]*cos_phi[x]
            v1 = sin_theta[y]*sin_phi[x]
            v2 = cos_theta[y]
            denom = v0*plane["n"][0] + v1*plane["n"][1] + v2*plane["n"][2]
            out[y*W + (W-x-1)] = abs(plane["d"] / denom)

    return out.reshape(H,W)

# ====================================================
# 4. FAST-SCNN SEGMENTATION
# ====================================================

class FastSCNNWrapper(nn.Module):
    """
    Wrapper around Tramac's Fast-SCNN (Cityscapes).
    Expects the repo to be on sys.path and weights at FAST_SCNN_WEIGHTS.
    """
    def __init__(self, ckpt_path):
        super().__init__()
        self.model = FSNet(num_classes=19)  # Cityscapes has 19 classes
        state = torch.load(ckpt_path, map_location="cpu")
        # Some checkpoints use "state_dict" wrapping
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def forward(self, x):
        return self.model(x)


# ----------------------------------------------------
# expected class IDs to treat as "road"
# ----------------------------------------------------
# expected class IDs to treat as "road"
CITYSCAPES_ROAD = {0, 1, 2, 3}   # road, sidewalk, curb-ish stuff
LANE_BRIGHT     = True  # use intensity threshold to detect lane markings

def segment_road(model, rgb, device=None):
    """
    rgb: HxWx3 uint8 (full pano)
    Returns: HxW uint8 mask (1 = road-ish, 0 = non-road)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    H, W = rgb.shape[:2]

    # Resize down to something manageable for Fast-SCNN
    target_h, target_w = 512, 1024

    tfm = T.Compose([
        T.ToPILImage(),
        T.Resize((target_h, target_w)),  # keep aspect-ish
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225])
    ])

    inp = tfm(rgb).unsqueeze(0).to(device)  # 1x3xH'xW'

    with torch.no_grad():
        out = model(inp)                 # 1x19xH'xW'
        if isinstance(out, (list, tuple)):
            logits = out[0]
        else:
            logits = out
        # Upsample logits back to full pano size
        logits = F.interpolate(
            logits, size=(H, W),
            mode="bilinear",
            align_corners=False
        )
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # Road-like classes
    road = np.isin(pred, list(CITYSCAPES_ROAD)).astype(np.uint8)

    # Optional: use bright pixels as lane markings
    if LANE_BRIGHT:
        gray = rgb[..., 1]  # green channel ~ luminance-ish
        high = gray > 200
        road |= high.astype(np.uint8)

    return road


# ====================================================
# 5. GEO UTILS (WGS84 → ENU)
# ====================================================
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3

def llh_to_ecef(lat,lon,h):
    lat = math.radians(lat)
    lon = math.radians(lon)
    a = WGS84_A; e2 = WGS84_E2
    N = a / math.sqrt(1 - e2*(math.sin(lat)**2))
    x = (N+h)*math.cos(lat)*math.cos(lon)
    y = (N+h)*math.cos(lat)*math.sin(lon)
    z = (N*(1-e2)+h)*math.sin(lat)
    return np.array([x,y,z], float)

def ecef_to_enu(xyz, lat0, lon0, h0, ref_ecef=None):
    if ref_ecef is None:
        ref_ecef = llh_to_ecef(lat0,lon0,h0)
    lat = math.radians(lat0)
    lon = math.radians(lon0)
    R = np.array([
        [-math.sin(lon),              math.cos(lon),             0],
        [-math.sin(lat)*math.cos(lon), -math.sin(lat)*math.sin(lon), math.cos(lat)],
        [ math.cos(lat)*math.cos(lon),  math.cos(lat)*math.sin(lon), math.sin(lat)]
    ])
    return R @ (xyz - ref_ecef)

# ====================================================
# 6. SPHERICAL BACKPROJECTION (for full pano)
# ====================================================
def spherical_rays(H,W):
    theta = (np.arange(H)+0.5)/H * math.pi        # polar
    phi   = (np.arange(W)+0.5)/W * 2*math.pi      # azimuth
    phi = phi + math.pi/2
    # build meshgrid
    phi,theta = np.meshgrid(phi,theta)
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    return np.stack([x,y,z],-1)  # HxWx3

def backproject_spherical(rgb, depth, cam_enu):
    H,W,_ = rgb.shape
    dirs = spherical_rays(H,W).reshape(-1,3)
    pts_cam = dirs * depth.reshape(-1,1)
    pts_world = pts_cam + cam_enu  # simple translation
    colors = rgb.reshape(-1,3)/255.0
    return pts_world, colors

def backproject_spherical_masked(rgb, depth, cam_enu, mask, R_cam=None):
    """
    Backproject only pixels where `mask` == True.

    R_cam: optional 3x3 rotation matrix that transforms camera-frame
           points into ENU (e.g., from pano heading). If None, no rotation.
    """
    H, W, _ = rgb.shape

    dirs = spherical_rays(H, W).reshape(-1, 3)

    depth_flat = depth.reshape(-1)
    mask_flat  = mask.reshape(-1)

    dirs_sel   = dirs[mask_flat]
    depth_sel  = depth_flat[mask_flat]

    pts_cam = dirs_sel * depth_sel[:, None]

    # NEW: rotate rays into ENU
    # if R_cam is not None:
    #     pts_cam = (R_cam @ pts_cam.T).T

    pts_world = pts_cam + cam_enu
    pts_world = apply_road_frame(pts_world)

    rgb_float = rgb.astype(np.float32) / 255.0
    colors    = rgb_float.reshape(-1, 3)[mask_flat]

    if colors.size > 0:
        print(
            f"[debug] backproject colors min={colors.min():.3f}, "
            f"max={colors.max():.3f}"
        )
    else:
        print("[debug] backproject colors: empty")

    return pts_world, colors



# ====================================================
# 7. ROAD PLANE FITTING + TEXTURE
# ====================================================
def fit_plane_ransac(points_3d):
    """
    Fit plane z = ax + by + c
    """
    XY = points_3d[:,:2]
    Z = points_3d[:,2]
    ransac = RANSACRegressor(min_samples=50, residual_threshold=0.15)
    ransac.fit(XY,Z)
    a,b = ransac.estimator_.coef_
    c   = ransac.estimator_.intercept_
    return a,b,c

def build_road_texture(points_3d,
                       colors,
                       res=ROAD_RES,
                       max_extent_m=100.0,
                       max_tex_size=4096):
    """
    Project road points onto a *local* ENU grid and build a texture.

    - Only keeps points within +/- max_extent_m of the median E,N.
    - If the resulting grid is still too big, automatically increases `res`
      so that H,W <= max_tex_size.
    """
    E = points_3d[:, 0]
    N = points_3d[:, 1]

    # ---- 1. Restrict to a local window around the median ----
    E_center = np.median(E)
    N_center = np.median(N)

    mask = (
        (np.abs(E - E_center) <= max_extent_m) &
        (np.abs(N - N_center) <= max_extent_m)
    )
    E = E[mask]
    N = N[mask]
    C = colors[mask]


    if E.size == 0:
        print("[road] WARNING: no road points inside local window; skipping texture.")
        return None

    Emin, Emax = E.min(), E.max()
    Nmin, Nmax = N.min(), N.max()

    # ---- 2. Compute initial grid size at requested resolution ----
    W = int((Emax - Emin) / res) + 1
    H = int((Nmax - Nmin) / res) + 1

    print(f"[road] initial grid {W}x{H} at res={res} m")

    print(f"[debug] road points before crop: {len(points_3d)}")
    print(f"[debug] road points after crop:  {len(E)}")
    print(f"[debug] grid size HxW = {H} x {W}, res={res}")


    # ---- 3. If grid is too big, coarsen the resolution ----
    scale = max(W / max_tex_size, H / max_tex_size, 1.0)
    if scale > 1.0:
        res = res * scale
        W = int((Emax - Emin) / res) + 1
        H = int((Nmax - Nmin) / res) + 1
        print(f"[road] grid too large, using coarser res={res:.3f} m → {W}x{H}")

    # ---- 4. Allocate texture (float32 to save RAM) ----
    tex = np.zeros((H, W, 3), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    # ---- 5. splat points into grid ----
    for (e, n, c) in zip(E, N, C):
        i = int((n - Nmin) / res)
        j = int((e - Emin) / res)
        if 0 <= i < H and 0 <= j < W:
            tex[i, j] += c
            count[i, j] += 1.0

    valid = count > 0
    tex[valid] /= count[valid, None]

    # ---- 6. Save PNG ----
    out = (tex * 255).astype(np.uint8)
    out = np.flipud(out)  # flip N-axis so north is "up" visually
    out_path = os.path.join(OUTPUT_DIR, "road_texture.png")
    Image.fromarray(out).save(out_path)
    print(f"[road] saved road texture: {out_path} (size {W}x{H})")

    return out

def filter_corridor(all_pts, all_cols, radius_m=60.0):
    """
    Build a local 'driving corridor' point cloud directly in the
    same coordinate frame as all_pts.

    - Centered on the median of ALL points (approx camera / scene center)
    - Keep only points within radius_m horizontally
    - z is just recentered around the median height (no plane fitting yet)

    Returns:
        pts_local: Nx3  (x,y,z in meters around the local center)
        cols_local: Nx3
    """
    if all_pts.size == 0:
        print("[corridor] WARNING: all_pts is empty")
        return np.zeros((0, 3)), np.zeros((0, 3))

    # --- center = median of ALL points in this frame ---
    E = all_pts[:, 0]
    N = all_pts[:, 1]
    Z = all_pts[:, 2]

    E0 = np.median(E)
    N0 = np.median(N)
    Z0 = np.median(Z)

    # radial distance
    r2 = (E - E0) ** 2 + (N - N0) ** 2
    mask = r2 <= radius_m ** 2

    print(f"[corridor] center ≈ ({E0:.2f}, {N0:.2f}, {Z0:.2f})")
    print(f"[corridor] kept {mask.sum()} of {mask.size} points within R={radius_m} m")

    if mask.sum() == 0:
        print("[corridor] WARNING: no points after radius filter")
        return np.zeros((0, 3)), np.zeros((0, 3))

    pts_local = np.stack(
        [E[mask] - E0,
         N[mask] - N0,
         Z[mask]],
        axis=1
    )
    cols_local = all_cols[mask]

    return pts_local, cols_local

def road_frame_axes(yaw_rad):
    """
    Build a right-handed ENU frame aligned with the road:
      t_axis: along the road (forward)
      s_axis: to the left of the road
      n_axis: up
    """
    t_axis = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float32)
    t_axis /= np.linalg.norm(t_axis)

    s_axis = np.array([-t_axis[1], t_axis[0], 0.0], dtype=np.float32)
    s_axis /= np.linalg.norm(s_axis)

    n_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # columns are basis vectors in ENU
    R_road = np.stack([t_axis, s_axis, n_axis], axis=1)  # 3x3
    return R_road, t_axis, s_axis, n_axis


def project_points_to_road_frame(points_enu, R_road, origin_enu):
    """
    points_enu: Nx3 in ENU
    R_road: 3x3 from road_frame_axes (columns are basis vectors in ENU)
    origin_enu: 3-vector, origin of road frame in ENU

    Returns:
        coords: Nx3, where:
           u = along-road coordinate (meters)
           v = lateral coordinate (meters)
           w = height (meters, approx)
    """
    rel = points_enu - origin_enu[None, :]
    # R_road^T * rel gives coords in road frame
    coords = (R_road.T @ rel.T).T
    return coords  # (u,v,w)

def apply_road_frame(points):
    """
    Rotate ENU XY into a road-aligned frame where the road runs along +X.
    points: (..., 3) array
    """
    if ROAD_FRAME_ROT is None:
        return points
    pts = np.asarray(points)
    xy = pts[..., :2].reshape(-1, 2).T  # 2 x N
    xy_rot = ROAD_FRAME_ROT @ xy        # 2 x N
    xy_rot = xy_rot.T.reshape(pts[..., :2].shape)
    out = pts.copy()
    out[..., 0:2] = xy_rot
    return out



# ====================================================
# MAIN
# ====================================================
# ====================================================
# MAIN
# ====================================================
def main():
    # Instead of fetch_pano_chain(), we hard-code the Google pano IDs
    # and their corresponding local JPG names.
    panos = LOCAL_PANOS
    for p in panos:
        print("   ", p["local_id"], "->", p["gsv_id"])


    # pick reference
    ref_lat, ref_lon = START_LAT, START_LON
    ref_ecef = llh_to_ecef(ref_lat, ref_lon, 0)

    lat0, lon0 = panos[0]["lat"], panos[0]["lon"]
    latN, lonN = panos[-1]["lat"], panos[-1]["lon"]

    p0_ecef = llh_to_ecef(lat0, lon0, 0.0)
    pN_ecef = llh_to_ecef(latN, lonN, 0.0)

    p0_enu = ecef_to_enu(p0_ecef, ref_lat, ref_lon, 0.0, ref_ecef)
    pN_enu = ecef_to_enu(pN_ecef, ref_lat, ref_lon, 0.0, ref_ecef)

    d = pN_enu - p0_enu  # ENU delta along the road
    road_yaw = math.atan2(d[1], d[0])  # angle from +E axis
    print(f"[cam] global road yaw ≈ {math.degrees(road_yaw):.1f} degrees")

    # rotation so that road direction → +X
    rot_angle = -road_yaw
    c, s = math.cos(rot_angle), math.sin(rot_angle)
    global ROAD_FRAME_ROT
    ROAD_FRAME_ROT = np.array([[c, -s],
                               [s,  c]], dtype=np.float32)

    # load segmentation model
    print("[seg] Loading Fast-SCNN…")
    model = FastSCNNWrapper(FAST_SCNN_WEIGHTS)

    all_pts = []
    all_cols = []
    road_pts = []
    road_cols = []

    for pan in panos:
        # 1. pano: load LOCAL image (u1.jpg / u2.jpg)
        #pano = fetch_full_pano(local_id)
        R_yaw = None

        gsv_id   = pan["gsv_id"]
        local_id = pan["local_id"]
        plat     = pan["lat"]
        plon     = pan["lon"]
        pano = fetch_full_pano(local_id)
        rgb_full = np.asarray(pano)

        MAX_W, MAX_H = 4096, 2048  # or 2048x1024 for low-memory debug

        H_full, W_full = rgb_full.shape[:2]
        scale = min(MAX_W / W_full, MAX_H / H_full, 1.0)
        if scale < 1.0:
            new_w = int(W_full * scale)
            new_h = int(H_full * scale)
            pano_small = pano.resize((new_w, new_h), Image.BILINEAR)
            rgb = np.asarray(pano_small)
        else:
            rgb = rgb_full

        print(f"[debug] rgb stats ({local_id}): min={rgb.min()}, max={rgb.max()}, "
              f"shape={rgb.shape}, dtype={rgb.dtype}")

        # 2. metadata for geo (still from Google pano ID)
        #m = meta_by_pano(gsv_id)
        #plat, plon = m["location"]["lat"], m["location"]["lng"]
        cam_enu = ecef_to_enu(llh_to_ecef(plat, plon, 0),
                              ref_lat, ref_lon, 0, ref_ecef)
        cam_enu[2] += CAM_HEIGHT
        print(f"[cam] {pan['local_id']} cam_enu = {cam_enu}")

        # 3. depth from GSV photometa using the *Google* pano ID
        header, idx, planes = decode_depth_map_json(gsv_id)
        depth_low = compute_spherical_depth(header, idx, planes)
        depth = resize(depth_low, rgb.shape[:2], order=1, mode="reflect")

        # 3.5: depth validity mask
        MAX_DEPTH = 60.0  # meters; tune as needed
        valid_depth = np.isfinite(depth) & (depth > 1.0) & (depth < MAX_DEPTH)

        print(f"[debug] depth stats ({gsv_id}): min={np.nanmin(depth):.3f}, "
              f"max={np.nanmax(depth):.3f}")
        print(f"[debug] valid_depth True: {valid_depth.sum()} of {valid_depth.size}")
        print(f"[debug] {local_id} depth_low shape={depth_low.shape} "
              f"min={depth_low.min():.3f} max={depth_low.max():.3f}")

        # 4. segmentation
        road_mask = segment_road(model, rgb, device="cpu")
        print(f"[debug] road_mask has {road_mask.sum()} positive pixels out of {road_mask.size}")
        Image.fromarray((road_mask * 255).astype(np.uint8)).save(
            os.path.join(OUTPUT_DIR, f"{local_id}_mask.png")
        )

        # Road pixels *and* valid depth
        road_valid_mask = (road_mask == 1) & valid_depth

        # 5. backproject: ALL valid pixels
        pts_all, cols_all = backproject_spherical_masked(
            rgb, depth, cam_enu, valid_depth, R_cam=R_yaw
        )
        all_pts.append(pts_all)
        all_cols.append(cols_all)

        # 5b. backproject: ONLY road pixels with valid depth
        pts_road, cols_road = backproject_spherical_masked(
            rgb, depth, cam_enu, road_valid_mask, R_cam=R_yaw
        )
        road_pts.append(pts_road)
        road_cols.append(cols_road)
        # --- DEBUG: save per-pano clouds ---
        pcd_all = o3d.geometry.PointCloud()
        pcd_all.points = o3d.utility.Vector3dVector(pts_all)
        pcd_all.colors = o3d.utility.Vector3dVector(cols_all)
        o3d.io.write_point_cloud(
            os.path.join(OUTPUT_DIR, f"corridor_{local_id}.ply"),
            pcd_all
        )
        print(f"[debug] wrote corridor_{local_id}.ply with {len(pts_all)} points")


    # Merge
    all_pts = np.concatenate(all_pts)
    all_cols = np.concatenate(all_cols)
    road_pts = np.concatenate(road_pts)
    road_cols = np.concatenate(road_cols)

    # -------------------------------
    # Road plane fitting
    print("[road] fitting road plane…")
    a, b, c = fit_plane_ransac(road_pts)
    print(f"Plane: z = {a:.3f}x + {b:.3f}y + {c:.3f}")

        # --- Road-aligned frame (uses same yaw as earlier) ---
    R_road, t_axis, s_axis, n_axis = road_frame_axes(road_yaw)

    # Use median of road points as road-frame origin
    road_origin = np.median(road_pts, axis=0)
    print(f"[road] origin ENU ≈ {road_origin}")

    # Project road points into road-frame coordinates
    road_coords = project_points_to_road_frame(road_pts, R_road, road_origin)
    u = road_coords[:, 0]   # along road
    v = road_coords[:, 1]   # across road
    w = road_coords[:, 2]   # height-ish (before flattening)

        # --- Rectangular road strip selection ---
    along_half_len = 50.0   # meters forward/back
    half_width     = 8.0    # meters left/right
    height_tol     = 2.0    # meters from road plane

    # distance from road plane in ENU
    plane_z = a * road_pts[:, 0] + b * road_pts[:, 1] + c
    dz = road_pts[:, 2] - plane_z

    road_strip_mask = (
        (u > -along_half_len) & (u < along_half_len) &
        (np.abs(v) < half_width) &
        (np.abs(dz) < height_tol)
    )

    print(f"[road] strip keeps {road_strip_mask.sum()} of {road_strip_mask.size} road points")

    strip_pts = road_pts[road_strip_mask].copy()
    strip_cols = road_cols[road_strip_mask].copy()

    # --- Flatten strip exactly onto the fitted plane ---
    plane_z_strip = a * strip_pts[:, 0] + b * strip_pts[:, 1] + c
    strip_pts[:, 2] = plane_z_strip

        # Save flattened road strip
    print("[pcd] saving flattened road strip…")
    pcd_road = o3d.geometry.PointCloud()
    pcd_road.points = o3d.utility.Vector3dVector(strip_pts)
    pcd_road.colors = o3d.utility.Vector3dVector(strip_cols)
    out_road_ply = os.path.join(OUTPUT_DIR, "road_strip_flat.ply")
    o3d.io.write_point_cloud(out_road_ply, pcd_road)
    print(f"[pcd] wrote {out_road_ply} with {len(strip_pts)} points")

    # -------------------------------
    # Road texture
    print("[road] building high-res texture…")
    build_road_texture(road_pts, road_cols)

    # --------------------------------
    # Build corridor-relative point cloud
    # --------------------------------
    corr_pts, corr_cols = filter_corridor(
        all_pts, all_cols,
        radius_m=60.0,
    )

    # Save *corridor* point cloud instead of the full sphere
    print("[pcd] saving corridor point cloud…")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(corr_pts)
    pcd.colors = o3d.utility.Vector3dVector(corr_cols)
    out_ply = os.path.join(OUTPUT_DIR, "pointcloud_corridor.ply")
    o3d.io.write_point_cloud(out_ply, pcd)
    print(f"[pcd] wrote {out_ply} with {len(corr_pts)} points")

    # ---- DEBUG: verify that colors are present and non-black ----
    cols_np = np.asarray(pcd.colors)
    if cols_np.size == 0:
        print("[debug] WARNING: no colors on pcd")
    else:
        print("[debug] colors min:", cols_np.min(), "max:", cols_np.max())
        print("[debug] first 5 colors:", cols_np[:5])

    # Optional: Poisson mesh on corridor only
    print("[mesh] Poisson reconstruction on corridor…")
    pcd_ds = pcd.voxel_down_sample(0.1)
    pcd_ds.estimate_normals()

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_ds, depth=9
    )

    bbox = pcd_ds.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    print("[mesh] transferring vertex colors from point cloud…")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_ds)

    mesh_colors = []
    for v in mesh.vertices:
        _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
        mesh_colors.append(pcd_ds.colors[idx[0]])

    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

    out_mesh = os.path.join(OUTPUT_DIR, "city_corridor_mesh.ply")
    o3d.io.write_triangle_mesh(out_mesh, mesh)
    print(f"[mesh] wrote colored corridor mesh to {out_mesh}")

    print("DONE.")


if __name__ == "__main__":
    main()
