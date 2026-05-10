"""
Debug script: project a BEV semantic map into a camera view and compare to GT seg.

This uses Occ3D/Occupancy3D `annotations.json` intrinsics + extrinsic, and assumes a
ground-plane intersection in ego coordinates.

It is intended as a *sanity check* for coordinate conventions (axis swap / flips).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


@dataclass(frozen=True)
class GridSpec:
    x_min: float = -40.0
    x_max: float = 40.0
    y_min: float = -40.0
    y_max: float = 40.0
    voxel: float = 0.4
    z_plane: float = 0.0  # ego-frame height used for ground-plane intersection


def _quat_wxyz_to_R(qwxyz: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] -> 3x3 rotation matrix."""
    qw, qx, qy, qz = [float(x) for x in qwxyz]
    # Normalize for safety
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    # Standard formula
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _load_rgb(path: str) -> torch.Tensor:
    """(3,H,W) float in [0,1]."""
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t


def _project_bev_to_cam(
    *,
    bev_rgb: torch.Tensor,  # (3,Hb,Wb) in [0,1]
    K: np.ndarray,  # (3,3)
    extrinsic: Dict,  # {"translation":[3], "rotation":[4 wxyz]}
    out_hw: Tuple[int, int],
    grid: GridSpec,
    mapping_variant: Tuple[bool, bool, bool],
    device: torch.device,
) -> torch.Tensor:
    """
    Returns projected RGB image (3,H,W) in [0,1] by sampling BEV on ground-plane intersections.

    mapping_variant = (swap_xy, flip_x, flip_y) applied when converting ego (x,y) -> bev pixel coords.
    """
    Hb, Wb = int(bev_rgb.shape[1]), int(bev_rgb.shape[2])
    H, W = out_hw

    # Camera pose in ego frame: assume extrinsic gives cam->ego
    t_ego = np.array(extrinsic["translation"], dtype=np.float64).reshape(3)
    R_cam_to_ego = _quat_wxyz_to_R(np.array(extrinsic["rotation"], dtype=np.float64).reshape(4))
    R_ego_to_cam = R_cam_to_ego.T

    K_t = torch.from_numpy(K).to(device=device, dtype=torch.float32)
    Kinv = torch.inverse(K_t)

    # Pixel grid (u,v)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=-1)  # (H,W,3)
    # Rays in camera frame
    rays_cam = torch.matmul(pix, Kinv.T)  # (H,W,3)
    # Rotate to ego frame
    R = torch.from_numpy(R_cam_to_ego).to(device=device, dtype=torch.float32)
    rays_ego = torch.matmul(rays_cam, R.T)  # (H,W,3)

    o = torch.tensor(t_ego, device=device, dtype=torch.float32).view(1, 1, 3)
    rz = rays_ego[..., 2]
    oz = o[..., 2]
    # s for intersection with z = z_plane
    denom = rz
    valid = denom.abs() > 1e-6
    s = (float(grid.z_plane) - oz) / denom.clamp(min=1e-6, max=-1e-6)  # placeholder, overwritten below
    # handle sign properly without branching
    s = (float(grid.z_plane) - oz) / denom
    valid = valid & (s > 0)
    p = o + s.unsqueeze(-1) * rays_ego  # (H,W,3)
    x = p[..., 0]
    y = p[..., 1]

    swap_xy, flip_x, flip_y = mapping_variant
    if swap_xy:
        x, y = y, x
    if flip_x:
        x = -x
    if flip_y:
        y = -y

    # Map meters -> bev pixel coords (col=u_bev, row=v_bev)
    # x spans [x_min,x_max] across Wb; y spans [y_min,y_max] across Hb.
    u_bev = (x - float(grid.x_min)) / float(grid.voxel)
    v_bev = (y - float(grid.y_min)) / float(grid.voxel)

    # grid_sample expects normalized [-1,1], with x=col, y=row.
    gx = (u_bev / (Wb - 1)) * 2.0 - 1.0
    gy = (v_bev / (Hb - 1)) * 2.0 - 1.0
    grid_norm = torch.stack([gx, gy], dim=-1)  # (H,W,2)

    # Sample
    bev = bev_rgb.unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,3,Hb,Wb)
    samp = F.grid_sample(bev, grid_norm.unsqueeze(0), mode="bilinear", padding_mode="zeros", align_corners=True)
    out = samp[0]  # (3,H,W)
    # Mask invalid pixels
    out = out * valid.to(dtype=out.dtype).unsqueeze(0)
    return out.clamp(0, 1)


def _make_side_by_side(gt: torch.Tensor, pred: torch.Tensor, title: str) -> Image.Image:
    """gt/pred: (3,H,W) float [0,1]."""
    gt_np = (gt.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    pr_np = (pred.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gt_im = Image.fromarray(gt_np, mode="RGB")
    pr_im = Image.fromarray(pr_np, mode="RGB")
    w, h = gt_im.size
    canvas = Image.new("RGB", (w * 2, h + 28), color=(10, 10, 10))
    canvas.paste(gt_im, (0, 28))
    canvas.paste(pr_im, (w, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"{title} | left=GT seg, right=proj(BEV)", fill=(240, 240, 240))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bev_root", default="./bev_outputs")
    ap.add_argument("--occ3d_annotations", default="/home/vbontal1/Downloads/Occupancy3D/annotations.json")
    ap.add_argument("--camera", default="CAM_FRONT", choices=CAMERAS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="debug_bev_proj.png")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--z_plane", type=float, default=0.0)
    ap.add_argument("--max_variants", type=int, default=8)
    args = ap.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    bev_dir = os.path.join(args.bev_root, "bev")
    seg_dir = os.path.join(args.bev_root, f"seg_{args.camera}")
    if not os.path.isdir(bev_dir):
        raise FileNotFoundError(f"Missing {bev_dir}")
    if not os.path.isdir(seg_dir):
        raise FileNotFoundError(f"Missing {seg_dir}")

    stems = [os.path.splitext(fn)[0] for fn in os.listdir(bev_dir) if fn.lower().endswith(".png")]
    if not stems:
        raise RuntimeError("No BEV pngs found")
    stem = random.choice(stems)

    with open(args.occ3d_annotations, "r") as f:
        ann = json.load(f)
    scene, frame_token = stem.split("_", 1)
    frame = ann["scene_infos"][scene][frame_token]
    cam = frame["camera_sensor"][args.camera]
    K = np.array(cam["intrinsics"], dtype=np.float64)
    extr = cam["extrinsic"]

    bev_path = os.path.join(bev_dir, f"{stem}.png")
    seg_path = os.path.join(seg_dir, f"{stem}.png")
    bev_rgb = _load_rgb(bev_path)
    gt_seg = _load_rgb(seg_path)
    H, W = int(gt_seg.shape[1]), int(gt_seg.shape[2])

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    grid = GridSpec(z_plane=float(args.z_plane))

    # Try a small set of mapping variants (swap + flips) to find the best alignment automatically.
    variants = []
    for swap_xy in (False, True):
        for flip_x in (False, True):
            for flip_y in (False, True):
                variants.append((swap_xy, flip_x, flip_y))
    variants = variants[: int(args.max_variants)]

    best = None
    best_err = None
    best_var = None
    for var in variants:
        pred = _project_bev_to_cam(
            bev_rgb=bev_rgb,
            K=K,
            extrinsic=extr,
            out_hw=(H, W),
            grid=grid,
            mapping_variant=var,
            device=device,
        )
        err = (pred.to("cpu") - gt_seg).abs().mean().item()
        if best_err is None or err < best_err:
            best_err = err
            best = pred
            best_var = var

    assert best is not None and best_err is not None and best_var is not None
    title = f"stem={stem} cam={args.camera} z={grid.z_plane:.2f} variant(swap_xy,flip_x,flip_y)={best_var} L1={best_err:.4f}"
    out_im = _make_side_by_side(gt_seg, best.to("cpu"), title)
    out_im.save(args.out)
    print("[OK]", title)
    print("[OK] wrote", args.out)


if __name__ == "__main__":
    main()

