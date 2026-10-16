"""Pure numpy geometry helpers for building BEV maps.

CONVENTION (used everywhere in this project)
  Ego frame: x = forward, y = left, z = up (nuScenes convention).
  BEV image: row 0 is the FAR FORWARD edge (+x), column 0 is the LEFT edge (+y).
  So the ego car sits at the image centre and "forward" points UP in the picture.
    row = (x_max - x) / resolution
    col = (y_max - y) / resolution
"""
import numpy as np
import cv2

# Cityscapes train id (0..18) -> coarse class (sparse input v2, 6 channels), -1 = ignore (sky).
# 0 road  1 sidewalk  2 terrain  3 vehicle  4 person  5 obstacle (building, wall, fence, pole, light, sign, vegetation)
# (v1 had 4 channels with road/sidewalk/terrain merged into one "flat"; see git history.)
CITYSCAPES_TO_COARSE = np.array([
    0,   # 0 road
    1,   # 1 sidewalk
    5,   # 2 building
    5,   # 3 wall
    5,   # 4 fence
    5,   # 5 pole
    5,   # 6 traffic light
    5,   # 7 traffic sign
    5,   # 8 vegetation
    2,   # 9 terrain
    -1,  # 10 sky
    4,   # 11 person
    4,   # 12 rider
    3,   # 13 car
    3,   # 14 truck
    3,   # 15 bus
    3,   # 16 train
    3,   # 17 motorcycle
    3,   # 18 bicycle
], dtype=np.int64)
NUM_COARSE = 6


def metres_to_cells(x, y, grid):
    """Convert ego-frame metres (x forward, y left) to float (row, col) BEV coordinates."""
    res = grid["resolution"]
    row = (grid["x_range"][1] - x) / res   # larger x (further forward) -> smaller row
    col = (grid["y_range"][1] - y) / res   # larger y (further left) -> smaller col
    return row, col


def boxes_to_bev(boxes_in_ego, grid):
    """Rasterise 3D boxes into binary BEV masks (multi-label: vehicles and pedestrians may overlap).

    boxes_in_ego: list of dicts with keys
        center (x, y), size (width, length), yaw (rad, in ego frame),
        category (nuScenes name), num_pts (lidar + radar points inside).
    Returns uint8 [2, size, size]: channel 0 vehicle, channel 1 pedestrian.
    """
    n = grid["size"]
    bev = np.zeros((2, n, n), dtype=np.uint8)

    for ch, prefix in enumerate(["vehicle.", "human.pedestrian."]):
        for b in boxes_in_ego:
            if not b["category"].startswith(prefix):
                continue
            if b["num_pts"] == 0:   # no lidar/radar hits: not observable, skip
                continue
            w, l = b["size"]
            cx, cy = b["center"]
            yaw = b["yaw"]
            # Footprint corners in the box's own frame (x along length, y along width) ...
            local = np.array([[ l / 2,  w / 2], [ l / 2, -w / 2],
                              [-l / 2, -w / 2], [-l / 2,  w / 2]])
            # ... rotated by yaw, then moved to the box centre (2D rotation matrix).
            c, s = np.cos(yaw), np.sin(yaw)
            rot = np.array([[c, -s], [s, c]])
            corners = local @ rot.T + np.array([cx, cy])
            row, col = metres_to_cells(corners[:, 0], corners[:, 1], grid)
            # cv2 wants integer (x=col, y=row) points.
            poly = np.stack([col, row], axis=1).round().astype(np.int32)
            cv2.fillPoly(bev[ch], [poly], 1)
    return bev


def depth_to_ego_points(depth_m, K, cam_to_ego, stride=4):
    """Unproject a depth map to 3D points in the ego frame.

    depth_m: [H, W] metric depth along the camera's optical axis (z in camera frame).
    K: 3x3 intrinsics. cam_to_ego: 4x4 camera->ego transform.
    Returns (points [N,3], v [N], u [N]) where (v, u) are the pixel indices used,
    so the caller can look up the segmentation label for each point.
    """
    h, w = depth_m.shape
    # Subsample pixels: every `stride`-th row/col. Full-res would be 1.4M points per camera.
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    v, u = v.reshape(-1), u.reshape(-1)
    d = depth_m[v, u]
    # Pinhole model: u = fx*X/Z + cx  =>  X = (u - cx) * Z / fx   (same for Y with v, fy)
    x = (u - K[0, 2]) * d / K[0, 0]
    y = (v - K[1, 2]) * d / K[1, 1]
    pts_cam = np.stack([x, y, d, np.ones_like(d)], axis=1)   # homogeneous [N,4]
    pts_ego = pts_cam @ cam_to_ego.T                          # apply the 4x4 transform
    return pts_ego[:, :3], v, u


def points_to_sparse_bev(points_ego, point_labels, grid, z_range=(-1.0, 3.0), max_depth=50.0,
                         num_classes=NUM_COARSE):
    """Drop points into BEV cells as binary occupancy per coarse class.

    points_ego: [N,3]; point_labels: [N] coarse class ids (-1 = ignore, e.g. sky).
    The max_depth filter is applied here on horizontal distance from the ego origin.
    Returns uint8 [num_classes, size, size]; 1 where at least one point fell in the cell.
    """
    n = grid["size"]
    sparse = np.zeros((num_classes, n, n), dtype=np.uint8)

    x, y, z = points_ego[:, 0], points_ego[:, 1], points_ego[:, 2]
    keep = (point_labels >= 0) & (z >= z_range[0]) & (z <= z_range[1]) \
           & (np.hypot(x, y) < max_depth)
    x, y, labels = x[keep], y[keep], point_labels[keep]

    row, col = metres_to_cells(x, y, grid)
    row, col = np.floor(row).astype(np.int64), np.floor(col).astype(np.int64)
    inside = (row >= 0) & (row < n) & (col >= 0) & (col < n)   # points beyond the grid are dropped
    sparse[labels[inside], row[inside], col[inside]] = 1
    return sparse
