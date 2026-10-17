"""Shared colour helpers so every script paints classes the same way."""
import numpy as np

# Sparse channels : road, sidewalk, terrain, vehicle, person, obstacle.
SPARSE_COLORS = np.array([[120, 120, 120], [190, 190, 255], [150, 110, 60], [255, 60, 60],
                          [60, 255, 60], [80, 120, 255]], dtype=np.uint8)
SPARSE_ORDER = [0, 1, 2, 5, 3, 4]   # paint order: small/important classes last, so they stay on top
# Target layers: drivable_area, ped_crossing, walkway, divider, vehicle, pedestrian (multi-label).
LAYER_COLORS = np.array([[70, 70, 90], [240, 240, 240], [120, 180, 120], [255, 200, 0],
                         [255, 60, 60], [60, 255, 60]], dtype=np.uint8)
LAYER_ORDER = [0, 2, 1, 3, 4, 5]    # drivable < walkway < ped_crossing < divider < vehicle < pedestrian


def sparse_to_rgb(sparse):
    """Paint the 6 sparse channels in SPARSE_ORDER (later = on top)."""
    img = np.zeros(sparse.shape[1:] + (3,), dtype=np.uint8)
    for c in SPARSE_ORDER:
        img[sparse[c] > 0] = SPARSE_COLORS[c]
    return img


def target_to_rgb(target):
    """[6,H,W] binary layers (overlapping allowed) -> RGB, painted in LAYER_ORDER."""
    img = np.zeros(target.shape[1:] + (3,), dtype=np.uint8)
    for c in LAYER_ORDER:
        img[target[c] > 0] = LAYER_COLORS[c]
    return img

