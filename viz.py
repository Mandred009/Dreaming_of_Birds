"""Shared colour helpers so every script paints classes the same way."""
import numpy as np

# Target / prediction classes: background, vehicle, pedestrian.
TARGET_COLORS = np.array([[0, 0, 0], [255, 60, 60], [60, 255, 60]], dtype=np.uint8)
# Sparse channels: flat, vehicle, person, obstacle.
SPARSE_COLORS = np.array([[120, 120, 120], [255, 60, 60], [60, 255, 60], [80, 120, 255]], dtype=np.uint8)


def sparse_to_rgb(sparse):
    """Paint classes in order flat -> obstacle -> vehicle -> person so small/important ones stay on top."""
    img = np.zeros(sparse.shape[1:] + (3,), dtype=np.uint8)
    for c in [0, 3, 1, 2]:
        img[sparse[c] > 0] = SPARSE_COLORS[c]
    return img


def label_to_rgb(label):
    """[H,W] int map of {0,1,2} -> RGB image."""
    return TARGET_COLORS[label]
