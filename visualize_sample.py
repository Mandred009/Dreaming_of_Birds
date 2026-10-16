"""Plot 6 camera images + sparse BEV + target BEV for one sample token.

Usage: python visualize_sample.py <sample_token> <out.png>
"""
import os
import sys
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from nuscenes.nuscenes import NuScenes

from dataset import dataset_cfg
from viz import TARGET_COLORS, sparse_to_rgb


def main(token, out_path):
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))
    dcfg = dataset_cfg(cfg)
    dataroot = os.path.expanduser(dcfg["dataroot"])
    data = np.load(os.path.join(os.path.expanduser(dcfg["out_dir"]), token + ".npz"))
    nusc = NuScenes(version=dcfg["version"], dataroot=dataroot, verbose=False)
    sample = nusc.get("sample", token)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    # Top row: front-left, front, front-right. Middle row: back-left, back, back-right.
    layout = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
    for ax, cam in zip(axes[:2].reshape(-1), layout):
        path = nusc.get_sample_data_path(sample["data"][cam])
        ax.imshow(Image.open(path))
        ax.set_title(cam)
    extent = [-50, 50, -50, 50]  # just for axis labels; image row 0 is top = +x
    sp, tg = sparse_to_rgb(data["sparse"]), TARGET_COLORS[data["target"]]
    for ax, img, title in zip(axes[2], [sp, tg, np.maximum(sp // 2, tg)],
                              ["sparse BEV", "target BEV", "overlay (target bright over sparse)"]):
        ax.imshow(img)
        ax.plot(100, 100, "w+", markersize=14)   # ego at the centre
        ax.set_title(title + "  (up = forward)")
    for ax in axes.reshape(-1):
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=60)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
