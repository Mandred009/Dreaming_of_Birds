"""Evaluate baselines on mini_val.  Usage: python evaluate.py --config config.yaml

Every method is a function  sparse [4,H,W] uint8 numpy -> prediction [H,W] int in {0,1,2}.
To add a method later (e.g. diffusion), write such a function and add it to METHODS.
"""
import argparse
import os
import cv2
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import BEVDataset
from metrics import confusion_matrix, summarize
from models.unet import build_unet
from viz import sparse_to_rgb, label_to_rgb


def projection_only(sparse):
    """Baseline A: trust the projected camera labels. Pedestrian drawn over vehicle."""
    pred = np.zeros(sparse.shape[1:], dtype=np.uint8)
    pred[sparse[1] == 1] = 1
    pred[sparse[2] == 1] = 2
    return pred


def projection_dilated(sparse):
    """Baseline A+: same, but each projected point is grown with a 3x3 dilation so the sparse
    dots cover a bit more of the object footprint."""
    kernel = np.ones((3, 3), np.uint8)
    grown = np.stack([cv2.dilate(sparse[c], kernel) for c in range(sparse.shape[0])])
    return projection_only(grown)


def make_unet_method(cfg, device="cuda"):
    model = build_unet(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg["train"]["ckpt_dir"], "best.pt"), map_location=device))
    model.eval()

    @torch.no_grad()
    def predict(sparse):
        x = torch.from_numpy(sparse).float()[None].to(device)
        return model(x).argmax(1)[0].cpu().numpy()
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    val = BEVDataset(cfg, "val", augment=False)

    # (name, function). Hook: append e.g. ("Diffusion", make_diffusion_method(cfg)) here.
    methods = [("A: projection only", projection_only),
               ("A+: projection + 3x3 dilation", projection_dilated),
               ("B: U-Net", make_unet_method(cfg))]

    preds = {name: np.stack([fn(s) for s in val.sparse]) for name, fn in methods}

    lines = ["| Method | Vehicle IoU | Pedestrian IoU | mIoU |", "|---|---|---|---|"]
    for name, _ in methods:
        m = summarize(confusion_matrix(preds[name], val.target, 3))
        lines.append(f"| {name} | {m['vehicle']:.3f} | {m['pedestrian']:.3f} | {m['miou']:.3f} |")
    table = "\n".join(lines)
    print(table)

    out_dir = cfg["eval"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(f"mIoU = mean of vehicle and pedestrian IoU, on mini_val ({len(val)} samples)\n\n{table}\n")

    # Qualitative figure: evenly spaced val samples; columns = sparse | each method | target.
    k = cfg["eval"]["num_qualitative"]
    ids = np.linspace(0, len(val) - 1, k).astype(int)
    names = [n for n, _ in methods if n in ("A: projection only", "B: U-Net")] or [methods[0][0]]
    cols = ["sparse input"] + names + ["target"]
    fig, axes = plt.subplots(k, len(cols), figsize=(4 * len(cols), 4 * k))
    for r, i in enumerate(ids):
        imgs = [sparse_to_rgb(val.sparse[i])] + [label_to_rgb(preds[n][i]) for n in names] + [label_to_rgb(val.target[i])]
        for c, (img, title) in enumerate(zip(imgs, cols)):
            axes[r, c].imshow(img)
            axes[r, c].plot(100, 100, "w+")   # ego car
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "qualitative_unet.png"), dpi=70)


if __name__ == "__main__":
    main()
