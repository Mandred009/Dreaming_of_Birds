"""Evaluate on the test split (v2: 6 BEV layers).  Usage: python evaluate.py --config config.yaml

Rows: A projection, B U-Net, D pixel diffusion (1 and 4 samples), F hybrid DiT (1 and 4 samples).
Every method is a function  (sparse [B,6,H,W] float cuda, cams or None) -> probabilities [B,6,H,W];
metrics.evaluate_model thresholds at 0.5 and accumulates per-layer TP/FP/FN.

Model C (SD1.5 + ControlNet) and E (diffusion refiner on B) were 3-class v1 experiments. They are not
ported to the 6-layer data and are not evaluated here; their code stays in git history and their
results in EXPERIMENTS.md and results/*.md (v1).
"""
import argparse
import os
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import BEVDataset
from metrics import LAYER_NAMES, evaluate_model
from models.dit import build_dit
from models.pixel_diffusion import cosine_alpha_bar, predict_probs
from models.unet import build_denoiser, build_unet
from viz import sparse_to_rgb, target_to_rgb


def projection_only(sparse, cams=None):
    """Baseline A: trust the projected camera labels. sparse channels: 0 road, 1 sidewalk, 2 terrain,
    3 vehicle, 4 person, 5 obstacle. Target layers: 0 drivable<-road, 1 crossing<-empty,
    2 walkway<-sidewalk, 3 divider<-empty, 4 vehicle<-vehicle, 5 pedestrian<-person."""
    zeros = torch.zeros_like(sparse[:, 0])
    return torch.stack([sparse[:, 0], zeros, sparse[:, 1], zeros, sparse[:, 3], sparse[:, 4]], dim=1)


def load_unet(cfg, device):
    path = os.path.join(cfg["train"]["ckpt_dir"], "best.pt")
    if not os.path.exists(path):
        return None
    model = build_unet(cfg).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return lambda sparse, cams: torch.sigmoid(model(sparse))


def load_diffusion(cfg, kind, device, num_samples):
    """kind 'D' (pixel U-Net denoiser) or 'F' (hybrid DiT). Returns a method function, or None if no checkpoint."""
    p = cfg["pixel_diffusion"]
    path = os.path.join(p["ckpt_dir"] if kind == "D" else cfg["dit"]["ckpt_dir"], "best.pt")
    if not os.path.exists(path):
        return None
    model = (build_denoiser(cfg) if kind == "D" else build_dit(cfg)).to(device)
    model.load_state_dict(torch.load(path, map_location=device))   # DiT best.pt holds the EMA weights
    model.eval()
    alpha_bar = cosine_alpha_bar(p["T"]).to(device)
    gen = torch.Generator(device=device).manual_seed(p["seed"])

    def fn(sparse, cams):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(kind == "F")):
            return predict_probs(model, sparse, alpha_bar, p["ddim_steps"], num_samples, gen, cams)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    device = "cuda"
    k_mean = cfg["pixel_diffusion"]["num_samples"]
    use_cams = cfg["dit"]["use_cameras"] and os.path.exists(os.path.join(cfg["dit"]["ckpt_dir"], "best.pt"))
    test = BEVDataset(cfg, "test", augment=False, with_cams=use_cams)
    loader = DataLoader(test, 16, num_workers=0)

    candidates = [("A: projection only", projection_only),
                  ("B: U-Net", load_unet(cfg, device)),
                  ("D: pixel diffusion (1 sample)", load_diffusion(cfg, "D", device, 1)),
                  (f"D: pixel diffusion ({k_mean}-sample mean)", load_diffusion(cfg, "D", device, k_mean)),
                  ("F: hybrid DiT (1 sample)", load_diffusion(cfg, "F", device, 1)),
                  (f"F: hybrid DiT ({k_mean}-sample mean)", load_diffusion(cfg, "F", device, k_mean))]
    k = cfg["eval"]["num_qualitative"]
    ids = set(np.linspace(0, len(test) - 1, k).astype(int).tolist())
    results, preds = {}, {}
    for name, fn in candidates:
        if fn is None:
            print(f"skip {name}: no checkpoint")
            continue
        results[name], preds[name] = evaluate_model(fn, loader, device, ids)
        print(name, {a: round(b, 3) for a, b in results[name].items()}, flush=True)

    cols = LAYER_NAMES + ["map_mIoU", "obj_mIoU", "mIoU"]
    lines = ["| Method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, m in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")
    table = "\n".join(lines)
    out_dir = cfg["eval"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(f"Per-layer IoU at threshold 0.5 on test ({len(test)} frames); map_mIoU = layers 0-3, "
                f"obj_mIoU = 4-5, mIoU = all 6.\n\n{table}\n")
    print(table)

    # Qualitative figure: sparse | B | D (1 sample) | DiT (1 sample) | target, for evenly spaced test frames.
    shown = [n for n in ("B: U-Net", "D: pixel diffusion (1 sample)", "F: hybrid DiT (1 sample)") if n in preds]
    if not shown:
        return
    ids = sorted(ids)
    fig, axes = plt.subplots(len(ids), len(shown) + 2, figsize=(4 * (len(shown) + 2), 4 * len(ids)), squeeze=False)
    for r, i in enumerate(ids):
        sparse, target = test[i][0].numpy(), test[i][1].numpy()
        imgs = [sparse_to_rgb(sparse)] + [target_to_rgb(preds[n][i]) for n in shown] + [target_to_rgb(target)]
        for c, (img, title) in enumerate(zip(imgs, ["sparse input"] + shown + ["target"])):
            axes[r, c].imshow(img)
            axes[r, c].plot(100, 100, "w+")   # ego car
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "qualitative.png"), dpi=70)


if __name__ == "__main__":
    main()
