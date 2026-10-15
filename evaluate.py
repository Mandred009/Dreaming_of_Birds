"""Evaluate baselines on the test split (= mini_val).  Usage: python evaluate.py --config config.yaml

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


def load_diffusion(cfg):
    """Model C: SD 1.5 + the trained ControlNet (checkpoints/diffusion/best).
    Returns None if there is no checkpoint yet."""
    path = os.path.join(cfg["diffusion"]["ckpt_dir"], "best")
    if not os.path.isdir(path):
        return None
    from diffusers import ControlNetModel
    from models.diffusion import BEVDiffusion, encode_prompt
    model = BEVDiffusion(cfg, encode_prompt(cfg))
    model.controlnet = ControlNetModel.from_pretrained(path).to("cuda")   # swap in trained weights
    return model.eval()


def run_diffusion(model, sparse_all, num_samples, bs=4):
    """Predict all frames in small batches (each frame needs num_samples x 25 UNet passes)."""
    return np.concatenate([model.predict(torch.from_numpy(sparse_all[i:i + bs]), num_samples, seed=i).cpu().numpy()
                           for i in range(0, len(sparse_all), bs)])


def load_pixel_diffusion(cfg, device="cuda", refiner=False):
    """Model D (E4) or, with refiner=True, model E (E5): the trained pixel-space denoiser.
    None if there is no checkpoint yet."""
    p = cfg["pixel_diffusion"]
    path = os.path.join(p["refiner_ckpt_dir" if refiner else "ckpt_dir"], "best.pt")
    if not os.path.exists(path):
        return None
    if refiner:
        import copy
        cfg = copy.deepcopy(cfg)
        cfg["pixel_diffusion"]["condition_on_unet"] = True   # 10-channel input
    from models.unet import build_denoiser
    model = build_denoiser(cfg).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()


def run_pixel_diffusion(model, cfg, sparse_all, num_samples, bs=16):
    from models.pixel_diffusion import cosine_alpha_bar, predict
    p = cfg["pixel_diffusion"]
    alpha_bar = cosine_alpha_bar(p["T"]).cuda()
    gen = torch.Generator(device="cuda").manual_seed(p["seed"])
    return np.concatenate([predict(model, torch.from_numpy(sparse_all[i:i + bs]).float().cuda(), alpha_bar,
                                   p["ddim_steps"], num_samples, gen).cpu().numpy()
                           for i in range(0, len(sparse_all), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    val = BEVDataset(cfg, "test", augment=False)

    # (name, function). Hook: append e.g. ("Diffusion", make_diffusion_method(cfg)) here.
    methods = [("A: projection only", projection_only),
               ("A+: projection + 3x3 dilation", projection_dilated),
               ("B: U-Net", make_unet_method(cfg))]

    preds = {name: np.stack([fn(s) for s in val.sparse]) for name, fn in methods}

    # Model C: loaded only if a checkpoint exists; 1 sample and k-sample vote; freed afterwards.
    model = load_diffusion(cfg)
    if model is not None:
        k_vote = cfg["diffusion"]["vote_samples"]
        for name, k in [("C: diffusion (1 sample)", 1), (f"C: diffusion ({k_vote}-sample vote)", k_vote)]:
            preds[name] = run_diffusion(model, val.sparse, k)
            methods.append((name, None))
        del model
        torch.cuda.empty_cache()

    # Model D: pixel diffusion; 1 sample and k-sample mean of softmax probabilities.
    model = load_pixel_diffusion(cfg)
    if model is not None:
        k_mean = cfg["pixel_diffusion"]["num_samples"]
        for name, k in [("D: pixel diffusion (1 sample)", 1), (f"D: pixel diffusion ({k_mean}-sample mean)", k_mean)]:
            preds[name] = run_pixel_diffusion(model, cfg, val.sparse, k)
            methods.append((name, None))
        del model
        torch.cuda.empty_cache()

    # Model E (E5): diffusion refiner conditioned on B's softmax. Its condition = sparse + B probs.
    model = load_pixel_diffusion(cfg, refiner=True)
    if model is not None:
        from models.pixel_diffusion import add_unet_probs
        cond = add_unet_probs(cfg, val.sparse)
        k_mean = cfg["pixel_diffusion"]["num_samples"]
        for name, k in [("E: diffusion refiner on U-Net (1 sample)", 1), (f"E: diffusion refiner on U-Net ({k_mean}-sample mean)", k_mean)]:
            preds[name] = run_pixel_diffusion(model, cfg, cond, k)
            methods.append((name, None))
        del model
        torch.cuda.empty_cache()

    lines = ["| Method | Vehicle IoU | Pedestrian IoU | mIoU |", "|---|---|---|---|"]
    for name, _ in methods:
        m = summarize(confusion_matrix(preds[name], val.target, 3))
        lines.append(f"| {name} | {m['vehicle']:.3f} | {m['pedestrian']:.3f} | {m['miou']:.3f} |")
    table = "\n".join(lines)
    print(table)

    out_dir = cfg["eval"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(f"mIoU = mean of vehicle and pedestrian IoU, on test = mini_val ({len(val)} samples)\n\n{table}\n")

    # Qualitative figure: evenly spaced test samples; columns = sparse | each method | target.
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

    # Qualitative figure with diffusion: sparse | U-Net | diffusion | target.
    if "C: diffusion (1 sample)" in preds:
        cols = ["sparse input", "U-Net", "diffusion (1 sample)", "target"]
        fig, axes = plt.subplots(k, 4, figsize=(16, 4 * k))
        for r, i in enumerate(ids):
            imgs = [sparse_to_rgb(val.sparse[i]), label_to_rgb(preds["B: U-Net"][i]),
                    label_to_rgb(preds["C: diffusion (1 sample)"][i]), label_to_rgb(val.target[i])]
            for c, (img, title) in enumerate(zip(imgs, cols)):
                axes[r, c].imshow(img)
                axes[r, c].axis("off")
                if r == 0:
                    axes[r, c].set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "qualitative_diffusion.png"), dpi=70)

    # Qualitative figure with pixel diffusion: sparse | U-Net | pixel diffusion | target.
    if "D: pixel diffusion (1 sample)" in preds:
        cols = ["sparse input", "U-Net", "pixel diffusion (1 sample)", "target"]
        fig, axes = plt.subplots(k, 4, figsize=(16, 4 * k))
        for r, i in enumerate(ids):
            imgs = [sparse_to_rgb(val.sparse[i]), label_to_rgb(preds["B: U-Net"][i]),
                    label_to_rgb(preds["D: pixel diffusion (1 sample)"][i]), label_to_rgb(val.target[i])]
            for c, (img, title) in enumerate(zip(imgs, cols)):
                axes[r, c].imshow(img)
                axes[r, c].axis("off")
                if r == 0:
                    axes[r, c].set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "qualitative_pixel_diffusion.png"), dpi=70)

    # Qualitative figure with the refiner: sparse | U-Net | refiner | target.
    if "E: diffusion refiner on U-Net (1 sample)" in preds:
        cols = ["sparse input", "U-Net", "refiner (1 sample)", "target"]
        fig, axes = plt.subplots(k, 4, figsize=(16, 4 * k))
        for r, i in enumerate(ids):
            imgs = [sparse_to_rgb(val.sparse[i]), label_to_rgb(preds["B: U-Net"][i]),
                    label_to_rgb(preds["E: diffusion refiner on U-Net (1 sample)"][i]), label_to_rgb(val.target[i])]
            for c, (img, title) in enumerate(zip(imgs, cols)):
                axes[r, c].imshow(img)
                axes[r, c].axis("off")
                if r == 0:
                    axes[r, c].set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "qualitative_refiner.png"), dpi=70)


if __name__ == "__main__":
    main()
