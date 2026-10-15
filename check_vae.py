"""Upper bound for model C: round-trip every test target through the SD VAE only
(paint -> encode -> decode -> snap to palette -> class map) and measure IoU vs the true target.
Usage: python check_vae.py --config config.yaml"""
import argparse
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import BEVDataset
from metrics import confusion_matrix, summarize
from models.diffusion import paint, snap, load_vae, encode_images, decode_latents
from viz import label_to_rgb

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
cfg = yaml.safe_load(open(ap.parse_args().config))
ds = BEVDataset(cfg, "test")
vae = load_vae(cfg)
target = torch.from_numpy(ds.target)

for size in [256, 512]:
    preds = []
    for i in range(0, len(target), 8):
        img = paint(target[i:i + 8].cuda(), size)
        rec = decode_latents(vae, encode_images(vae, img))
        preds.append(snap(rec).cpu())
    preds = torch.cat(preds).numpy()
    m = summarize(confusion_matrix(preds, ds.target, 3))
    print(f"size {size}: vehicle {m['vehicle']:.3f} ped {m['pedestrian']:.3f} mIoU {m['miou']:.3f}", flush=True)
    if size == 256:
        ids = np.linspace(0, len(ds) - 1, 4).astype(int)
        fig, ax = plt.subplots(2, 4, figsize=(16, 8))
        for c, i in enumerate(ids):
            ax[0, c].imshow(label_to_rgb(ds.target[i])); ax[0, c].set_title("target")
            ax[1, c].imshow(label_to_rgb(preds[i])); ax[1, c].set_title("VAE round trip 256")
        for a in ax.ravel():
            a.axis("off")
        plt.tight_layout()
        plt.savefig("results/vae_roundtrip.png", dpi=70)
print("peak VRAM MiB", torch.cuda.max_memory_allocated() / 2**20)
