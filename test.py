import argparse
import math
import os
import random
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw

from data_loader import BEVDataset, CAMERAS


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """
    Convert a tensor in [0,1] with shape (C,H,W) to a PIL Image.
    Supports C=1 (L) and C=3 (RGB).
    """
    if t.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(t.shape)}")
    c, h, w = t.shape
    t = t.detach().cpu().clamp(0, 1)
    if c == 1:
        arr = (t[0] * 255.0).to(torch.uint8).numpy()
        return Image.fromarray(arr, mode="L").convert("RGB")
    if c == 3:
        arr = (t.permute(1, 2, 0) * 255.0).to(torch.uint8).numpy()
        return Image.fromarray(arr, mode="RGB")
    raise ValueError(f"Unsupported channels: {c}")


def make_collage(
    items: List[Tuple[str, Image.Image]],
    *,
    cols: int = 4,
    pad: int = 10,
    title_h: int = 18,
    bg: Tuple[int, int, int] = (20, 20, 20),
) -> Image.Image:
    if len(items) == 0:
        raise ValueError("No items to collage")

    w, h = items[0][1].size
    for _, im in items:
        if im.size != (w, h):
            raise ValueError("All images must have the same size for collage")

    rows = math.ceil(len(items) / cols)
    cell_w = w + 2 * pad
    cell_h = h + 2 * pad + title_h

    out_w = cols * cell_w
    out_h = rows * cell_h
    canvas = Image.new("RGB", (out_w, out_h), color=bg)
    draw = ImageDraw.Draw(canvas)

    for i, (title, im) in enumerate(items):
        r = i // cols
        c = i % cols
        x0 = c * cell_w + pad
        y0 = r * cell_h + pad
        draw.text((x0, y0), title, fill=(230, 230, 230))
        canvas.paste(im, (x0, y0 + title_h))

    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="Dataset root (defaults to ./bev_outputs if present)")
    ap.add_argument("--occ3d", default=None, help="Occupancy3D base dir (where annotations.json and imgs/ live)")
    ap.add_argument("--ann", default=None, help="Path to annotations.json (optional)")
    ap.add_argument("--out", default="sample_collage.png", help="Output collage path")
    ap.add_argument("--resolution", type=int, default=None, help="Resize loaded images to NxN before collaging")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    ds = BEVDataset(
        root=args.root,
        resolution=args.resolution,
        return_paths=True,
        load_rgb=True,
        occ3d_base_dir=args.occ3d,
        annotations_json=args.ann,
    )
    idx = random.randrange(len(ds))
    sample = ds[idx]

    items: List[Tuple[str, Image.Image]] = []
    items.append(("BEV", tensor_to_pil(sample["bev"])))

    depth = sample["depth"]  # (6,1,H,W)
    seg = sample["seg"]      # (6,3,H,W)
    rgb = sample["rgb"]      # (6,3,H,W)
    for i, cam in enumerate(CAMERAS):
        items.append((f"DEPTH {cam}", tensor_to_pil(depth[i])))
    for i, cam in enumerate(CAMERAS):
        items.append((f"SEG {cam}", tensor_to_pil(seg[i])))
    for i, cam in enumerate(CAMERAS):
        items.append((f"RGB {cam}", tensor_to_pil(rgb[i])))

    collage = make_collage(items, cols=4)
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    collage.save(out_path)

    print(f"Saved collage for idx={idx} stem={sample['stem']} -> {out_path}")
    if "paths" in sample:
        print("BEV path:", sample["paths"]["bev"])


if __name__ == "__main__":
    main()

