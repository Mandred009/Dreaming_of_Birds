"""Frozen DINOv2-small camera tokens + per-token ray info for the hybrid DiT.  Usage: python extract_image_features.py --config config.yaml

For every frame of the BEV cache: 6 camera images -> DINOv2-small (fp16, frozen) at 224x392
-> 16x28 patch tokens (384-d) -> 2x2 average pool -> 8x14 = 112 tokens per camera, 672 per frame.
Saved as <token>_img.npz in cfg['img_feat_dir'] (uncompressed: fp16 features do not compress, and
reading is faster):
  feats [672,384] fp16
  rays  [672,6]  fp32  = camera centre in the ego frame (3) + unit ray direction in the ego frame (3)
                         through the centre of the image region the token covers.

Why rays (the PETR idea, Liu et al. 2022): the DiT never projects anything itself, so it has to learn
where each camera token looks. Giving every token its 3D position embedding (origin + ray direction
in the ego frame) lets the BEV queries attend to the tokens whose rays pass near them, without
needing depth. The camera-id embedding added in the model tells apart tokens from different cameras.

Needs NuScenes metadata for the calibration (mini only for now: the trainval metadata takes ~5 GB RAM).
Resumable. The 224x392 resize also changes the intrinsics: K is scaled by (392/W, 224/H).
"""
import argparse
import csv
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from transformers import Dinov2Model

from dataset import dataset_cfg

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def token_rays(K, rotation, translation, orig_hw, img_hw, patch, pool):
    """Ray info [N,6] for the pooled token grid of one camera.
    A pooled token covers (patch*pool)^2 pixels of the resized image; its centre is
    ((j + 0.5) * patch*pool, (i + 0.5) * patch*pool). Unproject that pixel with the scaled K to a
    camera-frame direction, rotate to the ego frame (R of cam->ego), normalise."""
    H, W = orig_hw
    h, w = img_hw
    K = K.copy()
    K[0] *= w / W   # fx, skew, cx scale with the width ratio
    K[1] *= h / H   # fy, cy scale with the height ratio
    cell = patch * pool
    v, u = np.meshgrid((np.arange(h // cell) + 0.5) * cell, (np.arange(w // cell) + 0.5) * cell, indexing="ij")
    pix = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], axis=1)          # [N,3] homogeneous pixels
    d_cam = pix @ np.linalg.inv(K).T                                          # direction with z = 1
    d_ego = d_cam @ Quaternion(rotation).rotation_matrix.T
    d_ego /= np.linalg.norm(d_ego, axis=1, keepdims=True)
    origin = np.tile(np.asarray(translation), (len(d_ego), 1))
    return np.concatenate([origin, d_ego], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    dcfg, icfg = dataset_cfg(cfg), cfg["img_features"]
    dataroot, cache = os.path.expanduser(dcfg["dataroot"]), os.path.expanduser(dcfg["out_dir"])
    out_dir = os.path.expanduser(dcfg["img_feat_dir"])
    os.makedirs(out_dir, exist_ok=True)
    h, w = icfg["image_hw"]
    patch, pool = 14, icfg["pool"]
    with open(os.path.join(cache, "index.csv")) as f:
        tokens = [r["token"] for r in csv.DictReader(f)]
    per_frame = 6 * (h // patch // pool) * (w // patch // pool) * 384 * 2 + 6 * (h // patch // pool) * (w // patch // pool) * 6 * 4
    print(f"{len(tokens)} frames; ~{per_frame / 2**20:.2f} MiB/frame -> ~{per_frame * len(tokens) / 2**30:.2f} GiB total "
          f"(13.5k trainval frames would need ~{per_frame * 13500 / 2**30:.1f} GiB)", flush=True)

    nusc = NuScenes(version=dcfg["version"], dataroot=dataroot, verbose=False)
    model = Dinov2Model.from_pretrained(icfg["model"]).half().cuda().eval()
    mean, std = torch.tensor(MEAN).view(1, 3, 1, 1).cuda().half(), torch.tensor(STD).view(1, 3, 1, 1).cuda().half()

    start = time.time()
    for n, tok in enumerate(tokens):
        path = os.path.join(out_dir, tok + "_img.npz")
        if os.path.exists(path):
            continue
        sample = nusc.get("sample", tok)
        imgs, rays = [], []
        for cam in cfg["cameras"]:
            sd = nusc.get("sample_data", sample["data"][cam])
            calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            im = Image.open(os.path.join(dataroot, sd["filename"])).convert("RGB")
            rays.append(token_rays(np.array(calib["camera_intrinsic"]), calib["rotation"], calib["translation"],
                                   (im.height, im.width), (h, w), patch, pool))
            imgs.append(np.asarray(im.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0)
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).cuda().half()
        with torch.no_grad():
            tokens_out = model(pixel_values=(x - mean) / std).last_hidden_state[:, 1:]   # drop CLS -> [6,16*28,384]
            grid = tokens_out.reshape(6, h // patch, w // patch, -1).permute(0, 3, 1, 2)  # [6,384,16,28]
            grid = F.avg_pool2d(grid, pool)                                              # [6,384,8,14]
            feats = grid.flatten(2).transpose(1, 2).reshape(-1, grid.shape[1])           # [672,384], camera-major
        np.savez(path + ".tmp.npz", feats=feats.cpu().numpy().astype(np.float16), rays=np.concatenate(rays))
        os.replace(path + ".tmp.npz", path)
        if n % 50 == 0:
            print(f"{n}/{len(tokens)} {time.time() - start:.0f}s", flush=True)
    print(f"done in {time.time() - start:.0f}s; peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
