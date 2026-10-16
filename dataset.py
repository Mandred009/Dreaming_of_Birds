"""BEV dataset: loads the cached .npz files fully into RAM (the whole cache is tiny)."""
import csv
import os
import numpy as np
import torch
from torch.utils.data import Dataset


def dataset_cfg(cfg):
    """dataroot/version/out_dir/data for the dataset chosen by cfg['dataset'] ('mini' or 'trainval')."""
    block = cfg if cfg.get("dataset", "mini") == "mini" else cfg[cfg["dataset"]]
    return {k: block[k] for k in ("dataroot", "version", "out_dir", "data", "img_feat_dir")}


def trainval_val_scenes(rows, frac, seed):
    """Seeded ~frac of the available official-train scenes (by scene), sorted."""
    scenes = sorted({r["scene_name"] for r in rows if r["split"] == "train"})
    n = max(1, round(frac * len(scenes)))
    return sorted(np.random.RandomState(seed).choice(scenes, n, replace=False).tolist())


def dcfg_cells(cfg):
    return cfg["grid"]["size"] ** 2


class BEVDataset(Dataset):
    def __init__(self, cfg, split, augment=False, with_cams=False):
        """split: 'train', 'val' or 'test'. Items: (sparse [6,H,W] float, target [6,H,W] float {0,1}[, feats, rays]).
        augment is only honoured if cfg['train']['augment_flip'] is on. with_cams adds the DINOv2 tokens.

        The split mapping lives here and only here. index.csv labels frames mini_train/mini_val
        (called 'train'/'val' there). Selecting the best checkpoint and early stopping on the
        same split we report would leak the test set into model selection. So:
          train = index 'train' minus the held-out scene cfg['data']['val_scene']
          val   = that held-out scene (model selection / early stopping, used by train.py)
          test  = index 'val' = mini_val (reported numbers only, used by evaluate.py)
        """
        dcfg = dataset_cfg(cfg)
        out_dir = os.path.expanduser(dcfg["out_dir"])
        with open(os.path.join(out_dir, "index.csv")) as f:
            all_rows = list(csv.DictReader(f))
        # val scenes: mini = the one held-out scene; trainval = seeded fraction of official-train scenes
        if "val_scene" in dcfg["data"]:
            val_scenes = {dcfg["data"]["val_scene"]}
        else:
            val_scenes = set(trainval_val_scenes(all_rows, dcfg["data"]["val_fraction"], dcfg["data"]["val_seed"]))
        if split == "test":
            rows = [r for r in all_rows if r["split"] == "val"]
        elif split == "val":
            rows = [r for r in all_rows if r["split"] == "train" and r["scene_name"] in val_scenes]
        elif split == "train":
            rows = [r for r in all_rows if r["split"] == "train" and r["scene_name"] not in val_scenes]
        else:
            raise ValueError(f"unknown split {split!r}")
        sparse, target = [], []
        for r in rows:
            d = np.load(os.path.join(out_dir, r["token"] + ".npz"))
            # Both arrays are binary, so pack 8 cells per byte along the column axis (200 -> 25 bytes):
            # 8x less RAM, which matters for ~13k trainval frames (2 x 3 GB unpacked vs 2 x 0.4 GB).
            sparse.append(np.packbits(d["sparse"], axis=-1))
            target.append(np.packbits(d["target"], axis=-1))
        self.sparse = np.stack(sparse)   # uint8 [N,6,200,25] (packed bits)
        self.target = np.stack(target)   # uint8 [N,6,200,25] (packed bits)
        # Fraction of positive cells per target layer: used for the BCE pos_weight (train split only).
        n_pos = np.zeros(self.target.shape[1])
        for t in self.target:
            n_pos += np.unpackbits(t, axis=-1).sum(axis=(1, 2))
        self.layer_freq = n_pos / (len(self.target) * dcfg_cells(cfg))
        self.augment = augment and cfg["train"].get("augment_flip", False)
        # Camera tokens (hybrid DiT only): lazily read from disk, they are ~0.5 MB per frame.
        self.feat_dir = os.path.expanduser(dcfg["img_feat_dir"]) if with_cams else None
        self.tokens = [r["token"] for r in rows]

    def __len__(self):
        return len(self.sparse)

    def __getitem__(self, i):
        sparse = np.unpackbits(self.sparse[i], axis=-1)   # [6,200,200] {0,1}
        target = np.unpackbits(self.target[i], axis=-1)
        # Only augmentation: left-right mirror (flip the column axis) of input AND target.
        # A mirrored world is still a plausible world, and the ego-centred grid stays aligned.
        # We do NOT rotate or flip front/back: the 6 cameras cover front and back differently,
        # so those would create input patterns that never occur at test time.
        if self.augment and np.random.rand() < 0.5:
            sparse, target = sparse[:, :, ::-1], target[:, :, ::-1]
        out = (torch.from_numpy(np.ascontiguousarray(sparse)).float(),
               torch.from_numpy(np.ascontiguousarray(target)).float())
        if self.feat_dir is not None:
            d = np.load(os.path.join(self.feat_dir, self.tokens[i] + "_img.npz"))
            out += (torch.from_numpy(d["feats"]), torch.from_numpy(d["rays"]))   # fp16 [672,384], fp32 [672,6]
        return out
