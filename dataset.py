"""BEV dataset: loads the cached .npz files fully into RAM (the whole cache is tiny)."""
import csv
import os
import numpy as np
import torch
from torch.utils.data import Dataset


def dataset_cfg(cfg):
    """dataroot/version/out_dir/data for the dataset chosen by cfg['dataset'] ('mini' or 'trainval')."""
    block = cfg if cfg.get("dataset", "mini") == "mini" else cfg[cfg["dataset"]]
    return {k: block[k] for k in ("dataroot", "version", "out_dir", "data")}


def trainval_val_scenes(rows, frac, seed):
    """Seeded ~frac of the available official-train scenes (by scene), sorted."""
    scenes = sorted({r["scene_name"] for r in rows if r["split"] == "train"})
    n = max(1, round(frac * len(scenes)))
    return sorted(np.random.RandomState(seed).choice(scenes, n, replace=False).tolist())


class BEVDataset(Dataset):
    def __init__(self, cfg, split, augment=False):
        """split: 'train', 'val' or 'test'.

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
            sparse.append(d["sparse"])
            target.append(d["target"])
        self.sparse = np.stack(sparse)   # uint8 [N,4,200,200]
        self.target = np.stack(target)   # uint8 [N,200,200]
        self.augment = augment

    def __len__(self):
        return len(self.sparse)

    def __getitem__(self, i):
        sparse, target = self.sparse[i], self.target[i]
        # Only augmentation: left-right mirror (flip the column axis) of input AND target.
        # A mirrored world is still a plausible world, and the ego-centred grid stays aligned.
        # We do NOT rotate or flip front/back: the 6 cameras cover front and back differently,
        # so those would create input patterns that never occur at test time.
        if self.augment and np.random.rand() < 0.5:
            sparse, target = sparse[:, :, ::-1], target[:, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(sparse)).float(),
                torch.from_numpy(np.ascontiguousarray(target)).long())
