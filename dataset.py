"""BEV dataset: loads the cached .npz files fully into RAM (the whole cache is tiny)."""
import csv
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BEVDataset(Dataset):
    def __init__(self, cfg, split, augment=False):
        """split: 'train' or 'val' (index.csv says mini_train / mini_val in the split column)."""
        out_dir = os.path.expanduser(cfg["out_dir"])
        with open(os.path.join(out_dir, "index.csv")) as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == split]
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
