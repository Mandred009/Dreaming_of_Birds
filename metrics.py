"""Multi-label segmentation metrics for the 6 BEV layers.

Layers are independent binary masks (a cell can be drivable AND a ped crossing), so there is no
confusion matrix: we accumulate TP / FP / FN per layer over the whole split and take
IoU = TP / (TP + FP + FN) once at the end (not a mean of per-frame IoUs, which would be dominated
by frames where a layer is almost empty).

This matches BEVFusion-style map segmentation reporting (per-layer IoU, mean over the map layers).
They sweep the score threshold and report the best IoU; we fix 0.5 for simplicity.
"""
import numpy as np
import torch

LAYER_NAMES = ["drivable", "ped_crossing", "walkway", "divider", "vehicle", "pedestrian"]
MAP_LAYERS, OBJ_LAYERS = [0, 1, 2, 3], [4, 5]


def new_stats(n_layers=6):
    """[3, n_layers] float64: rows = TP, FP, FN."""
    return np.zeros((3, n_layers), dtype=np.float64)


def accumulate(stats, pred, target):
    """pred, target: bool/{0,1} tensors [B,L,H,W] (pred = probability > 0.5). Adds to `stats` in place."""
    pred, target = pred.bool(), target.bool()
    stats[0] += (pred & target).sum(dim=(0, 2, 3)).cpu().numpy()
    stats[1] += (pred & ~target).sum(dim=(0, 2, 3)).cpu().numpy()
    stats[2] += (~pred & target).sum(dim=(0, 2, 3)).cpu().numpy()


def summarize(stats):
    """Per-layer IoU plus map_mIoU (layers 0-3), obj_mIoU (4-5), mIoU (all 6). NaN layers (no
    positives and no predictions in the whole split) are skipped by nanmean."""
    tp, fp, fn = stats
    union = tp + fp + fn
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    out = {name: float(iou[i]) for i, name in enumerate(LAYER_NAMES)}
    out["map_mIoU"] = float(np.nanmean(iou[MAP_LAYERS]))
    out["obj_mIoU"] = float(np.nanmean(iou[OBJ_LAYERS]))
    out["mIoU"] = float(np.nanmean(iou))
    return out


@torch.no_grad()
def evaluate_model(predict_fn, loader, device="cuda", keep_ids=()):
    """Run predict_fn(sparse [B,6,H,W], cams or None) -> probabilities [B,6,H,W] over a DataLoader
    and return (metrics dict, {dataset index: bool prediction [6,H,W] numpy} for keep_ids).
    Batches are (sparse, target[, feats, rays]); memory stays small because nothing but the
    running TP/FP/FN (and the few kept predictions) is stored."""
    stats, kept, seen = new_stats(), {}, 0
    for batch in loader:
        sparse, target = batch[0].to(device), batch[1].to(device)
        cams = tuple(b.to(device) for b in batch[2:]) or None
        pred = predict_fn(sparse, cams) > 0.5
        accumulate(stats, pred, target)
        for j in range(len(sparse)):
            if seen + j in keep_ids:
                kept[seen + j] = pred[j].cpu().numpy()
        seen += len(sparse)
    return summarize(stats), kept


def format_metrics(m):
    return " ".join(f"{k} {m[k]:.3f}" for k in LAYER_NAMES) + f" | map {m['map_mIoU']:.3f} obj {m['obj_mIoU']:.3f} mIoU {m['mIoU']:.3f}"


if __name__ == "__main__":
    t = (torch.rand(2, 6, 20, 20) > 0.7)
    s = new_stats()
    accumulate(s, t, t)
    print(summarize(s))   # IoU of target vs itself must be 1
