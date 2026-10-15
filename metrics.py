"""Segmentation metrics from a confusion matrix."""
import numpy as np

CLASS_NAMES = ["background", "vehicle", "pedestrian"]


def confusion_matrix(pred, target, n):
    """cm[t, p] = number of pixels with true class t predicted as p. Arrays of ints in [0, n).
    Encode each (t, p) pair as one number t*n+p, count with bincount, reshape to n x n.
    Add the result of each batch together to accumulate over the whole val set."""
    idx = target.reshape(-1).astype(np.int64) * n + pred.reshape(-1).astype(np.int64)
    return np.bincount(idx, minlength=n * n).reshape(n, n)


def iou_per_class(cm):
    """IoU = TP / (TP + FP + FN) = diag / (row sum + column sum - diag). NaN if class absent."""
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - tp
    return np.where(union > 0, tp / np.maximum(union, 1), np.nan)


def summarize(cm):
    """Headline numbers. mIoU = mean over the two FOREGROUND classes only (vehicle, pedestrian);
    background IoU is ~0.98 for any model and would hide the differences. BEV papers likewise
    report vehicle IoU (and pedestrian IoU) rather than a mean that includes background."""
    iou = iou_per_class(cm)
    return dict(background=iou[0], vehicle=iou[1], pedestrian=iou[2], miou=float(np.nanmean(iou[1:])))


if __name__ == "__main__":
    t = np.random.randint(0, 3, (4, 50, 50))
    print(summarize(confusion_matrix(t, t, 3)))   # IoU of target vs itself must be 1
