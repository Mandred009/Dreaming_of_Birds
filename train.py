"""Train the U-Net baseline.  Usage: python train.py --config config.yaml"""
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import BEVDataset
from metrics import confusion_matrix, summarize
from models.unet import build_unet


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(target, num_classes):
    """Inverse-sqrt-frequency weights, normalised to mean 1. Plain inverse frequency would give
    pedestrians (0.12% of pixels) a ~800x weight and make training unstable; sqrt softens it."""
    freq = np.bincount(target.reshape(-1), minlength=num_classes) / target.size
    w = 1.0 / np.sqrt(freq)
    return torch.tensor(w / w.mean(), dtype=torch.float32), freq


def soft_dice_loss(logits, target, eps=1.0):
    """1 - Dice on the foreground classes (1, 2), computed on softmax probabilities so it is
    differentiable. Averaged over classes, summed over the whole batch for stability."""
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, probs.shape[1]).permute(0, 3, 1, 2).float()
    probs, onehot = probs[:, 1:], onehot[:, 1:]
    inter = (probs * onehot).sum(dim=(0, 2, 3))
    total = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    return 1 - ((2 * inter + eps) / (total + eps)).mean()


def make_loss(weights, dice_weight):
    """Loss = weighted CE + dice_weight * soft Dice.
    CE with class weights gives dense, well-behaved gradients on every pixel; Dice directly
    optimises overlap (what IoU measures) and is insensitive to the huge background count.
    Together they are the standard, simple recipe for rare foreground classes."""
    def loss_fn(logits, target):
        ce = F.cross_entropy(logits, target, weight=weights)
        return ce + dice_weight * soft_dice_loss(logits, target)
    return loss_fn


@torch.no_grad()
def validate(model, loader, loss_fn, device, n_classes=3):
    model.eval()
    cm, total_loss, n = np.zeros((n_classes, n_classes), dtype=np.int64), 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * len(x)
        n += len(x)
        cm += confusion_matrix(logits.argmax(1).cpu().numpy(), y.cpu().numpy(), n_classes)
    return total_loss / n, summarize(cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    t = cfg["train"]
    seed_everything(t["seed"])
    device = "cuda"

    train_ds = BEVDataset(cfg, "train", augment=True)
    val_ds = BEVDataset(cfg, "val", augment=False)
    train_dl = DataLoader(train_ds, t["batch_size"], shuffle=True, num_workers=t["num_workers"], drop_last=True)
    val_dl = DataLoader(val_ds, t["batch_size"], shuffle=False, num_workers=t["num_workers"])
    print(f"train {len(train_ds)} samples, val {len(val_ds)} samples")

    weights, freq = class_weights(train_ds.target, cfg["unet"]["num_classes"])
    print("class freq", freq.round(5), "-> CE weights", weights.numpy().round(3))
    loss_fn = make_loss(weights.to(device), t["dice_weight"])

    model = build_unet(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    # Linear warmup (lr from 1% -> 100%) for a few epochs, then cosine decay to ~0.
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=t["warmup_epochs"])
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"] - t["warmup_epochs"])
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cosine], milestones=[t["warmup_epochs"]])

    os.makedirs(t["ckpt_dir"], exist_ok=True)
    writer = SummaryWriter(t["log_dir"])
    best, bad_epochs, start = -1.0, 0, time.time()

    for epoch in range(1, t["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_dl) * t["batch_size"]
        sched.step()

        val_loss, m = validate(model, val_dl, loss_fn, device)
        lr = opt.param_groups[0]["lr"]
        mem = torch.cuda.max_memory_allocated() / 2**20
        print(f"ep {epoch:3d} | train {train_loss:.4f} val {val_loss:.4f} | IoU veh {m['vehicle']:.3f} "
              f"ped {m['pedestrian']:.3f} mIoU {m['miou']:.3f} bg {m['background']:.3f} | lr {lr:.2e} | {mem:.0f} MiB",
              flush=True)
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("val_iou", {"vehicle": m["vehicle"], "pedestrian": m["pedestrian"], "miou": m["miou"]}, epoch)
        writer.add_scalar("lr", lr, epoch)
        writer.add_scalar("gpu_mem_MiB", mem, epoch)

        torch.save(model.state_dict(), os.path.join(t["ckpt_dir"], "last.pt"))
        if m["miou"] > best:
            best, bad_epochs = m["miou"], 0
            torch.save(model.state_dict(), os.path.join(t["ckpt_dir"], "best.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= t["patience"]:
                print(f"early stop: no val mIoU improvement for {t['patience']} epochs")
                break
    print(f"best val mIoU {best:.4f}; total time {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
