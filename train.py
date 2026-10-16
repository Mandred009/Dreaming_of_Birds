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
from metrics import accumulate, format_metrics, new_stats, summarize
from models.unet import build_unet


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def layer_pos_weights(freq):
    """BCE pos_weight per layer = sqrt(neg / pos). Full neg/pos would give pedestrians (~0.1% of cells)
    a ~1000x weight and make training unstable; the sqrt softens it (same idea as the v1 class weights)."""
    freq = np.clip(freq, 1e-6, 1 - 1e-6)
    return torch.tensor(np.sqrt((1 - freq) / freq), dtype=torch.float32)


def soft_dice_loss(logits, target, eps=1.0):
    """1 - Dice, per layer, on sigmoid probabilities (differentiable), averaged over the 6 layers.
    Sums run over the whole batch so a frame where a layer is empty does not give a 0/0 term."""
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(0, 2, 3))
    total = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    return 1 - ((2 * inter + eps) / (total + eps)).mean()


def make_loss(pos_weight, dice_weight):
    """Loss = pos_weighted BCE-with-logits + dice_weight * soft Dice.
    The 6 layers are independent binary problems (a cell can be in several), so sigmoid/BCE per layer
    replaces the softmax/CE of v1. BCE gives dense gradients on every cell; Dice optimises overlap
    (what IoU measures) and ignores the huge background count."""
    def loss_fn(logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight[None, :, None, None])
        return bce + dice_weight * soft_dice_loss(logits, target)
    return loss_fn


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    stats, total_loss, n = new_stats(), 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * len(x)
        n += len(x)
        accumulate(stats, logits > 0, y)   # logit > 0 <=> sigmoid > 0.5
    return total_loss / n, summarize(stats)


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
    print(f"train {len(train_ds)} samples, val {len(val_ds)} samples (config dataset: {cfg.get('dataset', 'mini')}; test is not used here)")

    pos_weight = layer_pos_weights(train_ds.layer_freq)
    print("layer freq", train_ds.layer_freq.round(5), "-> BCE pos_weight", pos_weight.numpy().round(2))
    loss_fn = make_loss(pos_weight.to(device), t["dice_weight"])

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
        print(f"ep {epoch:3d} | train {train_loss:.4f} val {val_loss:.4f} | {format_metrics(m)} | lr {lr:.2e} | {mem:.0f} MiB",
              flush=True)
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("val_iou", {k: v for k, v in m.items()}, epoch)
        writer.add_scalar("lr", lr, epoch)
        writer.add_scalar("gpu_mem_MiB", mem, epoch)

        torch.save(model.state_dict(), os.path.join(t["ckpt_dir"], "last.pt"))
        if m["mIoU"] > best:
            best, bad_epochs = m["mIoU"], 0
            torch.save(model.state_dict(), os.path.join(t["ckpt_dir"], "best.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= t["patience"]:
                print(f"early stop: no val mIoU improvement for {t['patience']} epochs")
                break
    print(f"best val mIoU {best:.4f}; total time {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
