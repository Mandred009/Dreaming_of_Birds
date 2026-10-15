"""Train the pixel-space conditional diffusion model (E4).
Usage: python train_pixel_diffusion.py --config config.yaml [--epochs N] [--probe]
--probe: no training; load best.pt and print one-step x0 IoU at t=50 and t=999 on train/val/test."""
import argparse
import os
import time
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import BEVDataset
from metrics import confusion_matrix, summarize
from models.pixel_diffusion import cosine_alpha_bar, denoise_loss, predict, predict_x0_once
from models.unet import build_denoiser
from train import class_weights, make_loss, seed_everything   # same loss as baseline B


@torch.no_grad()
def sample_miou(model, ds, alpha_bar, p, device, num_samples=1):
    """Full DDIM sampling over a dataset -> metrics. Fixed generator so successive validations
    use the same noise and differ only because the model changed."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(123)
    cm = np.zeros((3, 3), dtype=np.int64)
    for i in range(0, len(ds), 16):
        x = torch.from_numpy(ds.sparse[i:i + 16]).float().to(device)
        pred = predict(model, x, alpha_bar, p["ddim_steps"], num_samples, gen)
        cm += confusion_matrix(pred.cpu().numpy(), ds.target[i:i + 16], 3)
    return summarize(cm)


@torch.no_grad()
def probe(model, ds, alpha_bar, t_value, device):
    """IoU of the one-step x0 prediction from the true map noised to level t_value."""
    model.eval()
    torch.manual_seed(0)
    cm = np.zeros((3, 3), dtype=np.int64)
    for i in range(0, len(ds), 16):
        x = torch.from_numpy(ds.sparse[i:i + 16]).float().to(device)
        y = torch.from_numpy(ds.target[i:i + 16]).long().to(device)
        cm += confusion_matrix(predict_x0_once(model, x, y, t_value, alpha_bar).cpu().numpy(), ds.target[i:i + 16], 3)
    return summarize(cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override config (smoke test)")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    p, t = cfg["pixel_diffusion"], cfg["train"]
    epochs = args.epochs or p["epochs"]
    seed_everything(p["seed"])
    device = "cuda"

    train_ds = BEVDataset(cfg, "train", augment=True)
    val_ds = BEVDataset(cfg, "val", augment=False)
    alpha_bar = cosine_alpha_bar(p["T"]).to(device)
    model = build_denoiser(cfg).to(device)

    if args.probe:
        model.load_state_dict(torch.load(os.path.join(p["ckpt_dir"], "best.pt"), map_location=device))
        for name in ["train", "val", "test"]:
            ds = BEVDataset(cfg, name, augment=False)
            for tv in (50, 999):
                m = probe(model, ds, alpha_bar, tv, device)
                print(f"{name:5s} t={tv:3d}: veh {m['vehicle']:.3f} ped {m['pedestrian']:.3f} mIoU {m['miou']:.3f}")
        return

    train_dl = DataLoader(train_ds, p["batch_size"], shuffle=True, num_workers=p["num_workers"], drop_last=True)
    print(f"train {len(train_ds)} samples, val {len(val_ds)} samples")

    # Class weights and loss: exactly as in train.py (baseline B).
    weights, _ = class_weights(train_ds.target, cfg["unet"]["num_classes"])
    loss_fn = make_loss(weights.to(device), t["dice_weight"])

    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=p["warmup_epochs"])
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - p["warmup_epochs"])
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cosine], milestones=[p["warmup_epochs"]])

    os.makedirs(p["ckpt_dir"], exist_ok=True)
    writer = SummaryWriter(p["log_dir"])
    best, bad_vals, start = -1.0, 0, time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = denoise_loss(model, x, y, alpha_bar, loss_fn)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_dl) * p["batch_size"]
        sched.step()
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)
        torch.save(model.state_dict(), os.path.join(p["ckpt_dir"], "last.pt"))

        if epoch % p["val_every"] == 0 or epoch == epochs:
            m = sample_miou(model, val_ds, alpha_bar, p, device)
            mem = torch.cuda.max_memory_allocated() / 2**20
            print(f"ep {epoch:3d} | train loss {train_loss:.4f} | val (DDIM {p['ddim_steps']}) IoU veh {m['vehicle']:.3f} "
                  f"ped {m['pedestrian']:.3f} mIoU {m['miou']:.3f} | {mem:.0f} MiB | {(time.time() - start) / 60:.1f} min", flush=True)
            writer.add_scalars("val_iou", {"vehicle": m["vehicle"], "pedestrian": m["pedestrian"], "miou": m["miou"]}, epoch)
            if m["miou"] > best:
                best, bad_vals = m["miou"], 0
                torch.save(model.state_dict(), os.path.join(p["ckpt_dir"], "best.pt"))
            else:
                bad_vals += 1
                if bad_vals >= p["patience"]:
                    print(f"early stop: no val mIoU improvement for {p['patience']} validations")
                    break
        elif epoch % 5 == 0:
            print(f"ep {epoch:3d} | train loss {train_loss:.4f}", flush=True)
    print(f"best val mIoU {best:.4f}; total time {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
