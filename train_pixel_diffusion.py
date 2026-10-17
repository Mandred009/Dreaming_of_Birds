"""Train the pixel-space conditional diffusion model D on the 6-layer BEV data.
Usage: python train_pixel_diffusion.py --config config.yaml [--epochs N]
"""
import argparse
import os
import time
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import BEVDataset
from metrics import evaluate_model, format_metrics
from models.pixel_diffusion import cosine_alpha_bar, denoise_loss, predict_probs
from models.unet import build_denoiser
from train import layer_pos_weights, make_loss, seed_everything   # same loss as baseline B


def val_miou(model, ds, alpha_bar, p, device, num_samples=1, seed=123):
    """Full DDIM sampling over a dataset -> metrics. Fixed generator so successive validations
    use the same noise and differ only because the model changed."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    fn = lambda sparse, cams: predict_probs(model, sparse, alpha_bar, p["ddim_steps"], num_samples, gen, cams)
    return evaluate_model(fn, DataLoader(ds, 16), device)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override config (smoke test)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    p, t = cfg["pixel_diffusion"], cfg["train"]
    epochs = args.epochs or p["epochs"]
    seed_everything(p["seed"])
    device = "cuda"

    train_ds = BEVDataset(cfg, "train", augment=True)   # flip on if train.augment_flip
    val_ds = BEVDataset(cfg, "val", augment=False)
    alpha_bar = cosine_alpha_bar(p["T"]).to(device)
    model = build_denoiser(cfg).to(device)
    train_dl = DataLoader(train_ds, p["batch_size"], shuffle=True, num_workers=p["num_workers"], drop_last=True)
    print(f"train {len(train_ds)} samples, val {len(val_ds)} samples")

    # Class-imbalance handling: exactly as in train.py (baseline B).
    pos_weight = layer_pos_weights(train_ds.layer_freq)
    print("layer freq", train_ds.layer_freq.round(5), "-> BCE pos_weight", pos_weight.numpy().round(2))
    loss_fn = make_loss(pos_weight.to(device), t["dice_weight"])

    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=p["warmup_epochs"])
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - p["warmup_epochs"]))
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
            m = val_miou(model, val_ds, alpha_bar, p, device)
            mem = torch.cuda.max_memory_allocated() / 2**20
            print(f"ep {epoch:3d} | train loss {train_loss:.4f} | val (DDIM {p['ddim_steps']}) {format_metrics(m)} | "
                  f"{mem:.0f} MiB | {(time.time() - start) / 60:.1f} min", flush=True)
            writer.add_scalars("val_iou", m, epoch)
            if m["mIoU"] > best:
                best, bad_vals = m["mIoU"], 0
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
