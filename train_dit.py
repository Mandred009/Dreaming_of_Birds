"""Train the hybrid DiT (model F).  Usage: python train_dit.py --config config.yaml [--epochs N]

Same diffusion recipe as model D (cosine schedule, x0-logit prediction, pos_weighted BCE + Dice, DDIM);
only the denoiser differs, so the comparison D vs F isolates the network + camera tokens.
"""
import argparse
import copy
import math
import os
import time
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import BEVDataset
from metrics import evaluate_model, format_metrics
from models.dit import build_dit
from models.pixel_diffusion import cosine_alpha_bar, denoise_loss, predict_probs
from train import layer_pos_weights, make_loss, seed_everything


@torch.no_grad()
def ema_update(ema, model, step, decay):
    """ema <- decay * ema + (1 - decay) * model. Diffusion training is noisy (random t and noise every
    step), so the raw weights jump around; the exponential moving average is a much smoother, better
    sampler and is what gets evaluated. Early on we use a smaller decay (step-dependent) so the
    EMA is not dominated by the random init."""
    d = min(decay, (1 + step) / (10 + step))
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.lerp_(p, 1 - d)


def sample_val(model, ds, alpha_bar, cfg, device, steps=20, seed=123):
    """Full DDIM sampling (1 sample) over a dataset with bf16 autocast -> metrics."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    def fn(sparse, cams):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return predict_probs(model, sparse, alpha_bar, steps, 1, gen, cams)
    return evaluate_model(fn, DataLoader(ds, 16, num_workers=cfg["dit"]["num_workers"]), device)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override config (smoke test)")
    ap.add_argument("--max_steps", type=int, default=None, help="stop after this many optimizer steps (smoke test)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    d, t, p = cfg["dit"], cfg["train"], cfg["pixel_diffusion"]
    epochs = args.epochs or d["epochs"]
    seed_everything(d["seed"])
    device = "cuda"

    # No flip augmentation here: the precomputed camera features cannot be mirrored consistently with
    # the BEV (a flipped BEV would need flipped images, i.e. re-running DINOv2 and swapping left/right
    # cameras). augment=False so the dataset's flip is off whatever train.augment_flip says.
    train_ds = BEVDataset(cfg, "train", augment=False, with_cams=d["use_cameras"])
    val_ds = BEVDataset(cfg, "val", augment=False, with_cams=d["use_cameras"])
    train_dl = DataLoader(train_ds, d["batch_size"], shuffle=True, num_workers=d["num_workers"], drop_last=True)
    print(f"train {len(train_ds)} samples, val {len(val_ds)} samples; use_sparse={d['use_sparse']} use_cameras={d['use_cameras']}")

    pos_weight = layer_pos_weights(train_ds.layer_freq)
    print("layer freq", train_ds.layer_freq.round(5), "-> BCE pos_weight", pos_weight.numpy().round(2))
    loss_fn = make_loss(pos_weight.to(device), t["dice_weight"])
    alpha_bar = cosine_alpha_bar(p["T"]).to(device)

    model = build_dit(cfg).to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=d["lr"], weight_decay=d["weight_decay"])
    total_steps = epochs * (len(train_dl) // d["grad_accum"])
    # Linear warmup for warmup_steps optimizer steps, then cosine decay to 0 (stepped per optimizer step).
    lr_at = lambda s: (s + 1) / d["warmup_steps"] if s < d["warmup_steps"] else \
        0.5 * (1 + math.cos(math.pi * (s - d["warmup_steps"]) / max(1, total_steps - d["warmup_steps"])))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    os.makedirs(d["ckpt_dir"], exist_ok=True)
    writer = SummaryWriter(d["log_dir"])
    best, step, start = -1.0, 0, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, n_batches = 0.0, 0
        for i, batch in enumerate(train_dl):
            sparse, target = batch[0].to(device), batch[1].to(device)
            cams = tuple(b.to(device) for b in batch[2:]) or None
            with torch.autocast("cuda", dtype=torch.bfloat16):   # bf16: same range as fp32, no loss scaling needed
                loss = denoise_loss(model, sparse, target, alpha_bar, loss_fn, cams)
            (loss / d["grad_accum"]).backward()
            train_loss += loss.item()
            n_batches += 1
            if (i + 1) % d["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                ema_update(ema, model, step, d["ema_decay"])
                if step % 50 == 0:
                    writer.add_scalar("step_loss", loss.item(), step)
        train_loss /= n_batches
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)
        print(f"ep {epoch:3d} | train loss {train_loss:.4f} | step {step} | {(time.time() - start) / 60:.1f} min", flush=True)
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict()), os.path.join(d["ckpt_dir"], "last.pt"))

        done = args.max_steps is not None and step >= args.max_steps
        if epoch % d["val_every"] == 0 or epoch == epochs or done:
            m = sample_val(ema, val_ds, alpha_bar, cfg, device, p["ddim_steps"])   # EMA weights
            mem = torch.cuda.max_memory_allocated() / 2**20
            print(f"   val (EMA, DDIM {p['ddim_steps']}) {format_metrics(m)} | peak {mem:.0f} MiB", flush=True)
            writer.add_scalars("val_iou", m, epoch)
            writer.add_scalar("gpu_mem_MiB", mem, epoch)
            if m["mIoU"] > best:
                best = m["mIoU"]
                torch.save(ema.state_dict(), os.path.join(d["ckpt_dir"], "best.pt"))   # EMA weights only
        if done:
            break
    print(f"best val mIoU {best:.4f}; total time {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
