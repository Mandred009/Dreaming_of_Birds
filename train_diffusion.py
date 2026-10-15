"""Train the ControlNet of model C.  Usage: python train_diffusion.py --config config.yaml
(--smoke runs 20 steps on 2 val frames to check shapes and memory)."""
import argparse
import os
import random
import time
import numpy as np
import torch
import yaml
from diffusers import ControlNetModel
from torch.utils.tensorboard import SummaryWriter

from dataset import BEVDataset
from metrics import confusion_matrix, summarize
from models.diffusion import BEVDiffusion, paint, encode_images, encode_prompt, load_vae


def translate(x, dy, dx, fill):
    """Shift x [C,H,W] down by dy and right by dx pixels; the uncovered border is filled with
    `fill` ([C,1,1] or same shape as x) instead of wrapping around."""
    out = fill.expand_as(x).clone()
    H, W = x.shape[-2:]
    out[:, max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = \
        x[:, max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
    return out


def save_controlnet(model, path):
    """Only the ControlNet is trained, so only it is saved (the rest is downloaded SD 1.5)."""
    os.makedirs(path, exist_ok=True)
    model.controlnet.save_pretrained(path)


@torch.no_grad()
def validate(model, val_ds, bs=8):
    """Cheap eval: 1 DDIM sample per val frame -> class map -> mIoU against the target."""
    cm = np.zeros((3, 3), dtype=np.int64)
    for i in range(0, len(val_ds), bs):
        sparse = torch.from_numpy(val_ds.sparse[i:i + bs])
        pred = model.predict(sparse, num_samples=1, seed=i).cpu().numpy()
        cm += confusion_matrix(pred, val_ds.target[i:i + bs], 3)
    return summarize(cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume_steps", type=int, default=0,
                    help="continue from checkpoints/diffusion/last, which was trained for this many steps")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    d = cfg["diffusion"]
    random.seed(d["seed"]); np.random.seed(d["seed"]); torch.manual_seed(d["seed"])
    max_steps, eval_every = (20, 10) if args.smoke else (d["max_steps"], d["eval_every"])

    train_ds = BEVDataset(cfg, "train")
    val_ds = BEVDataset(cfg, "val")
    if args.smoke:
        val_ds.sparse, val_ds.target = val_ds.sparse[:2], val_ds.target[:2]
    print(f"train {len(train_ds)} frames, val {len(val_ds)} frames", flush=True)

    # Cache the target latents once (the VAE is frozen, so they never change).
    # Flip augmentation: a mirrored latent is only approximately the latent of the mirrored image
    # (the VAE convolutions are not exactly mirror-symmetric), so instead we encode both the
    # original AND the left-right flipped targets once, and pick one of the two at random.
    vae = load_vae(cfg)
    tgt = torch.from_numpy(train_ds.target)
    lat, lat_flip = [], []
    for i in range(0, len(tgt), 16):
        t = tgt[i:i + 16].cuda()
        lat.append(encode_images(vae, paint(t, d["image_size"])).cpu())
        lat_flip.append(encode_images(vae, paint(t.flip(-1), d["image_size"])).cpu())
    lat, lat_flip = torch.cat(lat), torch.cat(lat_flip)
    sparse_all = torch.from_numpy(train_ds.sparse)          # uint8, kept on CPU
    del vae
    torch.cuda.empty_cache()
    prompt_emb = encode_prompt(cfg)

    model = BEVDiffusion(cfg, prompt_emb)
    if args.resume_steps:
        # Continue an earlier run: load its ControlNet weights (the AdamW state is not saved, it restarts).
        model.controlnet = ControlNetModel.from_pretrained(os.path.join(d["ckpt_dir"], "last")).to("cuda")
        model.controlnet.enable_gradient_checkpointing()
    model.controlnet.train()
    model.unet.train()   # train() only needed for gradient checkpointing; UNet has no dropout/BN
    params = [p for p in model.controlnet.parameters()]
    print(f"ControlNet params {sum(p.numel() for p in params) / 1e6:.0f} M (fp32); training only these")
    opt = torch.optim.AdamW(params, lr=d["lr"], weight_decay=0.01)

    writer = SummaryWriter(d["log_dir"])
    ckpt = d["ckpt_dir"]
    best, start, run_loss, n = -1.0, time.time(), 0.0, 0
    for step in range(1, max_steps + 1):
        for _ in range(d["grad_accum"]):
            idx = torch.randint(0, len(lat), (d["batch_size"],))
            flip = torch.rand(len(idx)) < 0.5
            latents = torch.where(flip[:, None, None, None], lat_flip[idx], lat[idx])
            sparse = sparse_all[idx].cuda()
            # Mirror the sparse input with the same coin flip as the target latent.
            sparse = torch.where(flip.cuda()[:, None, None, None], sparse.flip(-1), sparse)
            # Extra augmentation (282 frames are few for a 361M-parameter ControlNet, which
            # memorised them: train IoU 0.5, unseen frames ~0.0). (1) Shift target and input
            # together by -4/0/+4 latent cells = 32 px at 256 = exactly 25 BEV cells, so the two
            # stay aligned exactly. Target border = blank-map latent, input border = no points.
            # (2) Randomly drop sparse points so it cannot memorise exact point patterns.
            latents, sparse = latents.cuda(), sparse.clone()
            for b in range(len(idx)):
                if np.random.rand() < d["shift_prob"]:
                    dy, dx = np.random.choice([-4, 0, 4], 2)
                    latents[b] = translate(latents[b], dy, dx, model.blank_latent[0])
                    sparse[b] = translate(sparse[b], dy // 4 * 25, dx // 4 * 25, torch.zeros(4, 1, 1, dtype=sparse.dtype, device="cuda"))
            keep = d["point_keep_min"] + (1 - d["point_keep_min"]) * torch.rand(len(idx), 1, 1, 1, device="cuda")
            sparse = sparse * (torch.rand_like(sparse, dtype=torch.float32) < keep)
            loss = model.training_loss(latents, sparse) / d["grad_accum"]
            loss.backward()
            run_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)   # guards against a rare bad step
        opt.step()
        opt.zero_grad(set_to_none=True)
        n += 1
        gstep = step + args.resume_steps   # step number counted over all runs

        if step % 50 == 0:
            writer.add_scalar("train_loss", run_loss / n, gstep)
            print(f"step {gstep} loss {run_loss / n:.4f} | {(time.time() - start) / 60:.1f} min | "
                  f"VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            run_loss, n = 0.0, 0
        if step % eval_every == 0:
            m = validate(model, val_ds)
            model.controlnet.train(); model.unet.train()
            print(f"== step {gstep} val IoU veh {m['vehicle']:.3f} ped {m['pedestrian']:.3f} mIoU {m['miou']:.3f}", flush=True)
            writer.add_scalars("val_iou", {"vehicle": m["vehicle"], "pedestrian": m["pedestrian"], "miou": m["miou"]}, gstep)
            save_controlnet(model, os.path.join(ckpt, "last"))
            if m["miou"] > best:
                best = m["miou"]
                save_controlnet(model, os.path.join(ckpt, "best"))
                print(f"   new best {best:.4f}", flush=True)
    print(f"best val mIoU {best:.4f}; total {(time.time() - start) / 60:.1f} min; "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
