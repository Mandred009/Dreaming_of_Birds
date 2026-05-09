"""Train SDXL 1.5 + multi-ControlNet to generate BEV.

Logic (high level):
- Load BEV (target) + condition images from dataloader
- Encode BEV -> latent with SDXL VAE
- Add noise at random timestep
- Run N ControlNets (RGB/Depth/Seg), optionally over all 6 views
- Run SDXL UNet with summed residuals
- Train with epsilon-prediction MSE
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

# Set HuggingFace download timeout to prevent hangs
# This applies to downloading model weights from HuggingFace Hub
os.environ.setdefault("HF_HUB_TIMEOUT", "600")  # 10 minutes

from data_loader import CAMERAS
from data_loader import make_dataloader
from losses import LossConfig, Losses
from my_model import SD15MultiControlNetBEV

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("Missing dependency `pyyaml`. Install with `pip install pyyaml`.") from e

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _save_checkpoint(path: str, *, model: nn.Module, optim: torch.optim.Optimizer, step: int) -> None:
    def trainable_state_dict(m: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Save only parameters that are currently trainable (requires_grad=True).
        This keeps checkpoints small and avoids storing frozen SD backbone weights.
        """
        full = m.state_dict()
        keep: Dict[str, torch.Tensor] = {}
        for name, p in m.named_parameters():
            if p.requires_grad:
                if name in full:
                    keep[name] = full[name]
        return keep

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": trainable_state_dict(model),
            "optim": optim.state_dict(),
        },
        path,
    )


def _latest_checkpoint(run_name: str) -> Optional[str]:
    ckpt_dir = os.path.join("checkpoints", run_name)
    if not os.path.isdir(ckpt_dir):
        return None
    best_step = None
    best_path = None
    for fn in os.listdir(ckpt_dir):
        if not (fn.startswith("step_") and fn.endswith(".pt")):
            continue
        stem = fn[len("step_") : -len(".pt")]
        if not stem.isdigit():
            continue
        s = int(stem)
        if best_step is None or s > best_step:
            best_step = s
            best_path = os.path.join(ckpt_dir, fn)
    return best_path


def _load_checkpoint(path: str, *, model: nn.Module, optim: torch.optim.Optimizer, device: torch.device) -> int:
    ckpt = torch.load(path, map_location=device)
    # Checkpoints store only trainable params; load them non-strictly.
    model.load_state_dict(ckpt["model"], strict=False)
    optim.load_state_dict(ckpt["optim"])
    return int(ckpt.get("step", 0))


def _flatten_for_hparams(d: Dict[str, Any], *, prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested dicts into a single-level dict for TensorBoard hparams.
    Keeps only scalars / short strings / bools / ints / floats.
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_for_hparams(v, prefix=key))
        elif isinstance(v, (int, float, bool)):
            out[key] = v
        elif isinstance(v, str):
            # Avoid huge strings in hparams; write full config separately as text.
            if len(v) <= 200:
                out[key] = v
        elif v is None:
            out[key] = "null"
        elif isinstance(v, (list, tuple)):
            # Store short lists (e.g., betas) compactly.
            if len(v) <= 16 and all(isinstance(x, (int, float, bool, str)) or x is None for x in v):
                out[key] = str(list(v))
    return out


def _get_control_images(batch: Dict[str, Any], sources: Sequence[str]) -> Sequence[torch.Tensor]:
    """
    Map config sources -> tensors for ControlNet.
    Returns a list where each element is either:
    - (B,3,H,W) for *_front
    - (B,6,3,H,W) for *_all
    """
    out = []
    for src in sources:
        if src == "rgb_front":
            out.append(batch["rgb"][:, 0])
        elif src == "rgb_all":
            out.append(batch["rgb"])
        elif src == "depth_front":
            out.append(batch["depth"][:, 0].repeat(1, 3, 1, 1))
        elif src == "depth_all":
            out.append(batch["depth"].repeat(1, 1, 3, 1, 1))
        elif src == "seg_front":
            out.append(batch["seg"][:, 0])
        elif src == "seg_all":
            out.append(batch["seg"])
        else:
            raise ValueError(f"Unknown controlnet source: {src}")
    return out


@torch.no_grad()
def _log_labeled_recon_collage(
    writer: SummaryWriter,
    *,
    tag: str,
    step: int,
    batch: Dict[str, Any],
    pred_bev: torch.Tensor,
) -> None:
    """
    Labeled collage:
    - GT BEV, Pred BEV
    - 6 RGB + 6 Depth + 6 Seg (first batch element)
    """
    import math
    import numpy as np
    from PIL import Image, ImageDraw

    def tensor_to_pil(t: torch.Tensor) -> Image.Image:
        t = t.detach().float().cpu().clamp(0, 1)
        c, h, w = t.shape
        if c == 1:
            arr = (t[0] * 255.0).to(torch.uint8).numpy()
            return Image.fromarray(arr, mode="L").convert("RGB")
        arr = (t.permute(1, 2, 0) * 255.0).to(torch.uint8).numpy()
        return Image.fromarray(arr, mode="RGB")

    def make_collage(items, cols: int = 5, pad: int = 10, title_h: int = 18) -> Image.Image:
        w, h = items[0][1].size
        rows = math.ceil(len(items) / cols)
        cell_w = w + 2 * pad
        cell_h = h + 2 * pad + title_h
        canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for i, (title, im) in enumerate(items):
            r = i // cols
            c = i % cols
            x0 = c * cell_w + pad
            y0 = r * cell_h + pad
            draw.text((x0, y0), title, fill=(230, 230, 230))
            canvas.paste(im, (x0, y0 + title_h))
        return canvas

    gt_bev = batch["bev"][0]
    pb_bev = pred_bev[0]

    items = [("GT BEV", tensor_to_pil(gt_bev)), ("PRED BEV", tensor_to_pil(pb_bev))]

    rgb = batch["rgb"][0]  # (6,3,H,W)
    for i, cam in enumerate(CAMERAS):
        items.append((f"RGB {cam}", tensor_to_pil(rgb[i])))

    depth = batch["depth"][0]  # (6,1,H,W)
    for i, cam in enumerate(CAMERAS):
        items.append((f"DEPTH {cam}", tensor_to_pil(depth[i])))

    seg = batch["seg"][0]  # (6,3,H,W)
    for i, cam in enumerate(CAMERAS):
        items.append((f"SEG {cam}", tensor_to_pil(seg[i])))

    collage = make_collage(items, cols=5)
    collage_np = np.array(collage, dtype=np.uint8)  # (H,W,3)
    collage_t = torch.from_numpy(collage_np).permute(2, 0, 1).to(torch.float32) / 255.0
    writer.add_image(tag, collage_t, global_step=step)


def _vae_latent_spatial_dims(vae: nn.Module, h_img: int, w_img: int) -> tuple[int, int]:
    """H,W of VAE latent grid (spatial factor is typically 8 for SD1.5-style autoencoders)."""
    bc = getattr(vae.config, "block_out_channels", None)
    if bc is None:
        sf = 8
    else:
        sf = 2 ** (len(bc) - 1)
    return h_img // int(sf), w_img // int(sf)


@torch.no_grad()
def _validation_reconstruction(
    *,
    net: SD15MultiControlNetBEV,
    batch: Dict[str, Any],
    control_images: Sequence[torch.Tensor],
    control_scales: Sequence[float],
    device: torch.device,
    num_inference_steps: int,
    collage_seed_offset: int = 0,
) -> torch.Tensor:
    """
    Inference-only collage generation: diffuse from Gaussian latent noise down to denoised
    latents (same sampler as pretrained DDPM scheduler), conditioned on ControlNets +
    frozen UNet + text embeds — then VAE decode to BEV pixels.

    This matches "random noise → full sampling" rather than one-step GT-based x0.
    """
    bev = batch["bev"]
    b, _, h_img, w_img = bev.shape
    latent_h, latent_w = _vae_latent_spatial_dims(net.vae, h_img, w_img)
    c = net.unet.config.in_channels

    gen = torch.Generator(device=device)
    gen.manual_seed(int(collage_seed_offset))

    sigma = getattr(net.noise_scheduler, "init_noise_sigma", 1.0)
    latents = torch.randn((b, c, latent_h, latent_w), device=device, dtype=torch.float32, generator=gen)
    if isinstance(sigma, torch.Tensor):
        latents = latents * sigma.to(device=device, dtype=latents.dtype)
    else:
        latents = latents * float(sigma)

    net.noise_scheduler.set_timesteps(int(num_inference_steps), device=device)
    tup_ctrl = tuple(control_images)
    tup_scale = tuple(float(s) for s in control_scales)

    for t in net.noise_scheduler.timesteps:
        ti = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        ts = torch.full((b,), ti, device=device, dtype=torch.long)
        eps = net.predict_eps(
            noisy_latents=latents,
            timesteps=ts,
            control_images=tup_ctrl,
            control_scales=tup_scale,
            view_ids=batch.get("view_ids"),
        )
        latents = net.noise_scheduler.step(eps, ti, latents).prev_sample

    latents = (latents / net.vae.config.scaling_factor).to(dtype=torch.float32)
    with torch.autocast(device_type=device.type, enabled=False):  # VAE decode in fp32
        bev_pred = net.vae.decode(latents).sample
    bev_pred = (bev_pred + 1.0) / 2.0
    return bev_pred.clamp(0, 1)


def train_model(config_path: str = "config_experiment.yaml", *, resume: bool = False) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    exp = cfg["experiment"]
    log_cfg = cfg["logging"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    optim_cfg = cfg["optim"]
    loss_cfg = cfg["loss"]

    seed = int(exp.get("seed", 42))
    _set_seed(seed)

    device = torch.device(train_cfg.get("device", "cuda"))
    precision = train_cfg.get("precision", "bf16")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_name = exp["name"]
    tb_root = log_cfg["tensorboard"]["log_dir"]
    tb_dir = os.path.join(tb_root, run_name)
    writer = SummaryWriter(tb_dir) if log_cfg["tensorboard"].get("enabled", True) else None

    # DataLoader setup with debugging output
    num_workers = int(data_cfg.get("num_workers", 0))
    print(f"[TRAIN] Creating dataloader with num_workers={num_workers}")
    print(f"[TRAIN] BEV root: {data_cfg.get('bev_root')}")
    print(f"[TRAIN] OCC3D base dir: {data_cfg.get('occ3d_base_dir')}")
    
    try:
        print(f"[TRAIN] Initializing dataset (this may take a moment if loading annotations.json)...")
        dl = make_dataloader(
            root=data_cfg.get("bev_root"),
            batch_size=int(train_cfg["batch_size"]),
            shuffle=bool(data_cfg.get("shuffle", True)),
            num_workers=num_workers,
            drop_last=bool(data_cfg.get("drop_last", False)),
            resolution=int(data_cfg.get("resolution", 256)),
            load_rgb=True,
            occ3d_base_dir=data_cfg.get("occ3d_base_dir"),
            annotations_json=data_cfg.get("annotations_json"),
        )
        print(f"[TRAIN] ✓ Dataloader created successfully with {len(dl.dataset)} samples")
    except Exception as e:
        print(f"[TRAIN] ✗ Failed to create dataloader: {e}", flush=True)
        raise

    if model_cfg.get("type") != "sd15_multicontrolnet":
        raise ValueError("This simplified trainer expects model.type == 'sd15_multicontrolnet'")

    sd_cfg = model_cfg["sd15"]
    control_cfg = model_cfg["controlnet"]
    sources = control_cfg.get("sources", ["rgb_all", "depth_all", "seg_all"])
    scales = control_cfg.get("scales", [1.0] * len(sources))
    if len(scales) != len(sources):
        raise ValueError("controlnet.scales must match controlnet.sources length")

    print(f"[TRAIN] Creating SD1.5 model with {len(sources)} ControlNets...")
    print(f"[TRAIN] Pretrained model: {sd_cfg.get('pretrained')}")
    print(f"[TRAIN] ControlNet sources: {sources}")
    print(f"[TRAIN] Local files only: {sd_cfg.get('local_files_only')}")
    print(f"[TRAIN]", flush=True)
    
    try:
        # Back-compat: older configs used `use_view_embed`. Newer configs can set `view_encoding`.
        view_encoding = str(control_cfg.get("view_encoding", "")).strip()
        if not view_encoding:
            view_encoding = "learned" if bool(control_cfg.get("use_view_embed", True)) else "none"
        net = SD15MultiControlNetBEV(
            pretrained=sd_cfg.get("pretrained", "runwayml/stable-diffusion-v1-5"),
            prompt=sd_cfg.get("prompt", "a high quality image"),
            local_files_only=bool(sd_cfg.get("local_files_only", False)),
            freeze_unet=bool(control_cfg.get("freeze_unet", True)),
            num_views=int(control_cfg.get("num_views", 6)),
            use_view_embed=bool(control_cfg.get("use_view_embed", True)),
            view_encoding=view_encoding,
            learned_view_fusion=bool(control_cfg.get("learned_view_fusion", True)),
            use_view_cross_attn=bool(control_cfg.get("use_view_cross_attn", True)),
            view_attn_d_model=int(control_cfg.get("view_attn_d_model", 256)),
            view_attn_nhead=int(control_cfg.get("view_attn_nhead", 8)),
            view_attn_dropout=float(control_cfg.get("view_attn_dropout", 0.0)),
        ).to(device)
        print(f"[TRAIN] Model instance created, initializing (this may take 1-5 minutes)...", flush=True)
        net.initialize(device, n_controlnets=len(sources))
        print(f"[TRAIN] ✓ Model loaded and initialized successfully!", flush=True)
    except Exception as e:
        print(f"[TRAIN] ✗ Failed to initialize model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    # ControlNet-only training: optimize ControlNet + optional view modules.
    params = list(net.controlnets.parameters())
    if getattr(net, "view_fusion_logits", None) is not None:
        params.append(net.view_fusion_logits)  # type: ignore[arg-type]
    if getattr(net, "view_cross_attn", None) is not None:
        params += list(net.view_cross_attn.parameters())
    if getattr(net, "view_embed", None) is not None:
        params += list(net.view_embed.parameters())  # type: ignore[union-attr]
    opt = AdamW(
        params,
        lr=float(optim_cfg["lr"]),
        betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
        eps=float(optim_cfg.get("eps", 1e-8)),
        weight_decay=float(optim_cfg.get("weight_decay", 1e-2)),
    )

    losses = Losses(
        LossConfig(
            noise_pred_weight=float(loss_cfg.get("noise_pred_weight", 0.0)),
            snr_gamma=float(loss_cfg.get("snr_gamma", 0.0)),
            x0_ssim_weight=float(loss_cfg.get("x0_ssim_weight", 0.0)),
            x0_ssim_window=int(loss_cfg.get("x0_ssim_window", 11)),
            x0_l1_weight=float(
                loss_cfg.get("x0_l1_weight", loss_cfg.get("x0_fg_l1_weight", 0.0))
            ),
        )
    )

    grad_accum = int(train_cfg.get("grad_accumulation", 1))
    max_steps = int(train_cfg.get("max_steps", 1000))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    log_every = int(log_cfg.get("log_every_steps", 25))
    save_every = int(log_cfg.get("save_every_steps", 1000))
    val_every = int(log_cfg.get("validate_every_steps", 500))
    collage_inference_steps = int(log_cfg.get("collage_inference_steps", 50))

    if writer is not None:
        writer.add_text("experiment/notes", str(exp.get("notes", "")), 0)
        writer.add_text("experiment/tags", ",".join(exp.get("tags", [])), 0)
        # Full config for reproducibility
        writer.add_text("config/yaml", "```yaml\n" + yaml.safe_dump(cfg, sort_keys=False) + "\n```", 0)

        # Flattened hparams (TensorBoard hparams plugin prefers a flat dict)
        hparams = _flatten_for_hparams(cfg)
        # Ensure a couple of high-signal keys exist even if schema changes
        hparams.setdefault("experiment.name", run_name)
        hparams.setdefault("experiment.seed", seed)
        writer.add_hparams(hparams, {})

    net.train()
    step = 0
    opt.zero_grad(set_to_none=True)

    if resume:
        ckpt_path = _latest_checkpoint(run_name)
        if ckpt_path is not None:
            step = _load_checkpoint(ckpt_path, model=net, optim=opt, device=device)
            # Continue at the next step index.
            step = step + 1
            if writer is not None:
                writer.add_text("resume/checkpoint_path", ckpt_path, step)

    start = time.time()
    data_iter = iter(dl)

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=max_steps, desc="train", dynamic_ncols=True, initial=step)
    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)

        batch = _to_device(batch, device)
        autocast_dtype: Optional[torch.dtype] = None
        if precision == "bf16":
            autocast_dtype = torch.bfloat16
        elif precision == "fp32":
            autocast_dtype = None
        else:
            raise ValueError(f"Unsupported precision: {precision}")

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            control_images = _get_control_images(batch, sources)
            out = net(
                batch,
                control_images=tuple(control_images),
                control_scales=tuple(float(s) for s in scales),
            )
            # SNR-weighted eps MSE (optional) + optional x0 SSIM (pixel-space)
            pred_eps = out["pred_eps"]
            target_eps = out["target_eps"]

            # Net stores timesteps only internally; recompute them from batch? We already have them in forward.
            # Instead: mirror net.forward() by sampling timesteps/noisy_latents here is invasive.
            # So we compute SNR-weighted loss inside the model return dict (added below).
            loss, loss_logs = losses.noise_pred_snr(
                pred_eps,
                target_eps,
                timesteps=out["timesteps"],
                alphas_cumprod=net.noise_scheduler.alphas_cumprod,
            )

            # Optional x0 losses: decode predicted x0 (VAE params frozen).
            do_x0 = (float(loss_cfg.get("x0_ssim_weight", 0.0)) != 0.0) or (
                float(loss_cfg.get("x0_l1_weight", loss_cfg.get("x0_fg_l1_weight", 0.0))) != 0.0
            )
            if do_x0:
                noisy_latents = out["noisy_latents"]
                t = out["timesteps"]
                a = net.noise_scheduler.alphas_cumprod.to(device=device, dtype=noisy_latents.dtype).gather(0, t).view(-1, 1, 1, 1)
                pred_x0 = (noisy_latents - (1.0 - a).sqrt() * pred_eps) / a.sqrt().clamp(min=1e-8)
                lat = (pred_x0 / net.vae.config.scaling_factor).to(dtype=torch.float32)
                with torch.autocast(device_type=device.type, enabled=False):
                    bev_pred = net.vae.decode(lat).sample
                bev_pred = ((bev_pred + 1.0) / 2.0).clamp(0, 1)
                target_bev = batch["bev"].to(dtype=bev_pred.dtype)

                ssim_loss_val, ssim_logs = losses.x0_ssim(bev_pred, target_bev)
                loss = loss + ssim_loss_val
                loss_logs.update(ssim_logs)

                l1_val, l1_logs = losses.x0_l1(bev_pred, target_bev)
                loss = loss + l1_val
                loss_logs.update(l1_logs)

            loss = loss / grad_accum

        loss.backward()

        if (step + 1) % grad_accum == 0:
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

        if writer is not None and (step % log_every == 0):
            elapsed = time.time() - start
            writer.add_scalar("time/step", elapsed / max(1, step + 1), step)
            for k, v in loss_logs.items():
                writer.add_scalar(k, v, step)
            # Monitor GPU memory usage
            if device.type == "cuda":
                writer.add_scalar(
                    "gpu/memory_allocated_gb",
                    torch.cuda.memory_allocated(device) / 1024**3,
                    step,
                )
                writer.add_scalar(
                    "gpu/memory_reserved_gb",
                    torch.cuda.memory_reserved(device) / 1024**3,
                    step,
                )

        if writer is not None and (step % val_every == 0):
            net.eval()
            val_amp = torch.bfloat16 if precision == "bf16" else None
            with torch.autocast(device_type=device.type, dtype=val_amp, enabled=val_amp is not None):
                pred_bev = _validation_reconstruction(
                    net=net,
                    batch=batch,
                    control_images=control_images,
                    control_scales=[float(s) for s in scales],
                    device=device,
                    num_inference_steps=collage_inference_steps,
                    collage_seed_offset=seed + step,
                )
            _log_labeled_recon_collage(
                writer,
                tag="viz/recon_collage",
                step=step,
                batch=batch,
                pred_bev=pred_bev,
            )
            net.train()
            # Clear CUDA cache after validation to prevent memory fragmentation
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if step > 0 and (step % save_every == 0):
            ckpt_path = os.path.join("checkpoints", run_name, f"step_{step:07d}.pt")
            _save_checkpoint(ckpt_path, model=net, optim=opt, step=step)

        step += 1
        if pbar is not None:
            # Show a few high-signal numbers.
            postfix = {}
            if "loss/noise_pred" in loss_logs:
                postfix["loss"] = f"{loss_logs['loss/noise_pred']:.6f}"
            pbar.set_postfix(postfix)
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    if writer is not None:
        writer.flush()
        writer.close()


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_experiment.yaml")
    ap.add_argument("--resume", action="store_true", help="Auto-resume from latest checkpoint if present")
    args = ap.parse_args()
    train_model(args.config, resume=args.resume)


if __name__ == "__main__":
    _main()
