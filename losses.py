"""Loss functions used by diffusion training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 11,
    c1: float = 0.01 ** 2,
    c2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    Lightweight SSIM loss for (B,C,H,W) in [0,1].
    Returns 1 - SSIM averaged over batch and channels.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    if pred.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tuple(pred.shape)}")

    # Use average pooling as a fast, differentiable "window".
    pad = window_size // 2
    mu_x = F.avg_pool2d(pred, kernel_size=window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, kernel_size=window_size, stride=1, padding=pad)

    sigma_x = F.avg_pool2d(pred * pred, window_size, 1, pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, window_size, 1, pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, window_size, 1, pad) - mu_x * mu_y

    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2))
    return (1.0 - ssim.clamp(-1, 1)).mean()


@dataclass
class LossConfig:
    noise_pred_weight: float = 0.0
    # Min-SNR weighting for diffusion MSE (https://arxiv.org/abs/2303.09556).
    # If snr_gamma is None/0, weighting is disabled.
    snr_gamma: float = 0.0
    # Extra BEV SSIM loss computed from predicted x0 reconstruction.
    x0_ssim_weight: float = 0.0
    x0_ssim_window: int = 11
    # Dense L1 on predicted x0 BEV: every (H,W) position and channel vs GT (same grid).
    x0_l1_weight: float = 0.0
    # VAE autoencoding: decode(encode(bev)) vs bev in [-1,1]. Use when finetuning VAE.
    vae_recon_weight: float = 0.0
    vae_recon_type: str = "l1"  # l1 | l2
    vae_kl_weight: float = 0.0


class Losses:
    def __init__(self, cfg: Optional[LossConfig] = None) -> None:
        self.cfg = cfg or LossConfig()

    def noise_pred_snr(
        self,
        pred_eps: torch.Tensor,
        target_eps: torch.Tensor,
        *,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Epsilon MSE with optional Min-SNR weighting.
        - timesteps: (B,) long
        - alphas_cumprod: (T,) float on device
        """
        if pred_eps.shape != target_eps.shape:
            raise ValueError(f"Shape mismatch: pred={tuple(pred_eps.shape)} target={tuple(target_eps.shape)}")
        if timesteps.ndim != 1:
            raise ValueError("timesteps must be (B,)")
        b = pred_eps.shape[0]
        if int(timesteps.shape[0]) != int(b):
            raise ValueError("timesteps batch mismatch")

        mse_per = (pred_eps - target_eps).pow(2).mean(dim=(1, 2, 3))  # (B,)

        gamma = float(self.cfg.snr_gamma or 0.0)
        if gamma > 0:
            a = alphas_cumprod.to(device=pred_eps.device, dtype=pred_eps.dtype).gather(0, timesteps)
            snr = a / (1.0 - a).clamp(min=1e-8)
            w = (snr.clamp(max=gamma) / snr.clamp(min=1e-8)).to(dtype=mse_per.dtype)
            loss = (w * mse_per).mean()
            logs = {
                "loss/noise_pred": float((mse_per.mean()).detach().cpu()),
                "loss/noise_pred_snr": float(loss.detach().cpu()),
                "snr/mean": float(snr.mean().detach().cpu()),
                "snr/min": float(snr.min().detach().cpu()),
                "snr/max": float(snr.max().detach().cpu()),
            }
        else:
            loss = mse_per.mean()
            logs = {"loss/noise_pred": float(loss.detach().cpu())}

        loss = loss * self.cfg.noise_pred_weight
        logs["loss/noise_pred_weighted"] = float(loss.detach().cpu())
        return loss, logs

    def x0_ssim(self, pred_bev: torch.Tensor, target_bev: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """SSIM loss on BEV reconstruction (B,3,H,W) in [0,1]."""
        if self.cfg.x0_ssim_weight == 0:
            z = pred_bev.new_tensor(0.0)
            return z, {"loss/x0_ssim": float(z.detach().cpu())}
        s = ssim_loss(pred_bev, target_bev, window_size=int(self.cfg.x0_ssim_window))
        loss = s * float(self.cfg.x0_ssim_weight)
        return loss, {"loss/x0_ssim": float(loss.detach().cpu())}

    def x0_l1(self, pred_bev: torch.Tensor, target_bev: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Mean L1 between pred and GT BEV at every spatial position and channel.
        pred_bev, target_bev: (B,C,H,W) in [0,1], aligned grids.
        """
        if self.cfg.x0_l1_weight == 0:
            z = pred_bev.new_tensor(0.0)
            return z, {"loss/x0_l1": float(z.detach().cpu())}

        if pred_bev.shape != target_bev.shape:
            raise ValueError(f"Shape mismatch: pred={tuple(pred_bev.shape)} target={tuple(target_bev.shape)}")
        if pred_bev.ndim != 4:
            raise ValueError("Expected (B,C,H,W)")

        l1 = F.l1_loss(pred_bev, target_bev)
        loss = l1 * float(self.cfg.x0_l1_weight)
        return loss, {"loss/x0_l1": float(loss.detach().cpu())}

    def vae_recon_and_kl(self, vae: nn.Module, bev_in: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Standard VAE-style reconstruction in SD latent space (no scaling_factor in decode path).
        bev_in: (B,3,H,W) in [-1,1], same as diffusers encode input.
        """
        w = float(self.cfg.vae_recon_weight)
        kw = float(self.cfg.vae_kl_weight)
        if w == 0 and kw == 0:
            z = bev_in.new_tensor(0.0)
            return z, {"loss/vae_recon": 0.0, "loss/vae_kl": 0.0}

        post = vae.encode(bev_in).latent_dist
        total = bev_in.new_tensor(0.0)
        logs: Dict[str, float] = {"loss/vae_recon": 0.0, "loss/vae_kl": 0.0}

        if w > 0:
            z = post.sample()
            recon = vae.decode(z).sample
            rtype = str(self.cfg.vae_recon_type).lower()
            if rtype == "l2":
                r = F.mse_loss(recon, bev_in)
            elif rtype == "l1":
                r = F.l1_loss(recon, bev_in)
            else:
                raise ValueError("vae_recon_type must be l1 or l2")
            total = total + r * w
            logs["loss/vae_recon"] = float((r * w).detach().cpu())

        if kw > 0:
            kl = post.kl().mean()
            total = total + kl * kw
            logs["loss/vae_kl"] = float((kl * kw).detach().cpu())

        return total, logs
