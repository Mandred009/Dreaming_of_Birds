"""SD 1.5 + (multi)ControlNet training modules.

This file is intentionally minimal: it only contains the SD + ControlNet logic
used by `train_model.py`.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewCrossAttentionBlock(nn.Module):
    """
    Multi-head attention across camera views: each view is a token; pooled (C,) per view.
    Output is a per-view channel bias broadcast over H×W and added to the residual map.
    """

    def __init__(self, in_channels: int, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.in_channels = in_channels
        self.proj_in = nn.Linear(in_channels, d_model)
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, C, H, W)
        b, v, c, h, w = x.shape
        pooled = x.mean(dim=(-2, -1))  # (B, V, C)
        t = self.proj_in(pooled)
        attn_out, _ = self.mha(t, t, t, need_weights=False)
        t = self.norm1(t + attn_out)
        t = self.norm2(t + self.ff(t))
        delta = self.proj_out(t).view(b, v, c, 1, 1)
        return x + delta


class LazyViewCrossAttention(nn.Module):
    """One ViewCrossAttentionBlock per distinct channel count (ControlNet down levels differ in C)."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self._blocks = nn.ModuleDict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = int(x.shape[2])
        key = str(c)
        if key not in self._blocks:
            blk = ViewCrossAttentionBlock(c, self.d_model, self.nhead, self.dropout)
            self._blocks[key] = blk.to(device=x.device, dtype=x.dtype)
        return self._blocks[key](x)


def _expand_time_ids(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    SDXL expects time_ids; for training we can use zeros as a simple default.
    Shape: (B, 6)
    """
    return torch.zeros((batch_size, 6), device=device)


@torch.no_grad()
def _encode_prompt_sd15(
    *,
    prompt: str,
    tokenizer,
    text_encoder,
    device: torch.device,
) -> torch.Tensor:
    toks = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)
    return text_encoder(toks.input_ids)[0]


class SD15MultiControlNetBEV(nn.Module):
    """
    SD 1.5 + ControlNet training module.

    Each entry in `control_images` uses one ControlNet (e.g. rgb vs depth). For a
    single multi-view tensor (B, V, 3, H, W), all views share that same ControlNet;
    per-view residuals are fused with a learned softmax weighting (scaled by V so
    uniform weights match the former plain sum). Optional multi-head attention across
    views is applied to residuals before fusion.
    """

    def __init__(
        self,
        *,
        pretrained: str,
        prompt: str = "a high quality image of a road scene as seen from a bird's eye view",
        local_files_only: bool = False,
        freeze_unet: bool = True,
        train_vae: bool = False,
        num_views: int = 6,
        use_view_embed: bool = True,
        view_encoding: str = "learned",  # learned | sinusoidal | none
        learned_view_fusion: bool = True,
        use_view_cross_attn: bool = True,
        view_attn_d_model: int = 256,
        view_attn_nhead: int = 8,
        view_attn_dropout: float = 0.0,
        # CFG-style training: randomly drop text / control so inference can use classifier-free guidance.
        cfg_dropout_prompt_p: float = 0.125,
        cfg_dropout_control_p: float = 0.125,
    ) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.prompt = prompt
        self.local_files_only = local_files_only
        self.freeze_unet = freeze_unet
        self.train_vae = bool(train_vae)
        self.cfg_dropout_prompt_p = float(cfg_dropout_prompt_p)
        self.cfg_dropout_control_p = float(cfg_dropout_control_p)
        self.num_views = int(num_views)
        self.learned_view_fusion = bool(learned_view_fusion)
        self.use_view_cross_attn = bool(use_view_cross_attn)
        self.view_attn_d_model = int(view_attn_d_model)
        self.view_attn_nhead = int(view_attn_nhead)
        self.view_attn_dropout = float(view_attn_dropout)
        # Back-compat: use_view_embed=False implies no view conditioning unless view_encoding overrides.
        self.view_encoding = str(view_encoding)
        if self.view_encoding not in ("learned", "sinusoidal", "none"):
            raise ValueError("view_encoding must be one of: learned | sinusoidal | none")
        self.use_view_embed = bool(use_view_embed) and self.view_encoding == "learned"
        self.view_embed: Optional[nn.Embedding] = None
        self._initialized = False
        # Logits over views -> softmax; fused residual is a weighted average over views (weights sum to 1).
        if self.learned_view_fusion and self.num_views > 0:
            self.view_fusion_logits = nn.Parameter(torch.zeros(self.num_views))
        else:
            self.view_fusion_logits = None

        if self.use_view_cross_attn:
            self.view_cross_attn = LazyViewCrossAttention(
                d_model=self.view_attn_d_model,
                nhead=self.view_attn_nhead,
                dropout=self.view_attn_dropout,
            )
        else:
            self.view_cross_attn = None

    def _view_attn_on_residual(self, dres: torch.Tensor, b0: int, v0: int) -> torch.Tensor:
        """Apply cross-view attention on (B*V,C,H,W) ControlNet residuals; returns same shape."""
        if self.view_cross_attn is None:
            return dres
        _, c, h, w = dres.shape
        x = dres.view(b0, v0, c, h, w)
        x = self.view_cross_attn(x)
        return x.reshape(b0 * v0, c, h, w)

    @staticmethod
    def _fuse_multiview_residuals(x: torch.Tensor, b0: int, v0: int, w: torch.Tensor) -> torch.Tensor:
        """x: (B*V, ...), w: (V,) nonnegative, typically softmax * v0 for scale preservation."""
        rest = x.shape[1:]
        x_bv = x.view(b0, v0, *rest)
        w_shape = (1, v0) + (1,) * len(rest)
        return (x_bv * w.view(w_shape)).sum(dim=1)

    @staticmethod
    def _sinusoidal_view_encoding(view_ids: torch.Tensor, *, dim: int) -> torch.Tensor:
        """
        Standard sinusoidal positional encoding for integer view ids.
        Returns (N, dim) float tensor.
        """
        if view_ids.ndim != 1:
            raise ValueError("view_ids must be 1D")
        device = view_ids.device
        half = dim // 2
        if half == 0:
            return torch.zeros((view_ids.shape[0], dim), device=device)
        inv_freq = torch.exp(
            -torch.log(torch.tensor(10000.0, device=device)) * torch.arange(0, half, device=device).float() / float(half)
        )  # (half,)
        pos = view_ids.to(dtype=torch.float32).unsqueeze(1)  # (N,1)
        sinus = pos * inv_freq.unsqueeze(0)  # (N,half)
        emb = torch.cat([torch.sin(sinus), torch.cos(sinus)], dim=1)  # (N,2*half)
        if emb.shape[1] < dim:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], dim - emb.shape[1]), device=device)], dim=1)
        return emb

    def _init(self, device: torch.device, n_controlnets: int) -> None:
        import sys
        from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        print(f"[MODEL] Initializing SD1.5 model on device {device}", flush=True)
        print(f"[MODEL] Pretrained: {self.pretrained}", flush=True)
        print(f"[MODEL] Local files only: {self.local_files_only}", flush=True)
        print(f"[MODEL] Number of ControlNets: {n_controlnets}", flush=True)

        print(f"[MODEL] Loading tokenizer...", end="", flush=True)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.pretrained, subfolder="tokenizer", local_files_only=self.local_files_only
        )
        print(f" ✓", flush=True)

        print(f"[MODEL] Loading text_encoder...", end="", flush=True)
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained, subfolder="text_encoder", local_files_only=self.local_files_only
        )
        print(f" ✓", flush=True)

        if self.use_view_embed:
            if self.num_views <= 0:
                raise ValueError("num_views must be > 0 when use_view_embed=True")
            hidden = int(getattr(self.text_encoder.config, "hidden_size"))
            self.view_embed = nn.Embedding(self.num_views, hidden)

        print(f"[MODEL] Loading VAE...", end="", flush=True)
        self.vae = AutoencoderKL.from_pretrained(self.pretrained, subfolder="vae", local_files_only=self.local_files_only)
        print(f" ✓", flush=True)

        print(f"[MODEL] Loading UNet...", end="", flush=True)
        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained, subfolder="unet", local_files_only=self.local_files_only
        )
        print(f" ✓", flush=True)

        print(f"[MODEL] Creating {n_controlnets} ControlNets from UNet...", flush=True)
        print(f"[MODEL] WARNING: ControlNet creation is slow/memory intensive. This may take 2-10 minutes.", flush=True)
        print(f"[MODEL] You can press Ctrl+C to cancel and try with fewer ControlNets.", flush=True)
        
        self.controlnets = nn.ModuleList()
        for i in range(n_controlnets):
            try:
                print(f"[MODEL]   Creating ControlNet {i+1}/{n_controlnets}...", end="", flush=True)
                # Create on CPU first to avoid GPU memory issues, then move to GPU later
                cn = ControlNetModel.from_unet(self.unet)
                self.controlnets.append(cn)
                print(f" ✓ (Size: {sum(p.numel() for p in cn.parameters()) / 1e6:.1f}M params)", flush=True)
            except KeyboardInterrupt:
                print(f"\n[MODEL] ✗ ControlNet creation cancelled by user", flush=True)
                raise
            except Exception as e:
                print(f"\n[MODEL] ✗ Failed to create ControlNet {i+1}: {e}", flush=True)
                raise

        print(f"[MODEL] Loading noise scheduler...", end="", flush=True)
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained, subfolder="scheduler", local_files_only=self.local_files_only
        )
        print(f" ✓", flush=True)

        print(f"[MODEL] Freezing pretrained models...", flush=True)
        self.vae.requires_grad_(self.train_vae)
        self.text_encoder.requires_grad_(False)
        # ControlNet-only training: always freeze UNet (and VAE/text encoders above).
        self.unet.requires_grad_(False)
        for cn in self.controlnets:
            cn.requires_grad_(True)
        if self.train_vae:
            print(f"[MODEL]   ✓ Frozen: UNet + text encoder; trainable: ControlNet + VAE", flush=True)
        else:
            print(f"[MODEL]   ✓ Frozen: UNet + VAE + text encoder; trainable: ControlNet", flush=True)

        # Memory savers
        print(f"[MODEL] Enabling memory optimizations...", flush=True)
        for i, cn in enumerate(self.controlnets):
            try:
                print(f"[MODEL]   ControlNet {i+1} gradient checkpointing...", end="", flush=True)
                cn.enable_gradient_checkpointing()
                print(f" ✓", flush=True)
            except Exception as e:
                print(f" ✗ ({e})", flush=True)
        
        try:
            print(f"[MODEL]   UNet gradient checkpointing...", end="", flush=True)
            self.unet.enable_gradient_checkpointing()
            print(f" ✓", flush=True)
        except Exception as e:
            print(f" ✗ ({e})", flush=True)
        
        try:
            print(f"[MODEL]   VAE slicing...", end="", flush=True)
            self.vae.enable_slicing()
            print(f" ✓", flush=True)
        except Exception as e:
            print(f" ✗ ({e})", flush=True)

        print(f"[MODEL] Moving model to device {device}...", end="", flush=True)
        self.to(device)
        print(f" ✓", flush=True)
        
        self._initialized = True
        print(f"[MODEL] Model initialization complete!", flush=True)

        self.register_buffer("_prompt_embeds", torch.empty(0), persistent=False)
        self.register_buffer("_prompt_embeds_empty", torch.empty(0), persistent=False)

    @torch.no_grad()
    def _encode_prompt(self, device: torch.device) -> None:
        self._prompt_embeds = _encode_prompt_sd15(
            prompt=self.prompt,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            device=device,
        )

    @torch.no_grad()
    def _encode_empty_prompt(self, device: torch.device) -> None:
        """Unconditional text embedding for CFG-style dropout (empty prompt)."""
        self._prompt_embeds_empty = _encode_prompt_sd15(
            prompt="",
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            device=device,
        )

    def initialize(self, device: torch.device, n_controlnets: int) -> None:
        if not self._initialized:
            print(f"[MODEL] Starting initialization (first-time setup)...", flush=True)
            self._init(device, n_controlnets=n_controlnets)
        
        if self._prompt_embeds.numel() == 0:
            print(f"[MODEL] Encoding prompt...", flush=True)
            self._encode_prompt(device)
            print(f"[MODEL] ✓ Prompt encoding complete", flush=True)
        if self._prompt_embeds_empty.numel() == 0:
            print(f"[MODEL] Encoding empty prompt (CFG dropout)...", flush=True)
            self._encode_empty_prompt(device)
            print(f"[MODEL] ✓ Empty prompt encoding complete", flush=True)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        control_images: Tuple[torch.Tensor, ...],
        control_scales: Optional[Tuple[float, ...]] = None,
    ) -> Dict[str, torch.Tensor]:
        if batch.get("bev") is None:
            raise KeyError("batch['bev'] is required")
        device = batch["bev"].device
        self.initialize(device, n_controlnets=len(control_images))

        if len(control_images) != len(self.controlnets):
            raise ValueError("control_images length must match number of controlnets")

        if control_scales is None:
            control_scales = tuple([1.0] * len(control_images))
        if len(control_scales) != len(control_images):
            raise ValueError("control_scales length must match control_images length")

        bev = batch["bev"]
        b = bev.shape[0]

        bev_in = bev * 2.0 - 1.0
        latents = self.vae.encode(bev_in).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (b,), device=device, dtype=torch.long
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.predict_eps(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            control_images=control_images,
            control_scales=control_scales,
            view_ids=batch.get("view_ids"),
        )

        return {"pred_eps": noise_pred, "target_eps": noise, "timesteps": timesteps, "noisy_latents": noisy_latents}

    def predict_eps(
        self,
        *,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        control_images: Tuple[torch.Tensor, ...],
        control_scales: Optional[Tuple[float, ...]] = None,
        view_ids: Optional[torch.Tensor] = None,
        force_empty_prompt: bool = False,
        force_zero_control: bool = False,
    ) -> torch.Tensor:
        """
        Predict epsilon for a provided (noisy_latents, timesteps) pair.

        For classifier-free guidance at inference, call twice with
        (force_empty_prompt=False, force_zero_control=False) vs (True, True), then combine.
        """
        device = noisy_latents.device
        b = noisy_latents.shape[0]

        if len(control_images) != len(self.controlnets):
            raise ValueError("control_images length must match number of controlnets")

        if control_scales is None:
            control_scales = tuple([1.0] * len(control_images))
        if len(control_scales) != len(control_images):
            raise ValueError("control_scales length must match control_images length")

        prompt_embeds = self._prompt_embeds.expand(b, -1, -1).clone()
        drop_ctrl = None
        if force_empty_prompt:
            prompt_embeds = self._prompt_embeds_empty.expand(b, -1, -1).to(dtype=prompt_embeds.dtype)
        elif self.training:
            pp = self.cfg_dropout_prompt_p
            if pp > 0:
                drop_p = torch.rand(b, device=device) < pp
                pe_empty = self._prompt_embeds_empty.expand(b, -1, -1).to(dtype=prompt_embeds.dtype)
                prompt_embeds = torch.where(drop_p[:, None, None], pe_empty, prompt_embeds)

        if force_zero_control:
            drop_ctrl = torch.ones(b, device=device, dtype=torch.bool)
        elif self.training:
            pc = self.cfg_dropout_control_p
            if pc > 0:
                drop_ctrl = torch.rand(b, device=device) < pc

        # Sum residuals across controlnets (and across views inside each control image)
        down_sum = None
        mid_sum = None
        for cn, img, scale in zip(self.controlnets, control_images, control_scales):
            # img can be (B,3,H,W) or (B,V,3,H,W)
            if img.ndim == 4:
                ctrl = img * 2.0 - 1.0
                if drop_ctrl is not None:
                    ctrl = ctrl * (~drop_ctrl).float().view(b, 1, 1, 1)
                cn_out = cn(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=ctrl,
                    return_dict=True,
                )
                down = [d * float(scale) for d in cn_out.down_block_res_samples]
                mid = cn_out.mid_block_res_sample * float(scale)
            elif img.ndim == 5:
                # Flatten views into batch for one CN call, then sum residuals over views.
                b0, v0, c0, h0, w0 = img.shape
                if b0 != b:
                    raise ValueError("control image batch mismatch")
                ctrl = (img * 2.0 - 1.0)
                if drop_ctrl is not None:
                    ctrl = ctrl * (~drop_ctrl).float().view(b0, 1, 1, 1, 1)
                ctrl = ctrl.reshape(b0 * v0, c0, h0, w0)

                lat_rep = noisy_latents.repeat_interleave(v0, dim=0)
                ts_rep = timesteps.repeat_interleave(v0, dim=0)
                pe_rep = prompt_embeds.repeat_interleave(v0, dim=0)
                if self.view_encoding != "none":
                    if view_ids is None:
                        view_ids_rep = (
                            torch.arange(v0, device=device, dtype=torch.long).unsqueeze(0).expand(b0, v0).reshape(b0 * v0)
                        )
                    else:
                        if view_ids.ndim == 1:
                            if int(view_ids.shape[0]) != int(v0):
                                raise ValueError("view_ids shape mismatch: expected (B,V) or (V,) matching control views")
                            view_ids_rep = (
                                view_ids.to(device=device, dtype=torch.long).unsqueeze(0).expand(b0, v0).reshape(b0 * v0)
                            )
                        elif view_ids.ndim == 2:
                            if tuple(view_ids.shape) != (b0, v0):
                                raise ValueError("view_ids shape mismatch: expected (B,V) matching control views")
                            view_ids_rep = view_ids.to(device=device, dtype=torch.long).reshape(b0 * v0)
                        else:
                            raise ValueError("view_ids must be shaped (B,V) or (V,)")

                    if int(view_ids_rep.min().item()) < 0:
                        raise ValueError("view_ids contains negative ids")
                    if self.num_views > 0 and int(view_ids_rep.max().item()) >= self.num_views:
                        raise ValueError(f"view_ids contains id >= num_views ({self.num_views})")

                    if self.view_encoding == "learned":
                        if self.view_embed is None:
                            raise RuntimeError("view_encoding='learned' but view_embed was not initialized")
                        ve = self.view_embed(view_ids_rep)  # (B*V, hidden)
                    elif self.view_encoding == "sinusoidal":
                        ve = self._sinusoidal_view_encoding(view_ids_rep, dim=int(pe_rep.shape[-1])).to(
                            device=device, dtype=pe_rep.dtype
                        )
                    else:
                        raise RuntimeError(f"Unexpected view_encoding={self.view_encoding}")

                    if drop_ctrl is not None:
                        # Drop view embedding when control is dropped; keep text tokens intact.
                        keep_v = (~drop_ctrl).repeat_interleave(v0).float().unsqueeze(-1)
                        ve = ve * keep_v
                    pe_rep = pe_rep + ve.unsqueeze(1)  # add per-view conditioning to all tokens
                cn_out = cn(
                    lat_rep,
                    ts_rep,
                    encoder_hidden_states=pe_rep,
                    controlnet_cond=ctrl,
                    return_dict=True,
                )

                # Fuse per-view residuals: same ControlNet, learned convex weights (normalized; sums to 1).
                if self.learned_view_fusion and self.view_fusion_logits is not None:
                    if int(v0) != int(self.num_views):
                        raise ValueError(
                            f"Multi-view control has V={v0} but num_views={self.num_views}; fix config or tensor shape."
                        )
                    w = F.softmax(self.view_fusion_logits, dim=0)
                else:
                    w = torch.full((v0,), 1.0 / float(v0), device=device, dtype=torch.float32)  # mean

                down = []
                for dres in cn_out.down_block_res_samples:
                    dres = self._view_attn_on_residual(dres, b0, v0)
                    dsum = self._fuse_multiview_residuals(dres, b0, v0, w.to(dtype=dres.dtype))
                    down.append(dsum * float(scale))
                mid_feat = self._view_attn_on_residual(cn_out.mid_block_res_sample, b0, v0)
                mid = self._fuse_multiview_residuals(
                    mid_feat, b0, v0, w.to(dtype=mid_feat.dtype)
                )
                mid = mid * float(scale)
            else:
                raise ValueError(f"Unsupported control image shape: {tuple(img.shape)}")

            if down_sum is None:
                down_sum = down
                mid_sum = mid
            else:
                down_sum = [a + b for a, b in zip(down_sum, down)]
                mid_sum = mid_sum + mid  # type: ignore[operator]

        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_sum,
            mid_block_additional_residual=mid_sum,
        ).sample
        return noise_pred

