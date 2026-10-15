"""Model C: Stable Diffusion 1.5 + ControlNet that turns a sparse BEV map into a BEV class map.

The class map is painted as a 3-colour image, so the frozen SD VAE can encode it to a 4x32x32 latent.
The ControlNet (trained) sees the sparse map and steers the frozen SD UNet while it denoises.
"""
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (AutoencoderKL, ControlNetModel, DDIMScheduler, DDPMScheduler,
                       UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer

from viz import TARGET_COLORS   # bg black, vehicle red, pedestrian green: far apart in RGB

PALETTE = torch.from_numpy(TARGET_COLORS).float()   # [3,3] in 0..255


def paint(target, size):
    """target [B,200,200] int tensor of classes -> RGB image [B,3,size,size] in [-1,1].
    Nearest-neighbour resize keeps the colours exactly the 3 palette colours."""
    img = PALETTE.to(target.device)[target.long()]            # [B,H,W,3]
    img = img.permute(0, 3, 1, 2)
    img = F.interpolate(img, size=(size, size), mode="nearest")
    return img / 127.5 - 1.0


def snap(img, out_size=200):
    """Decoded image [B,3,S,S] in [-1,1] -> class map [B,out_size,out_size].
    The VAE output is never exactly a palette colour, so each pixel takes the class of the
    nearest palette colour (L2 distance). Then nearest-resize back to the BEV grid."""
    rgb = (img.float().clamp(-1, 1) + 1) * 127.5                         # [B,3,S,S] in 0..255
    dist = ((rgb.permute(0, 2, 3, 1)[:, :, :, None, :] - PALETTE.to(img.device)) ** 2).sum(-1)
    cls = dist.argmin(-1)                                                 # [B,S,S]
    # Go back to the BEV grid by picking, for each BEV cell j, the first big-image pixel that the
    # forward nearest-resize copied from j (ceil(j*S/out)). Resizing back with a plain nearest
    # resize would be off by up to one cell (it picks a neighbour), which costs ~0.4 vehicle IoU.
    src = torch.ceil(torch.arange(out_size, device=cls.device) * cls.shape[-1] / out_size).long()
    return cls[:, src][:, :, src]


def load_vae(cfg, device="cuda"):
    return AutoencoderKL.from_pretrained(cfg["diffusion"]["base_model"], subfolder="vae",
                                         torch_dtype=torch.float16, variant="fp16").to(device).eval()


@torch.no_grad()
def encode_images(vae, img, bs=16):
    """img [N,3,S,S] in [-1,1] -> latents [N,4,S/8,S/8] float32 (scaled the way SD expects).
    We use the mean of the VAE posterior (deterministic) instead of a random sample."""
    out = []
    for i in range(0, len(img), bs):
        x = img[i:i + bs].to(vae.device, torch.float16)
        out.append(vae.encode(x).latent_dist.mean.float() * vae.config.scaling_factor)
    return torch.cat(out)


@torch.no_grad()
def decode_latents(vae, latents, bs=16):
    out = []
    for i in range(0, len(latents), bs):
        z = latents[i:i + bs].to(torch.float16) / vae.config.scaling_factor
        out.append(vae.decode(z).sample.float())
    return torch.cat(out)


@torch.no_grad()
def encode_prompt(cfg, device="cuda"):
    """Encode the one fixed prompt once. The text encoder is deleted afterwards (frees ~250 MB VRAM)."""
    name = cfg["diffusion"]["base_model"]
    tok = CLIPTokenizer.from_pretrained(name, subfolder="tokenizer")
    enc = CLIPTextModel.from_pretrained(name, subfolder="text_encoder", torch_dtype=torch.float16,
                                        variant="fp16").to(device)
    ids = tok(cfg["diffusion"]["prompt"], padding="max_length", max_length=tok.model_max_length,
              truncation=True, return_tensors="pt").input_ids.to(device)
    emb = enc(ids)[0].float().cpu()      # [1,77,768]
    del enc, tok
    torch.cuda.empty_cache()
    return emb


class BEVDiffusion(torch.nn.Module):
    def __init__(self, cfg, prompt_emb, device="cuda"):
        super().__init__()
        d = cfg["diffusion"]
        self.size, self.device, self.ddim_steps = d["image_size"], device, d["ddim_steps"]
        self.prompt_emb = prompt_emb.to(device, torch.bfloat16)
        self.vae = load_vae(cfg, device)
        unet = UNet2DConditionModel.from_pretrained(d["base_model"], subfolder="unet",
                                                    torch_dtype=torch.float32, variant="fp16")
        # ControlNet = trainable copy of the UNet encoder + a small conv net for the 4-channel
        # control image. Built from the fp32 UNet so it starts as a copy of SD's weights.
        self.controlnet = ControlNetModel.from_unet(unet, conditioning_channels=4).to(device)
        # Frozen UNet in bf16 (not fp16): gradients flow back *through* it to the ControlNet,
        # and bf16 has fp32's range, so no loss scaling is needed and no overflow.
        self.unet = unet.to(device, torch.bfloat16).requires_grad_(False)
        self.controlnet.enable_gradient_checkpointing()
        self.unet.enable_gradient_checkpointing()   # saves activation memory in the backward pass
        self.vae.requires_grad_(False)
        blank = paint(torch.zeros(1, 200, 200, dtype=torch.long, device=device), self.size)   # all background
        self.blank_latent = encode_images(self.vae, blank)
        self.train_sched = DDPMScheduler.from_pretrained(d["base_model"], subfolder="scheduler")
        self.ddim = DDIMScheduler.from_pretrained(d["base_model"], subfolder="scheduler")
        self.train_sched.alphas_cumprod = self.train_sched.alphas_cumprod.to(device)

    def control_image(self, sparse):
        """sparse uint8/float [B,4,200,200] with values 0/1 -> float [B,4,S,S]."""
        x = sparse.to(self.device).float()
        return F.interpolate(x, size=(self.size, self.size), mode="nearest")

    def denoise_step(self, z, t, cond):
        """Predict the noise in z at timestep t. The ControlNet looks at the sparse map and its
        residuals are added into the frozen UNet's skip connections and middle block."""
        emb = self.prompt_emb.expand(len(z), -1, -1)
        down, mid = self.controlnet(z, t, encoder_hidden_states=emb, controlnet_cond=cond, return_dict=False)
        return self.unet(z, t, encoder_hidden_states=emb, down_block_additional_residuals=down,
                         mid_block_additional_residual=mid).sample

    def training_loss(self, latents, sparse):
        """Standard epsilon-prediction loss: noise the clean target latent at a random timestep,
        ask the model to predict the noise, MSE. latents [B,4,32,32], sparse [B,4,200,200]."""
        latents = latents.to(self.device)
        noise = torch.randn_like(latents)
        t = torch.randint(0, self.train_sched.config.num_train_timesteps, (len(latents),), device=self.device)
        z = self.train_sched.add_noise(latents, noise, t)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.denoise_step(z, t, self.control_image(sparse))
        return F.mse_loss(pred.float(), noise)

    @torch.no_grad()
    def predict(self, sparse, num_samples=1, seed=None):
        """sparse [B,4,200,200] -> class maps [B,200,200] (long, on the device).
        Each sample starts from different random noise. With num_samples>1 take the per-pixel
        majority vote over the class maps."""
        self.eval()
        gen = None if seed is None else torch.Generator(self.device).manual_seed(seed)
        cond = self.control_image(sparse).repeat_interleave(num_samples, 0)   # [B*S,4,H,W]
        noise = torch.randn(len(cond), 4, self.size // 8, self.size // 8, device=self.device, generator=gen)
        self.ddim.set_timesteps(self.ddim_steps)
        # Start point. SD's noise schedule does not reach pure noise at the last training timestep:
        # a training input there still contains sqrt(alpha_bar_T) ~ 7% of the clean latent, and our
        # latents have a large non-zero mean, so the model learned to read that faint offset.
        # Pure randn at inference lacks it (train frames: vehicle IoU 0.05, garbage). So start from
        # the blank-map latent noised to the first sampling timestep, like a training input of an
        # (almost) empty map. It carries no information about the frame being predicted.
        blank = self.blank_latent.expand(len(cond), -1, -1, -1)
        z = self.ddim.add_noise(blank, noise, self.ddim.timesteps[:1].expand(len(cond)).to(self.device))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for t in self.ddim.timesteps:
                eps = self.denoise_step(z, t.to(self.device).expand(len(z)), cond)
                z = self.ddim.step(eps.float(), t, z).prev_sample
        cls = snap(decode_latents(self.vae, z))                               # [B*S,200,200]
        cls = cls.view(len(sparse), num_samples, *cls.shape[1:])
        votes = F.one_hot(cls, 3).sum(1)                                      # [B,H,W,3] votes per class
        return votes.argmax(-1)    # ties go to the lowest class index (background)
