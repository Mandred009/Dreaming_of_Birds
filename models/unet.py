"""A small U-Net written out explicitly: 4 levels, widths 32-64-128-256."""
import math
import torch
import torch.nn as nn


def double_conv(cin, cout):
    """Two (3x3 conv -> BatchNorm -> ReLU). padding=1 keeps the spatial size."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def timestep_embedding(t, dim):
    """Sinusoidal embedding of the integer diffusion step t [B] -> [B, dim] (Transformer-style:
    sin/cos at geometrically spaced frequencies, so nearby steps get similar vectors and the
    network can tell 'a little noisy' from 'pure noise')."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class UNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=3, channels=(32, 64, 128, 256), time_dim=None):
        """time_dim=None: the plain baseline-B U-Net (no new parameters, same state_dict keys).
        time_dim=int: also takes a diffusion step t; every conv block gets a per-channel bias
        computed from t (used by pixel diffusion, E4)."""
        super().__init__()
        self.time_dim = time_dim
        c1, c2, c3, c4 = channels
        # Encoder: conv block, then halve the resolution with max-pool.
        self.enc1 = double_conv(in_channels, c1)   # 200x200
        self.enc2 = double_conv(c1, c2)            # 100x100
        self.enc3 = double_conv(c2, c3)            # 50x50
        self.enc4 = double_conv(c3, c4)            # 25x25 (bottleneck)
        self.pool = nn.MaxPool2d(2)
        # Decoder: transposed conv doubles resolution, then concat with the encoder
        # feature of the same size (skip connection) and convolve again.
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = double_conv(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = double_conv(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = double_conv(c1 * 2, c1)
        self.head = nn.Conv2d(c1, num_classes, 1)  # per-pixel class logits

        if time_dim is not None:
            # t -> sinusoidal vector -> small MLP -> one Linear per block giving a per-channel bias.
            self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
            widths = dict(enc1=c1, enc2=c2, enc3=c3, enc4=c4, dec3=c3, dec2=c2, dec1=c1)
            self.time_proj = nn.ModuleDict({k: nn.Linear(time_dim, w) for k, w in widths.items()})

    def block(self, name, x, temb):
        """Run a double_conv. With a time embedding, add it (as a per-channel bias) after the
        first conv-BN-ReLU, so both the second conv and everything after it knows the noise level.
        Layers 0-2 are the first conv/BN/ReLU, layers 3-5 the second (see double_conv)."""
        b = getattr(self, name)
        if temb is None:
            return b(x)
        x = b[:3](x)
        x = x + self.time_proj[name](temb)[:, :, None, None]
        return b[3:](x)

    def forward(self, x, t=None):
        temb = None
        if self.time_dim is not None:
            temb = self.time_mlp(timestep_embedding(t, self.time_dim))
        e1 = self.block("enc1", x, temb)
        e2 = self.block("enc2", self.pool(e1), temb)
        e3 = self.block("enc3", self.pool(e2), temb)
        e4 = self.block("enc4", self.pool(e3), temb)
        d3 = self.block("dec3", torch.cat([self.up3(e4), e3], dim=1), temb)
        d2 = self.block("dec2", torch.cat([self.up2(d3), e2], dim=1), temb)
        d1 = self.block("dec1", torch.cat([self.up1(d2), e1], dim=1), temb)
        return self.head(d1)


def build_unet(cfg):
    u = cfg["unet"]
    model = UNet(u["in_channels"], u["num_classes"], u["channels"])
    print(f"U-Net parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    return model


def build_denoiser(cfg):
    """E4 denoiser: the SAME backbone widths as B, but 7 input channels (3 noisy-map + 4 sparse)
    and a timestep embedding. Output: 3 channels = class logits for the clean map."""
    p = cfg["pixel_diffusion"]
    extra = cfg["unet"]["num_classes"] if p.get("condition_on_unet") else 0   # E5: + B's softmax probs
    model = UNet(3 + cfg["unet"]["in_channels"] + extra, cfg["unet"]["num_classes"], cfg["unet"]["channels"], time_dim=p["time_dim"])
    print(f"Denoiser parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    return model
