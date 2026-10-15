"""A small U-Net written out explicitly: 4 levels, widths 32-64-128-256."""
import torch
import torch.nn as nn


def double_conv(cin, cout):
    """Two (3x3 conv -> BatchNorm -> ReLU). padding=1 keeps the spatial size."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=3, channels=(32, 64, 128, 256)):
        super().__init__()
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

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def build_unet(cfg):
    u = cfg["unet"]
    model = UNet(u["in_channels"], u["num_classes"], u["channels"])
    print(f"U-Net parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    return model
