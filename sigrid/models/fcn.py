# sigrid/models/fcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SimpleFCN", "build_fcn"]

class SimpleFCN(nn.Module):
    """
    Minimal FCN for SIGrid inputs (C x H x W) -> 1 x H x W logits.
    Encoder: strided convs
    Decoder: transposed convs
    """
    def __init__(self, in_ch: int = 3, base: int = 64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.down1 = nn.Conv2d(base, base, 2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.down2 = nn.Conv2d(base*2, base*2, 2, stride=2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base*2, base*4, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base*4, base*4, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True)
        )

        self.out = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        x = self.down1(e1)
        e2 = self.enc2(x)
        x = self.down2(e2)
        x = self.bottleneck(x)
        x = self.up2(x)
        x = self.dec2(x)
        x = self.up1(x)
        x = self.dec1(x)
        return self.out(x)

def build_fcn(input_channels: int, base: int = 64) -> SimpleFCN:
    return SimpleFCN(in_ch=input_channels, base=base)