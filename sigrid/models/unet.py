# sigrid/models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UNet", "build_unet"]

class UNet(nn.Module):
    """
    UNet with two variants:
    - model="reduced": 2 downs / 2 ups (64→128→256 bottleneck)
    - model="full":    3 downs / 3 ups (64→128→256→512 bottleneck)

    Downsample can be 'pool' (MaxPool) or strided conv.
    Pads input to a multiple of 8 (reduced) or 16 (full) and crops back to original size.
    Output: 1-channel logits (use BCEWithLogitsLoss or equivalent).
    """
    def __init__(self, input_channels: int, model: str = "full", downsample: str = "pool"):
        super().__init__()
        self.input_channels = input_channels
        self.model = model
        self.downsample = downsample

        if "reduced" in self.model:
            self.e11 = nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1)
            self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.down1 = nn.MaxPool2d(2, 2) if self.downsample == "pool" else nn.Conv2d(64, 64, 2, 2)

            self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.down2 = nn.MaxPool2d(2, 2) if self.downsample == "pool" else nn.Conv2d(128, 128, 2, 2)

            self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

            self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.d11 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
            self.d12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

            self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.d21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
            self.d22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

            self.outconv = nn.Conv2d(64, 1, kernel_size=1)
        else:
            self.instance_norm2d = nn.InstanceNorm2d(self.input_channels, affine=True)

            self.e11 = nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1)
            self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.down1 = nn.MaxPool2d(2, 2) if self.downsample == "pool" else nn.Conv2d(64, 64, 2, 2)

            self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.down2 = nn.MaxPool2d(2, 2) if self.downsample == "pool" else nn.Conv2d(128, 128, 2, 2)

            self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.down3 = nn.MaxPool2d(2, 2) if self.downsample == "pool" else nn.Conv2d(256, 256, 2, 2)

            self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

            self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.d11 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
            self.d12 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

            self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.d21 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
            self.d22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

            self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.d31 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
            self.d32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

            self.outconv = nn.Conv2d(64, 1, kernel_size=1)

    @staticmethod
    def _crop_to_match(larger_tensor, reference_tensor):
        _, _, h, w = reference_tensor.size()
        _, _, H, W = larger_tensor.size()
        dh, dw = (H - h) // 2, (W - w) // 2
        return larger_tensor[:, :, dh:dh + h, dw:dw + w]

    @staticmethod
    def _pad_to_multiple(x, multiple: int):
        _, _, h, w = x.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        return F.pad(x, (0, pad_w, 0, pad_h)), h, w

    @staticmethod
    def _crop_to_original(x, h, w):
        return x[:, :, :h, :w]

    def forward(self, x):
        if "reduced" in self.model:
            x, orig_h, orig_w = self._pad_to_multiple(x, multiple=8)
            xe11 = F.relu(self.e11(x))
            xe12 = F.relu(self.e12(xe11))
            xp1 = self.down1(xe12)

            xe21 = F.relu(self.e21(xp1))
            xe22 = F.relu(self.e22(xe21))
            xp2 = self.down2(xe22)

            xe31 = F.relu(self.e31(xp2))
            xe32 = F.relu(self.e32(xe31))

            xu1 = self.upconv1(xe32)
            xe22 = self._crop_to_match(xe22, xu1)
            xu11 = torch.cat([xu1, xe22], dim=1)
            xd11 = F.relu(self.d11(xu11))
            xd12 = F.relu(self.d12(xd11))

            xu2 = self.upconv2(xd12)
            xe12 = self._crop_to_match(xe12, xu2)
            xu22 = torch.cat([xu2, xe12], dim=1)
            xd21 = F.relu(self.d21(xu22))
            xd22 = F.relu(self.d22(xd21))

            out = self.outconv(xd22)
            out = self._crop_to_original(out, orig_h, orig_w)
        else:
            x, orig_h, orig_w = self._pad_to_multiple(x, multiple=16)
            x = self.instance_norm2d(x)

            xe11 = F.relu(self.e11(x))
            xe12 = F.relu(self.e12(xe11))
            xp1 = self.down1(xe12)

            xe21 = F.relu(self.e21(xp1))
            xe22 = F.relu(self.e22(xe21))
            xp2 = self.down2(xe22)

            xe31 = F.relu(self.e31(xp2))
            xe32 = F.relu(self.e32(xe31))
            xp3 = self.down3(xe32)

            xe41 = F.relu(self.e41(xp3))
            xe42 = F.relu(self.e42(xe41))

            xu1 = self.upconv1(xe42)
            xe32 = self._crop_to_match(xe32, xu1)
            xu11 = torch.cat([xu1, xe32], dim=1)
            xd11 = F.relu(self.d11(xu11))
            xd12 = F.relu(self.d12(xd11))

            xu2 = self.upconv2(xd12)
            xe22 = self._crop_to_match(xe22, xu2)
            xu22 = torch.cat([xu2, xe22], dim=1)
            xd21 = F.relu(self.d21(xu22))
            xd22 = F.relu(self.d22(xd21))

            xu3 = self.upconv3(xd22)
            xe12 = self._crop_to_match(xe12, xu3)
            xu33 = torch.cat([xu3, xe12], dim=1)
            xd31 = F.relu(self.d31(xu33))
            xd32 = F.relu(self.d32(xd31))

            out = self.outconv(xd32)
            out = self._crop_to_original(out, orig_h, orig_w)

        return out


def build_unet(input_channels: int, model: str = "full", downsample: str = "pool") -> UNet:
    return UNet(input_channels=input_channels, model=model, downsample=downsample)