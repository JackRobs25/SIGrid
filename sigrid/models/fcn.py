# sigrid/models/fcn.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG

__all__ = ["FCN", "VGGNet", "build_fcn"]

class FCN(nn.Module):
    def __init__(self, pretrained_net, n_class, input_channels):
        super().__init__()
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.input_channels = input_channels
        self.instance_norm2d = nn.InstanceNorm2d(self.input_channels, affine=True)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(128)

        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(64)

        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def crop_like(self, src, target):
        src_h, src_w = src.size(2), src.size(3)
        tgt_h, tgt_w = target.size(2), target.size(3)
        crop_h = (src_h - tgt_h) // 2
        crop_w = (src_w - tgt_w) // 2
        return src[:, :, crop_h:crop_h + tgt_h, crop_w:crop_w + tgt_w]

    def forward(self, x):
        x = self.instance_norm2d(x)
        output = self.pretrained_net(x)
        x3 = output['x3']  # (N, 256, H/8, W/8)
        x2 = output['x2']  # (N, 128, H/4, W/4)
        x1 = output['x1']  # (N, 64,  H/2, W/2)

        score = self.relu(self.deconv1(x3))         # (N, 128, H/4, W/4)
        x2 = self.crop_like(x2, score)
        score = self.bn1(score + x2)

        score = self.relu(self.deconv2(score))      # (N, 64, H/2, W/2)
        x1 = self.crop_like(x1, score)
        score = self.bn2(score + x1)

        score = self.relu(self.deconv3(score))      # (N, 32, H, W)
        score = self.bn3(score)

        score = self.classifier(score)              # (N, n_class, H, W)
        return score

class VGGNet(VGG):
    def __init__(self, pretrained=True, channels=3, model='vgg_light3', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model], channels))
        self.ranges = ranges[model]

        # SAFETY: only try to load torchvision weights for official models and channels==3
        if pretrained and channels == 3 and model in {"vgg11","vgg13","vgg16","vgg19"}:
            tv = getattr(models, model)(weights="IMAGENET1K_V1" if model in {"vgg11","vgg13"} else "IMAGENET1K_V1")
            self.load_state_dict(tv.state_dict(), strict=False)
        # else: vgg_light3 or non-3ch → skip pretrained

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:
            # delete FC layers to save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output[f"x{idx+1}"] = x
        return output

def make_layers(cfg_list, channels, batch_norm=False):
    layers = []
    in_channels = channels
    for v in cfg_list:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

ranges = {
    'vgg_light3': ((0, 5), (5, 10), (10, 15)),  # 3 pooling stages
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37)),
}

cfg = {
    'vgg_light3': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def build_fcn(
    input_channels: int,
    n_class: int = 1,
    backbone: str = "vgg_light3",
    pretrained: bool = False,
    requires_grad: bool = True,
):
    """
    Builder used by the CLI. Note:
    - If input_channels != 3, pretrained weights are skipped automatically.
    - For 'vgg_light3' there are no official torchvision weights; pretrained is ignored.
    """
    use_pretrained = pretrained and (input_channels == 3) and (backbone in {"vgg11","vgg13","vgg16","vgg19"})
    vgg = VGGNet(pretrained=use_pretrained, channels=input_channels, model=backbone,
                 requires_grad=requires_grad, remove_fc=True, show_params=False)
    return FCN(pretrained_net=vgg, n_class=n_class, input_channels=input_channels)