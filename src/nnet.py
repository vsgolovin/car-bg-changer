import torch
from torch import nn, Tensor
from torchvision import models


def get_conv_block(c1: int, c2: int, c3: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(c1, c2, 3, 1, 1, bias=False),
        nn.BatchNorm2d(c2),
        nn.ReLU(),
        nn.Conv2d(c2, c3, 3, 1, 1, bias=False),
        nn.BatchNorm2d(c3),
        nn.ReLU()
    )


class ResNetUNet(nn.Module):
    def __init__(self, weights: str | None = "DEFAULT", freeze: int = 5):
        super().__init__()
        rn = models.resnet34(weights=weights)
        self.encoder = nn.ModuleList([
            nn.Sequential(rn.conv1, rn.bn1, rn.relu),
            nn.Sequential(rn.maxpool, rn.layer1),
            rn.layer2,
            rn.layer3,
            rn.layer4
        ])
        # freeze several first layers
        for i, module in enumerate(self.encoder):
            if i + 1 <= freeze:
                for param in module.parameters():
                    param.requires_grad = False
        # decoder
        self.decoder = nn.ModuleList([
            get_conv_block(512, 256, 256),
            get_conv_block(512, 256, 128),
            get_conv_block(256, 128, 64),
            get_conv_block(128, 64, 64),
            get_conv_block(128, 64, 32),
        ])
        self.out_conv = nn.Sequential(
            get_conv_block(32 + 3, 16, 16),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        self.upscale = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x: Tensor) -> Tensor:
        skip_connections = []
        for module in self.encoder:
            skip_connections.append(x)
            x = module(x)
        for module in self.decoder:
            x = module(x)
            x = self.upscale(x)
            sc = skip_connections.pop()
            if x.shape != sc.shape:
                x = x[..., :, :sc.size(2), :sc.size(3)]
            x = torch.cat([sc, x], dim=1)
        return self.out_conv(x).squeeze(1)
