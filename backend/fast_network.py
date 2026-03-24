import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        upsample: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        if upsample:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=stride - 1,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return out + residual


class TransformNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, 32, kernel_size=7, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),
        )
        self.residual = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])
        self.decoder = nn.Sequential(
            ConvBlock(128, 64, kernel_size=3, stride=2, upsample=True),
            ConvBlock(64, 32, kernel_size=3, stride=2, upsample=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        out = self.residual(out)
        out = self.decoder(out)
        out = self.final(out)
        return (out + 1.0) * 127.5
