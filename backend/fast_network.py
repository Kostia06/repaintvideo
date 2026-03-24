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
    def __init__(self, num_resblocks: int = 5, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, c, kernel_size=7, stride=1),
            nn.InstanceNorm2d(c, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(c * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(c * 4, affine=True),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(*[ResidualBlock(c * 4) for _ in range(num_resblocks)])
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c * 4, c * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(c * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c * 2, c, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(c, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(c, 3, kernel_size=7, stride=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        out = self.residual(out)
        out = self.decoder(out)
        return (out + 1.0) * 127.5


class LightweightTransformNet(TransformNet):
    """Faster variant for real-time webcam (3 ResBlocks, half channels)."""

    def __init__(self) -> None:
        super().__init__(num_resblocks=3, base_channels=16)
