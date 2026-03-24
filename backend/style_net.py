"""TransformerNet from pytorch/examples/fast_neural_style.

Architecture matches pretrained checkpoints (mosaic, candy, rain_princess, udnie)
downloaded from https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip.
"""
import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(self.reflection_pad(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(track_running_stats=True, num_features=channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(track_running_stats=True, num_features=channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, upsample: int | None = None) -> None:
        super().__init__()
        self.upsample = upsample
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = nn.functional.interpolate(x, mode="nearest", scale_factor=self.upsample)
        return self.conv2d(self.reflection_pad(x))


class TransformerNet(nn.Module):
    """Exact architecture from pytorch/examples — compatible with pretrained .pth files."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(track_running_stats=True, num_features=32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(track_running_stats=True, num_features=64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(track_running_stats=True, num_features=128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(track_running_stats=True, num_features=64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(track_running_stats=True, num_features=32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y
