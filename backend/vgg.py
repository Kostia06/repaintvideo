import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

_VGG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_VGG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_for_vgg(tensor: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization before passing tensor to VGG-19."""
    device = tensor.device
    return (tensor - _VGG_MEAN.to(device)) / _VGG_STD.to(device)


class VGGFeatures(nn.Module):
    SLICE_POINTS = [3, 8, 17, 26, 35]

    def __init__(self) -> None:
        super().__init__()
        features = vgg19(weights=VGG19_Weights.DEFAULT).features

        for param in features.parameters():
            param.requires_grad = False

        slices: list[nn.Sequential] = []
        prev = 0
        for end in self.SLICE_POINTS:
            slices.append(nn.Sequential(*list(features.children())[prev:end]))
            prev = end

        self.slices = nn.ModuleList(slices)
        self.eval()

    def forward(self, x: torch.Tensor, already_normalized: bool = False) -> list[torch.Tensor]:
        if not already_normalized:
            x = normalize_for_vgg(x)
        outputs: list[torch.Tensor] = []
        h = x
        for s in self.slices:
            h = s(h)
            outputs.append(h)
        return outputs


get_vgg_features = VGGFeatures
