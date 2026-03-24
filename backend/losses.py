import torch
import torch.nn.functional as F
from torch import Tensor


def gram_matrix(x: Tensor) -> Tensor:
    b, c, h, w = x.size()
    features = x.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)


def style_loss(gen_features: list[Tensor], style_features: list[Tensor]) -> Tensor:
    loss = torch.tensor(0.0, device=gen_features[0].device)
    for gf, sf in zip(gen_features, style_features):
        loss = loss + F.mse_loss(gram_matrix(gf), gram_matrix(sf))
    return loss


def content_loss(gen_features: list[Tensor], content_features: list[Tensor]) -> Tensor:
    return F.mse_loss(gen_features[2], content_features[2])


def warp_loss(
    frame_styled: Tensor,
    prev_styled_warped: Tensor | None,
    mask: Tensor,
) -> Tensor:
    if prev_styled_warped is None:
        return torch.tensor(0.0, device=frame_styled.device)
    diff = (frame_styled - prev_styled_warped) ** 2
    return (diff * mask).mean()


def total_variation_loss(x: Tensor) -> Tensor:
    h_diff = (x[:, :, 1:, :] - x[:, :, :-1, :]) ** 2
    w_diff = (x[:, :, :, 1:] - x[:, :, :, :-1]) ** 2
    return h_diff.sum() + w_diff.sum()
