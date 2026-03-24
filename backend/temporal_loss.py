import torch

from warp import warp_frame


def temporal_consistency_loss(
    styled_curr: torch.Tensor,
    styled_prev: torch.Tensor,
    flow: torch.Tensor,
    occlusion_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize style differences between consecutive frames in visible regions.

    Args:
        styled_curr: (B, C, H, W) current styled frame.
        styled_prev: (B, C, H, W) previous styled frame.
        flow: (B, 2, H, W) optical flow from prev to curr.
        occlusion_mask: (B, 1, H, W) where 1 = visible, 0 = occluded.

    Returns:
        Scalar loss tensor.
    """
    warped_prev = warp_frame(styled_prev, flow)
    diff = (styled_curr - warped_prev) ** 2
    masked_diff = diff * occlusion_mask
    visible_count = occlusion_mask.sum().clamp(min=1.0)
    return masked_diff.sum() / visible_count
