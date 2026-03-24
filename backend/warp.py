import torch
import torch.nn.functional as F


def warp_frame(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp frame using optical flow via grid_sample.

    Args:
        frame: (B, C, H, W) source frame.
        flow: (B, 2, H, W) pixel displacement vectors.

    Returns:
        Warped frame (B, C, H, W).
    """
    B, _, H, W = flow.shape

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=flow.device, dtype=torch.float32),
        torch.arange(W, device=flow.device, dtype=torch.float32),
        indexing="ij",
    )

    sample_x = grid_x + flow[:, 0]
    sample_y = grid_y + flow[:, 1]

    sample_x = 2.0 * sample_x / (W - 1) - 1.0
    sample_y = 2.0 * sample_y / (H - 1) - 1.0

    grid = torch.stack([sample_x, sample_y], dim=-1)

    return F.grid_sample(
        frame,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def compute_occlusion_mask(
    flow_forward: torch.Tensor,
    flow_backward: torch.Tensor,
    alpha1: float = 0.01,
    alpha2: float = 0.5,
) -> torch.Tensor:
    """Forward-backward consistency check for occlusion detection.

    Args:
        flow_forward: (B, 2, H, W) flow from frame1 to frame2.
        flow_backward: (B, 2, H, W) flow from frame2 to frame1.
        alpha1: relative threshold factor.
        alpha2: absolute threshold (pixels squared).

    Returns:
        Mask (B, 1, H, W) where 1 = visible, 0 = occluded.
    """
    warped_backward = warp_frame(flow_backward, flow_forward)

    round_trip = flow_forward + warped_backward

    round_trip_sq = (round_trip ** 2).sum(dim=1, keepdim=True)
    forward_sq = (flow_forward ** 2).sum(dim=1, keepdim=True)
    backward_sq = (warped_backward ** 2).sum(dim=1, keepdim=True)

    threshold = alpha1 * (forward_sq + backward_sq) + alpha2

    visible = (round_trip_sq < threshold).float()
    return visible
