import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

FLOW_DOWNSCALE = 320


class FlowEstimator:
    """RAFT optical flow wrapper with lazy model loading."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: torch.nn.Module | None = None

    def _load_model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(self.device)
            self._model.eval()
        return self._model

    @torch.no_grad()
    def compute_flow(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dense optical flow from frame1 to frame2.

        Args:
            frame1: (B, 3, H, W) in [0, 1]
            frame2: (B, 3, H, W) in [0, 1]

        Returns:
            Flow field (B, 2, H, W) — pixel displacement vectors.
        """
        model = self._load_model()
        orig_h, orig_w = frame1.shape[2:]

        f1 = frame1.to(self.device)
        f2 = frame2.to(self.device)

        if self.device == "cpu" and max(orig_h, orig_w) > FLOW_DOWNSCALE:
            scale = FLOW_DOWNSCALE / max(orig_h, orig_w)
            h_small = int(orig_h * scale) // 8 * 8
            w_small = int(orig_w * scale) // 8 * 8
            f1 = F.interpolate(f1, (h_small, w_small), mode="bilinear", align_corners=False)
            f2 = F.interpolate(f2, (h_small, w_small), mode="bilinear", align_corners=False)
        else:
            h_small = orig_h // 8 * 8
            w_small = orig_w // 8 * 8
            if h_small != orig_h or w_small != orig_w:
                f1 = F.interpolate(f1, (h_small, w_small), mode="bilinear", align_corners=False)
                f2 = F.interpolate(f2, (h_small, w_small), mode="bilinear", align_corners=False)

        f1_raft = f1 * 2.0 - 1.0
        f2_raft = f2 * 2.0 - 1.0

        flow_predictions = model(f1_raft, f2_raft)
        flow = flow_predictions[-1]

        if flow.shape[2:] != (orig_h, orig_w):
            flow = F.interpolate(flow, (orig_h, orig_w), mode="bilinear", align_corners=False)
            flow[:, 0] *= orig_w / flow.shape[3] if flow.shape[3] != orig_w else 1.0
            flow[:, 1] *= orig_h / flow.shape[2] if flow.shape[2] != orig_h else 1.0

        return flow
