import shutil
import subprocess
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import onnxruntime as ort
import torch

from style_net import TransformerNet

STYLE_MODELS: dict[str, str] = {
    "monet": "models/weights/monet.onnx",
    "starry_night": "models/weights/starry_night.onnx",
    "cyberpunk": "models/weights/cyberpunk.onnx",
    "ukiyo_e": "models/weights/ukiyo_e.onnx",
    "anime": "models/weights/anime.onnx",
}

NEURAL_MODELS: dict[str, str] = {
    "mosaic": "models/pretrained/mosaic.pth",
    "candy": "models/pretrained/candy.pth",
    "rain_princess": "models/pretrained/rain_princess.pth",
    "udnie": "models/pretrained/udnie.pth",
}

BASE_DIR = Path(__file__).resolve().parent.parent
MIN_OUTPUT_SIZE = 720
HAS_FFMPEG = shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vignette(frame: np.ndarray, strength: float = 0.4) -> np.ndarray:
    h, w = frame.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    mask = 1.0 - strength * (((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
    mask = np.clip(mask, 0, 1).astype(np.float32)
    return (frame.astype(np.float32) * mask[:, :, np.newaxis]).astype(np.uint8)


def _compute_target_size(h: int, w: int) -> tuple[int, int]:
    if max(h, w) >= MIN_OUTPUT_SIZE:
        return w, h
    scale = MIN_OUTPUT_SIZE / max(h, w)
    return int(w * scale), int(h * scale)


# ---------------------------------------------------------------------------
# Demo filters — improved
# ---------------------------------------------------------------------------

def _demo_monet(frame: np.ndarray) -> np.ndarray:
    styled = cv2.stylization(frame, sigma_s=60, sigma_r=0.07)
    bilateral = cv2.bilateralFilter(frame, 9, 75, 75)
    blended = cv2.addWeighted(styled, 0.6, bilateral, 0.4, 0)
    hsv = cv2.cvtColor(blended, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + 8) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    warm = np.full_like(result, (200, 220, 240), dtype=np.uint8)
    result = cv2.addWeighted(result, 0.92, warm, 0.08, 0)
    return _vignette(result, 0.3)


def _demo_starry_night(frame: np.ndarray) -> np.ndarray:
    base = cv2.detailEnhance(frame, sigma_s=20, sigma_r=0.15)
    h, w = frame.shape[:2]
    np.random.seed(42)
    dx = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32) * 8, (0, 0), 12)
    dy = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32) * 8, (0, 0), 12)
    map_x = (np.arange(w)[np.newaxis, :] + dx).astype(np.float32)
    map_y = (np.arange(h)[:, np.newaxis] + dy).astype(np.float32)
    warped = cv2.remap(base, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] * 0.6 + 55, 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edges = np.clip(edges / edges.max() * 255, 0, 255).astype(np.uint8)
    edges = cv2.GaussianBlur(edges, (5, 5), 0)
    edge_mask = (edges.astype(np.float32) / 255.0)[:, :, np.newaxis]
    result = (result.astype(np.float32) * (1.0 - edge_mask * 0.6)).astype(np.uint8)
    return result


def _demo_cyberpunk(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    result = cv2.convertScaleAbs(result, alpha=1.4, beta=-20)
    h, w = result.shape[:2]
    shifted = np.zeros_like(result)
    shifted[:, 3:, 2] = result[:, :-3, 2]
    shifted[:, :-3, 0] = result[:, 3:, 0]
    shifted[:, :, 1] = result[:, :, 1]
    result = shifted
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    glow = cv2.GaussianBlur(bright, (21, 21), 0)
    glow_pink = np.zeros_like(result)
    glow_pink[:, :, 1] = 0
    glow_pink[:, :, 2] = glow
    glow_pink[:, :, 0] = (glow * 0.7).astype(np.uint8)
    result = cv2.add(result, glow_pink)
    result[::3, :] = (result[::3, :].astype(np.float32) * 0.82).astype(np.uint8)
    lum = gray.astype(np.float32) / 255.0
    shadow_mask = (1.0 - lum)[:, :, np.newaxis]
    teal = np.array([180, 140, 40], dtype=np.float32)
    result = (result.astype(np.float32) + shadow_mask * teal * 0.2).clip(0, 255).astype(np.uint8)
    return result


def _demo_ukiyo_e(frame: np.ndarray) -> np.ndarray:
    # 1. Flatten texture
    smooth = cv2.bilateralFilter(frame, 15, 80, 80)
    smooth = cv2.bilateralFilter(smooth, 15, 80, 80)
    # 2. K-means quantize to 8 colors
    h, w = smooth.shape[:2]
    scale = min(1.0, 256.0 / max(h, w))
    small = cv2.resize(smooth, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 8, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    quantized_small = centers[labels.flatten()].reshape(small.shape).astype(np.uint8)
    quantized = cv2.resize(quantized_small, (w, h), interpolation=cv2.INTER_NEAREST)
    quantized = cv2.bilateralFilter(quantized, 9, 150, 150)
    # 3. Edge detect on quantized (not original)
    gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    # 4. Composite: ink outlines over flat colors
    result = quantized.copy()
    result[edges > 0] = (15, 15, 15)
    # 5. Warm paper tone + desaturate
    warm_paper = np.full_like(result, (195, 215, 230), dtype=np.uint8)
    result = cv2.addWeighted(result, 0.95, warm_paper, 0.05, 0)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.85, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _demo_anime(frame: np.ndarray) -> np.ndarray:
    # 1. Flatten micro-texture (fur, fabric) with strong bilateral
    smooth = cv2.bilateralFilter(frame, 15, 80, 80)
    smooth = cv2.bilateralFilter(smooth, 15, 80, 80)
    # 2. K-means quantize to 8 flat colors
    h, w = smooth.shape[:2]
    scale = min(1.0, 256.0 / max(h, w))
    small = cv2.resize(smooth, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 8, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    quantized_small = centers[labels.flatten()].reshape(small.shape).astype(np.uint8)
    quantized = cv2.resize(quantized_small, (w, h), interpolation=cv2.INTER_NEAREST)
    # 3. Edge detect on quantized image (not original)
    gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    # 4. Composite: edge pixels become near-black over flat colors
    edge_mask = edges[:, :, np.newaxis] > 0
    result = quantized.copy()
    result[edge_mask.squeeze()] = (20, 20, 20)
    # 5. Saturation boost
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _demo_watercolor(frame: np.ndarray) -> np.ndarray:
    base = cv2.edgePreservingFilter(frame, flags=1, sigma_s=60, sigma_r=0.4)
    soft = cv2.GaussianBlur(base, (15, 15), 0)
    result = cv2.addWeighted(base, 0.7, soft, 0.3, 0)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.9, 0, 255)
    hsv[:, :, 0] = (hsv[:, :, 0] + 5) % 180
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    h, w = result.shape[:2]
    np.random.seed(123)
    grain = np.random.normal(0, 8, (h, w)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (3, 3), 0)
    grain_bgr = np.stack([grain] * 3, axis=-1)
    result = np.clip(result.astype(np.float32) + grain_bgr, 0, 255).astype(np.uint8)
    return _vignette(result, 0.25)


def _demo_pixel_art(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    pixel_w = 80
    pixel_h = max(1, int(h * (pixel_w / w)))
    small = cv2.resize(frame, (pixel_w, pixel_h), interpolation=cv2.INTER_AREA)
    quantized = ((small >> 4) << 4).astype(np.uint8)
    hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
    quantized = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv2.resize(quantized, (w, h), interpolation=cv2.INTER_NEAREST)


def _demo_oil_painting(frame: np.ndarray) -> np.ndarray:
    styled = cv2.stylization(frame, sigma_s=150, sigma_r=0.25)
    emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    emboss = cv2.filter2D(styled, -1, emboss_kernel)
    result = cv2.addWeighted(styled, 0.85, emboss, 0.15, 128)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
    hsv[:, :, 0] = (hsv[:, :, 0] + 5) % 180
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return _vignette(result, 0.2)


def _demo_pop_art(frame: np.ndarray) -> np.ndarray:
    # 1. Smooth first to kill micro-texture
    smooth = cv2.bilateralFilter(frame, 15, 80, 80)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    # 2. Posterize to 4 bold color bands
    posterized = (gray // 64).astype(np.uint8)
    lut = np.array([
        [140, 30, 20],
        [180, 50, 220],
        [60, 230, 255],
        [240, 240, 240],
    ], dtype=np.uint8)
    h, w = gray.shape
    result = lut[posterized.flatten()].reshape(h, w, 3)
    # 3. Edge detect on smoothed posterized (not raw frame)
    edges = cv2.Canny(gray, 60, 140)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    # 4. Composite: bold black outlines
    result[edges > 0] = (0, 0, 0)
    return result


def _demo_sketch(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    tinted = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    color_blend = cv2.addWeighted(tinted, 0.85, frame, 0.15, 0)
    return color_blend


def _demo_vintage(frame: np.ndarray) -> np.ndarray:
    sepia_kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189],
    ], dtype=np.float32)
    sepia = cv2.transform(frame, sepia_kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    result = cv2.convertScaleAbs(sepia, alpha=0.85, beta=30)
    h, w = result.shape[:2]
    np.random.seed(77)
    grain = np.random.normal(0, 12, (h, w, 3)).astype(np.float32)
    result = np.clip(result.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.6, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    leak = np.zeros_like(result, dtype=np.float32)
    leak[:, :, 2] = 60
    leak[:, :, 1] = 30
    y_grad = np.linspace(0.3, 0, h)[:, np.newaxis]
    x_grad = np.linspace(0.3, 0, w)[np.newaxis, :]
    mask = (y_grad * x_grad)[:, :, np.newaxis]
    result = np.clip(result.astype(np.float32) + leak * mask, 0, 255).astype(np.uint8)
    return _vignette(result, 0.5)


def _demo_neon_glow(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    # Smooth first to only detect major edges, not micro-texture
    smooth = cv2.bilateralFilter(frame, 15, 80, 80)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    hue_map = np.zeros((h, w, 3), dtype=np.uint8)
    hue_vals = np.linspace(0, 179, w).astype(np.uint8)
    hue_map[:, :, 0] = hue_vals[np.newaxis, :]
    hue_map[:, :, 1] = 255
    hue_map[:, :, 2] = edges
    colored_edges = cv2.cvtColor(hue_map, cv2.COLOR_HSV2BGR)
    glow = cv2.GaussianBlur(colored_edges, (15, 15), 0)
    neon = cv2.add(colored_edges, glow)
    dark_bg = (frame.astype(np.float32) * 0.25).astype(np.uint8)
    return cv2.add(dark_bg, neon)


# ---------------------------------------------------------------------------
# Legacy OpenCV filters (kept for fallback — neural models preferred)
# ---------------------------------------------------------------------------
LEGACY_FILTERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "monet": _demo_monet,
    "starry_night": _demo_starry_night,
    "cyberpunk": _demo_cyberpunk,
    "ukiyo_e": _demo_ukiyo_e,
    "anime": _demo_anime,
    "watercolor": _demo_watercolor,
    "pixel_art": _demo_pixel_art,
    "oil_painting": _demo_oil_painting,
    "pop_art": _demo_pop_art,
    "sketch": _demo_sketch,
    "vintage": _demo_vintage,
    "neon_glow": _demo_neon_glow,
}


# ---------------------------------------------------------------------------
# Preprocessing for ONNX models
# ---------------------------------------------------------------------------

def preprocess_frame(
    frame: np.ndarray,
    size: int = 512,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Returns (tensor, original_hw) so the caller can resize the output back."""
    orig_h, orig_w = frame.shape[:2]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    scale = size / min(orig_h, orig_w)
    new_w = (int(orig_w * scale) // 8) * 8
    new_h = (int(orig_h * scale) // 8) * 8
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    return np.expand_dims(tensor, axis=0), (orig_h, orig_w)


def postprocess_tensor(
    tensor: np.ndarray | torch.Tensor,
    orig_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Convert TransformNet output (Tanh, range [-1,1]) to uint8 HWC numpy array."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0.0, 1.0)
    tensor = tensor.permute(1, 2, 0)
    result = (tensor.numpy() * 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    if orig_hw is not None:
        orig_h, orig_w = orig_hw
        result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
    return result


def warp_frame(prev_frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp prev_frame forward using dense optical flow field."""
    h, w = flow.shape[:2]
    map_x = (np.arange(w)[None, :] + flow[..., 0]).astype(np.float32)
    map_y = (np.arange(h)[:, None] + flow[..., 1]).astype(np.float32)
    return cv2.remap(prev_frame, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def compute_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Compute dense optical flow with Farneback (OpenCV built-in)."""
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )


def _reencode_with_ffmpeg(raw_path: str, final_path: str) -> bool:
    if not HAS_FFMPEG:
        return False
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", raw_path,
                "-c:v", "libx264", "-crf", "23",
                "-preset", "medium", "-movflags", "+faststart",
                "-an", final_path,
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class StyleTransferEngine:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Neural .pth models (primary — real style transfer)
        self.neural_models: dict[str, TransformerNet] = {}
        for name, rel_path in NEURAL_MODELS.items():
            full_path = BASE_DIR / rel_path
            if not full_path.exists():
                print(f"[skip] {name}: {full_path} not found")
                continue
            try:
                model = TransformerNet()
                sd = torch.load(str(full_path), map_location=self.device, weights_only=True)
                model.load_state_dict(sd)
                model.to(self.device).eval()
                self.neural_models[name] = model
                print(f"[neural] {name} loaded on {self.device}")
            except Exception as e:
                print(f"[error] {name}: {e}")

        # 2. ONNX models (secondary)
        self.sessions: dict[str, ort.InferenceSession] = {}
        for name, rel_path in STYLE_MODELS.items():
            full_path = BASE_DIR / rel_path
            if not full_path.exists():
                continue
            try:
                session = ort.InferenceSession(
                    str(full_path), providers=["CPUExecutionProvider"]
                )
                self.sessions[name] = session
                print(f"[onnx] {name} loaded")
            except Exception as e:
                print(f"[error] {name}: {e}")

        neural = list(self.neural_models.keys())
        onnx = list(self.sessions.keys())
        legacy = list(LEGACY_FILTERS.keys())
        print(f"[init] neural: {neural} | onnx: {onnx or 'none'} | legacy: {legacy}")

    def available_styles(self) -> list[str]:
        return sorted(
            set(self.neural_models.keys())
            | set(self.sessions.keys())
            | set(LEGACY_FILTERS.keys())
        )

    def _neural_stylize(self, frame: np.ndarray, model: TransformerNet) -> np.ndarray:
        """Run a frame through a pretrained TransformerNet (.pth)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)
        with torch.no_grad():
            output = model(tensor)
        output = output.squeeze(0).cpu().clamp(0, 255).byte()
        result = output.permute(1, 2, 0).numpy()
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    def apply_style(self, frame: np.ndarray, style: str) -> np.ndarray:
        # Priority 1: neural .pth models
        neural_model = self.neural_models.get(style)
        if neural_model is not None:
            return self._neural_stylize(frame, neural_model)

        # Priority 2: ONNX models
        session = self.sessions.get(style)
        if session is not None:
            input_tensor, orig_hw = preprocess_frame(frame)
            input_name = session.get_inputs()[0].name
            result = session.run(None, {input_name: input_tensor})
            return postprocess_tensor(result[0], orig_hw=orig_hw)

        # Priority 3: legacy OpenCV filters
        legacy_fn = LEGACY_FILTERS.get(style)
        if legacy_fn is not None:
            return legacy_fn(frame)

        available = self.available_styles()
        raise ValueError(f"Style '{style}' not available. Available: {available}")

    def apply_style_video(
        self,
        input_path: str,
        style: str,
        output_path: str,
        progress_cb: Callable[[int], None] | None = None,
        temporal_weight: float = 0.5,
    ) -> str:
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_w, target_h = _compute_target_size(orig_h, orig_w)

        raw_path = output_path + ".raw.mp4"
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_path, fourcc, fps, (target_w, target_h))

        prev_styled: np.ndarray | None = None
        prev_gray: np.ndarray | None = None
        frame_idx = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            styled = self.apply_style(frame_bgr, style).astype(np.float32)
            styled = cv2.resize(styled, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)

            if prev_styled is not None and prev_gray is not None:
                prev_gray_resized = cv2.resize(prev_gray, (target_w, target_h))
                curr_gray_resized = cv2.resize(curr_gray, (target_w, target_h))
                flow = compute_flow(prev_gray_resized, curr_gray_resized)
                warped = warp_frame(prev_styled, flow).astype(np.float32)

                flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                reliable = (flow_mag < 5).astype(np.float32)[..., None]

                blended = (
                    reliable * ((1.0 - temporal_weight) * styled + temporal_weight * warped)
                    + (1.0 - reliable) * styled
                )
                styled = blended.clip(0, 255).astype(np.uint8)
            else:
                styled = styled.clip(0, 255).astype(np.uint8)

            writer.write(styled)
            prev_styled = styled
            prev_gray = curr_gray
            frame_idx += 1

            if progress_cb and frame_idx % 10 == 0 and total_frames > 0:
                pct = int((frame_idx / total_frames) * 100)
                progress_cb(min(pct, 95))

        cap.release()
        writer.release()

        if _reencode_with_ffmpeg(raw_path, output_path):
            Path(raw_path).unlink(missing_ok=True)
        else:
            Path(raw_path).rename(output_path)

        if progress_cb:
            progress_cb(100)

        return output_path
