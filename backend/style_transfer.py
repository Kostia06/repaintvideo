import shutil
import subprocess
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import onnxruntime as ort

STYLE_MODELS: dict[str, str] = {
    "monet": "models/weights/monet.onnx",
    "starry_night": "models/weights/starry_night.onnx",
    "cyberpunk": "models/weights/cyberpunk.onnx",
    "ukiyo_e": "models/weights/ukiyo_e.onnx",
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
    h, w = frame.shape[:2]
    small_size = min(256, min(h, w))
    scale = small_size / max(h, w)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 8, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    quantized_small = centers[labels.flatten()].reshape(small.shape)
    quantized = cv2.resize(quantized_small, (w, h), interpolation=cv2.INTER_NEAREST)
    smooth = cv2.bilateralFilter(quantized, 9, 150, 150)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)
    edge_mask = (edges.astype(np.float32) / 255.0)[:, :, np.newaxis]
    result = (smooth.astype(np.float32) * (edge_mask * 0.7 + 0.3)).astype(np.uint8)
    warm_paper = np.full_like(result, (195, 215, 230), dtype=np.uint8)
    result = cv2.addWeighted(result, 0.95, warm_paper, 0.05, 0)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.85, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _demo_anime(frame: np.ndarray) -> np.ndarray:
    color = cv2.bilateralFilter(frame, 9, 200, 200)
    color = cv2.bilateralFilter(color, 9, 200, 200)
    posterized = (color // 21 * 21 + 10).astype(np.uint8)
    hsv = cv2.cvtColor(posterized, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
    posterized = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 3)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(posterized, edges_bgr)


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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    posterized = (gray // 64).astype(np.uint8)
    lut = np.array([
        [140, 30, 20],
        [180, 50, 220],
        [60, 230, 255],
        [240, 240, 240],
    ], dtype=np.uint8)
    h, w = gray.shape
    result = lut[posterized.flatten()].reshape(h, w, 3)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray_blur, 80, 160)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edge_mask = (edges == 0).astype(np.uint8)[:, :, np.newaxis]
    return (result * edge_mask).astype(np.uint8)


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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 80)
    kernel = np.ones((2, 2), np.uint8)
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


DEMO_FILTERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
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

def preprocess_frame(frame: np.ndarray, size: int = 512) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    return np.expand_dims(tensor, axis=0)


def postprocess_tensor(tensor: np.ndarray) -> np.ndarray:
    output = tensor.squeeze(0).transpose(1, 2, 0)
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


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
        self.sessions: dict[str, ort.InferenceSession] = {}
        for name, rel_path in STYLE_MODELS.items():
            full_path = BASE_DIR / rel_path
            if not full_path.exists():
                print(f"[skip] {name}: {full_path} not found")
                continue
            try:
                session = ort.InferenceSession(
                    str(full_path), providers=["CPUExecutionProvider"]
                )
                self.sessions[name] = session
                print(f"[loaded] {name} from {full_path}")
            except Exception as e:
                print(f"[error] {name}: {e}")

        loaded = list(self.sessions.keys())
        demo = list(DEMO_FILTERS.keys())
        print(f"[init] ONNX models: {loaded or 'none'} | demo filters: {demo}")

    def available_styles(self) -> list[str]:
        return sorted(set(self.sessions.keys()) | set(DEMO_FILTERS.keys()))

    def apply_style(self, frame: np.ndarray, style: str) -> np.ndarray:
        session = self.sessions.get(style)
        if session is not None:
            input_tensor = preprocess_frame(frame)
            input_name = session.get_inputs()[0].name
            result = session.run(None, {input_name: input_tensor})
            return postprocess_tensor(result[0])

        demo_fn = DEMO_FILTERS.get(style)
        if demo_fn is not None:
            return demo_fn(frame)

        return frame

    def apply_style_video(
        self,
        input_path: str,
        style: str,
        output_path: str,
        progress_cb: Callable[[int], None] | None = None,
    ) -> str:
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_w, target_h = _compute_target_size(orig_h, orig_w)

        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError(f"Cannot read video: {input_path}")

        styled_first = self.apply_style(first_frame, style)
        styled_first = cv2.resize(styled_first, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        raw_path = output_path + ".raw.mp4"
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_path, fourcc, fps, (target_w, target_h))
        writer.write(styled_first)

        frame_idx = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            styled = self.apply_style(frame, style)
            styled = cv2.resize(styled, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            writer.write(styled)
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
