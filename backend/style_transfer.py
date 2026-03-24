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


def _demo_monet(frame: np.ndarray) -> np.ndarray:
    smooth = cv2.bilateralFilter(frame, 9, 75, 75)
    smooth = cv2.bilateralFilter(smooth, 9, 75, 75)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _demo_starry_night(frame: np.ndarray) -> np.ndarray:
    smooth = cv2.bilateralFilter(frame, 9, 100, 100)
    edges = cv2.Canny(frame, 50, 150)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + 20) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv2.addWeighted(result, 0.85, edges_bgr, 0.15, 0)


def _demo_cyberpunk(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR).astype(np.float32)
    result[:, :, 0] = np.clip(result[:, :, 0] * 1.3, 0, 255)  # blue boost
    result[:, :, 2] = np.clip(result[:, :, 2] * 0.7, 0, 255)  # red reduce
    result[:, :, 1] = np.clip(result[:, :, 1] * 1.1, 0, 255)  # slight green
    return result.astype(np.uint8)


def _demo_ukiyo_e(frame: np.ndarray) -> np.ndarray:
    quantized = (frame // 32 * 32 + 16).astype(np.uint8)
    smooth = cv2.bilateralFilter(quantized, 9, 75, 75)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(smooth, edges_bgr)


DEMO_FILTERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "monet": _demo_monet,
    "starry_night": _demo_starry_night,
    "cyberpunk": _demo_cyberpunk,
    "ukiyo_e": _demo_ukiyo_e,
}


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
        print(f"[init] StyleTransferEngine ready — models loaded: {loaded or 'none'}")

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

        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError(f"Cannot read video: {input_path}")

        styled_first = self.apply_style(first_frame, style)
        h, w = styled_first.shape[:2]
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        writer.write(styled_first)

        frame_idx = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            styled = self.apply_style(frame, style)
            styled = cv2.resize(styled, (w, h))
            writer.write(styled)
            frame_idx += 1

            if progress_cb and frame_idx % 10 == 0 and total_frames > 0:
                pct = int((frame_idx / total_frames) * 100)
                progress_cb(min(pct, 99))

        cap.release()
        writer.release()

        if progress_cb:
            progress_cb(100)

        return output_path
