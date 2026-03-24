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
        if session is None:
            return frame

        input_tensor = preprocess_frame(frame)
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: input_tensor})
        return postprocess_tensor(result[0])

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
