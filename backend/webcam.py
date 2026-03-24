"""Live webcam style transfer with keyboard controls."""
import argparse
import glob
import time

import cv2
import numpy as np
import torch

from fast_network import LightweightTransformNet, TransformNet

WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720
CPU_WIDTH = 640
CPU_HEIGHT = 360


def _load_model(path: str, device: str, lightweight: bool) -> torch.nn.Module:
    model = LightweightTransformNet() if lightweight else TransformNet()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def _frame_to_tensor(frame: np.ndarray, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0


def _tensor_to_frame(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.squeeze(0).detach().cpu().clamp(0, 255).byte()
    return cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)


def run_webcam(
    model_paths: list[str],
    device: str = "auto",
    lightweight: bool = True,
) -> None:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model_paths:
        print("[error] No model files found. Train a model first.")
        return

    models = [_load_model(p, device, lightweight) for p in model_paths]
    names = [p.split("/")[-1].replace(".pth", "") for p in model_paths]
    style_idx = 0

    cap = cv2.VideoCapture(0)
    target_w = WEBCAM_WIDTH if device == "cuda" else CPU_WIDTH
    target_h = WEBCAM_HEIGHT if device == "cuda" else CPU_HEIGHT
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)

    recording = False
    writer: cv2.VideoWriter | None = None
    fps_history: list[float] = []

    print(f"[webcam] device={device} styles={names}")
    print("[webcam] keys: s=cycle style, r=record, q=quit")

    with torch.no_grad():
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            proc_h = target_h // 8 * 8
            proc_w = target_w // 8 * 8
            resized = cv2.resize(frame, (proc_w, proc_h))

            tensor = _frame_to_tensor(resized, device)
            styled = models[style_idx](tensor)
            output = _tensor_to_frame(styled)

            dt = time.perf_counter() - t0
            fps_history.append(1.0 / max(dt, 1e-6))
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            label = f"{names[style_idx]} | {avg_fps:.1f} FPS"
            if recording:
                label += " [REC]"
            cv2.putText(output, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Style Transfer", output)

            if recording and writer is not None:
                writer.write(output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                style_idx = (style_idx + 1) % len(models)
                print(f"[style] switched to {names[style_idx]}")
            elif key == ord("r"):
                if recording:
                    recording = False
                    if writer is not None:
                        writer.release()
                        writer = None
                    print("[record] stopped")
                else:
                    recording = True
                    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                    writer = cv2.VideoWriter("webcam_recording.mp4", fourcc, 30, (proc_w, proc_h))
                    print("[record] started -> webcam_recording.mp4")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Single model path")
    parser.add_argument("--model-dir", default="checkpoints", help="Dir to glob *.pth from")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--full", action="store_true", help="Use full TransformNet instead of lightweight")
    args = parser.parse_args()

    if args.model:
        paths = [args.model]
    else:
        paths = sorted(glob.glob(f"{args.model_dir}/*.pth"))

    run_webcam(paths, device=args.device, lightweight=not args.full)
