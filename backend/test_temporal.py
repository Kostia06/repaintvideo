"""
Measures temporal consistency and checkerboard score on a video file.
Run from backend/:  python test_temporal.py --video path/to/output.mp4
Pass threshold: mean_diff < 25, checkerboard < 8, high_freq_ratio < 0.65
"""
import argparse
import sys

import cv2
import numpy as np


def checkerboard_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
    h, w = gray.shape
    even_h = (h // 2) * 2
    even_w = (w // 2) * 2
    gray = gray[:even_h, :even_w]
    return float(np.abs(gray[::2, ::2] - gray[1::2, 1::2]).mean())


def high_freq_ratio(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(fft))
    h, w = mag.shape
    cx, cy = w // 2, h // 2
    mask = np.zeros((h, w))
    mask[cy - 30:cy + 30, cx - 30:cx + 30] = 1
    lo = (mag * mask).sum()
    hi = (mag * (1 - mask)).sum()
    return float(hi / (lo + hi))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    frames: list[np.ndarray] = []
    diffs: list[float] = []
    cb_scores: list[float] = []
    hf_ratios: list[float] = []

    while len(frames) < 30:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
        cb_scores.append(checkerboard_score(f))
        hf_ratios.append(high_freq_ratio(f))
        if len(frames) > 1:
            diff = np.abs(frames[-1].astype(float) - frames[-2].astype(float))
            diffs.append(float(diff.mean()))

    cap.release()

    if not diffs:
        print("ERROR: Could not read enough frames")
        sys.exit(1)

    mean_diff = float(np.mean(diffs))
    mean_cb = float(np.mean(cb_scores))
    mean_hf = float(np.mean(hf_ratios))

    print(f"\nTemporal diff   : {mean_diff:.2f}  (target <25)  "
          f"{'PASS' if mean_diff < 25 else 'FAIL'}")
    print(f"Checkerboard    : {mean_cb:.2f}   (target <8)   "
          f"{'PASS' if mean_cb < 8 else 'FAIL'}")
    print(f"High-freq ratio : {mean_hf:.3f}  (target <0.65) "
          f"{'PASS' if mean_hf < 0.65 else 'FAIL'}")

    passed = mean_diff < 25 and mean_cb < 8 and mean_hf < 0.65
    print(f"\n{'ALL PASS' if passed else 'FAILURES DETECTED — re-check fixes above'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
