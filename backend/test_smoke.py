"""
Smoke test: loads a single test frame, runs it through the full pipeline,
and saves the result. Run from the backend/ directory.
Usage: python test_smoke.py --frame path/to/frame.jpg --style monet
"""
import argparse

import cv2
import numpy as np
import torch

from style_transfer import preprocess_frame, postprocess_tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True)
    parser.add_argument("--style", default="monet")
    args = parser.parse_args()

    frame = cv2.imread(args.frame)
    assert frame is not None, f"Could not read {args.frame}"
    print(f"Input:  shape={frame.shape}  dtype={frame.dtype}  "
          f"mean={frame.mean():.1f}")

    tensor = preprocess_frame(frame)
    tensor = torch.from_numpy(tensor)
    print(f"Tensor: shape={tuple(tensor.shape)}  "
          f"min={tensor.min():.3f}  max={tensor.max():.3f}")

    fake_output = (tensor * 2.0) - 1.0
    result = postprocess_tensor(fake_output)
    print(f"Output: shape={result.shape}  dtype={result.dtype}  "
          f"mean={result.mean():.1f}")

    assert result.dtype == np.uint8, "FAIL: output is not uint8"
    assert result.shape[2] == 3, "FAIL: output is not 3-channel"
    assert 50 < result.mean() < 200, "FAIL: output mean out of expected range"
    assert abs(result.mean() - frame.mean()) < 30, "FAIL: colors drifted too far"

    out_path = "smoke_result.jpg"
    cv2.imwrite(out_path, result)
    print(f"\nPASS — result saved to {out_path}")
    print("Open smoke_result.jpg and confirm it looks like the input frame "
          "(same content, no color inversion, no noise).")


if __name__ == "__main__":
    main()
