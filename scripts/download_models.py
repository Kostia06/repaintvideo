"""
Downloads ONNX style model weights from Hugging Face Hub at container startup.
Models are stored in models/weights/ which is gitignored.
Replace HF_REPO with the actual model repo once weights are uploaded.
"""
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_REPO = os.getenv("MODEL_REPO", "Kostia06/repaintvideo-models")
WEIGHTS_DIR = Path(__file__).parent.parent / "models" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["monet.onnx", "starry_night.onnx", "cyberpunk.onnx", "ukiyo_e.onnx", "anime.onnx"]


def download_models() -> None:
    for model_name in MODELS:
        dest = WEIGHTS_DIR / model_name
        if dest.exists():
            print(f"[skip] {model_name} already present")
            continue
        try:
            print(f"[download] {model_name} from {HF_REPO}")
            path = hf_hub_download(
                repo_id=HF_REPO,
                filename=model_name,
                local_dir=str(WEIGHTS_DIR),
            )
            print(f"[ok] saved to {path}")
        except Exception as e:
            print(f"[warn] could not download {model_name}: {e}")
            print("       App will start without this style — upload weights to HF Hub")


if __name__ == "__main__":
    download_models()
