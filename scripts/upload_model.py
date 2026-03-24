"""
Upload a single .onnx model weight to HF Hub.
Usage: python scripts/upload_model.py --file models/weights/anime.onnx \
           --repo Kostia06/repaintvideo-models
"""
import argparse

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to .onnx file")
    parser.add_argument("--repo", required=True, help="HF Hub repo id")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=args.file,
        path_in_repo=args.file.split("/")[-1],
        repo_id=args.repo,
        repo_type="model",
    )
    print(f"Uploaded {args.file} -> {args.repo}")
    print(f"Set MODEL_REPO={args.repo} in your HF Space environment variables")


if __name__ == "__main__":
    main()
