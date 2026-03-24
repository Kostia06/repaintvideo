#!/usr/bin/env bash
# Train the anime style model and export to ONNX in one command.
# Prerequisites: pip install -r backend/requirements.txt
#                MS-COCO 2017 train images at ~/coco/train2017/
#                An anime style reference image at training/styles/anime.jpg
#
# Usage: bash training/train_anime.sh

set -e

STYLE_IMG="training/styles/anime.jpg"
OUTPUT="models/weights/anime.onnx"
CHECKPOINT_DIR="checkpoints/anime"

if [ ! -f "$STYLE_IMG" ]; then
  echo "ERROR: $STYLE_IMG not found."
  echo "Provide a high-quality anime frame (1920x1080+ recommended) as the style image."
  exit 1
fi

mkdir -p "$CHECKPOINT_DIR" "models/weights"

echo "Training anime style model..."
python training/train.py \
  --style anime

echo "Done. Model saved to $OUTPUT"
echo "Upload to HF Hub with:"
echo "  python scripts/upload_model.py --file $OUTPUT --repo Kostia06/repaintvideo-models"
