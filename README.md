# RepaintVideo

Neural style transfer for video, with temporal consistency.

Transform any video into a living painting — choose from Monet, Starry Night, Cyberpunk, or Ukiyo-e styles. Powered by a feed-forward network with VGG-19 perceptual loss and optical flow warping for flicker-free results.

**Demo:** https://huggingface.co/spaces/YOUR_USERNAME/repaintvideo

## Architecture

- **Feed-forward TransformNet** (Johnson et al.) trained per style, exported to ONNX for fast CPU inference
- **VGG-19 perceptual loss** at 5 relu layers for content/style balance, plus temporal warp loss for video consistency
- **FastAPI + Next.js** served from a single Docker container on Hugging Face Spaces

## Local development

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --port 7860

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Set `NEXT_PUBLIC_API_URL=http://localhost:7860` in `frontend/.env` for local dev.

## Training

Requires a GPU and MS-COCO train2017 images.

```bash
cd training
python train.py --style monet
```

This trains a TransformNet for the chosen style and exports an ONNX model to `models/weights/`. Upload the `.onnx` file to your HF Hub model repo.

## Deployment

Push to `main` and the GitHub Action auto-deploys to Hugging Face Spaces.

```bash
git push origin main
```

## License

MIT
