import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from config import TrainConfig
from dataset import CocoDataset
from fast_network import TransformNet
from losses import content_loss, style_loss, total_variation_loss
from vgg import VGGFeatures

STYLE_CONFIGS: dict[str, dict[str, str]] = {
    "monet": {"style_image": "assets/monet.jpg", "output_onnx": "models/weights/monet.onnx"},
    "starry_night": {"style_image": "assets/starry_night.jpg", "output_onnx": "models/weights/starry_night.onnx"},
    "cyberpunk": {"style_image": "assets/cyberpunk.jpg", "output_onnx": "models/weights/cyberpunk.onnx"},
    "ukiyo_e": {"style_image": "assets/ukiyo_e.jpg", "output_onnx": "models/weights/ukiyo_e.onnx"},
}


def load_style_image(path: str, size: int) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


def train(cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    vgg = VGGFeatures().to(device)
    style_img = load_style_image(cfg.style_image, cfg.image_size).to(device)
    style_features = vgg(style_img)

    model = TransformNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    dataset = CocoDataset(cfg.dataset_dir, cfg.image_size)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(cfg.epochs):
        for batch in loader:
            batch = batch.to(device)
            generated = model(batch) / 255.0

            gen_features = vgg(generated)
            content_features = vgg(batch)

            c_loss = content_loss(gen_features, content_features) * cfg.content_weight
            s_loss = style_loss(gen_features, style_features) * cfg.style_weight
            tv_loss = total_variation_loss(generated) * cfg.tv_weight
            total = c_loss + s_loss + tv_loss

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            step += 1
            if step % cfg.log_every == 0:
                print(
                    f"[step {step}] content={c_loss.item():.4f} "
                    f"style={s_loss.item():.4f} tv={tv_loss.item():.4f} "
                    f"total={total.item():.4f}"
                )

            if step % cfg.save_every == 0:
                ckpt_path = checkpoint_dir / f"checkpoint_{step}.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"[save] {ckpt_path}")

    output_path = Path(cfg.output_onnx)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, cfg.image_size, cfg.image_size).to(device)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"[export] ONNX model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", required=True, choices=list(STYLE_CONFIGS.keys()))
    args = parser.parse_args()

    overrides = STYLE_CONFIGS[args.style]
    cfg = TrainConfig(**overrides)
    print(f"[config] Training style: {args.style}")
    train(cfg)
