from dataclasses import dataclass


@dataclass
class TrainConfig:
    style_image: str = "assets/monet.jpg"
    dataset_dir: str = "~/coco/train2017"
    checkpoint_dir: str = "checkpoints/"
    output_onnx: str = "models/weights/monet.onnx"
    image_size: int = 256
    batch_size: int = 4
    lr: float = 1e-3
    epochs: int = 2
    content_weight: float = 1.0
    style_weight: float = 1e5
    tv_weight: float = 1e-6
    log_every: int = 100
    save_every: int = 1000
