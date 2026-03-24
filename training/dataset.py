from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CocoDataset(Dataset):
    def __init__(self, root: str, size: int = 256) -> None:
        self.root = Path(root).expanduser()
        self.files = sorted(
            list(self.root.glob("*.jpg")) + list(self.root.glob("*.png"))
        )
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)
