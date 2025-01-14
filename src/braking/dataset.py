from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.braking import DATA_ROOT


class KyushuDataset(Dataset):
    def __init__(self, csv_id, video_name, device):
        self.device = device
        self.path_to_imgs = sorted(list(Path(DATA_ROOT / "images" / csv_id / video_name).glob("*.png")))

    def __len__(self):
        return len(self.path_to_imgs)

    def __getitem__(self, idx):
        path_to_img = self.path_to_imgs[idx]
        img = torch.tensor(np.array(Image.open(path_to_img)).transpose(2, 0, 1), device=self.device)
        return img
