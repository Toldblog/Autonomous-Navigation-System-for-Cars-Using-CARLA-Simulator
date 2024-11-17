import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from PIL import Image
from .constants import *

class ObstacleDetectionDataset(Dataset):
    def __init__(self, matched_files):
        self.matched_files = matched_files
        self.transform = transforms.Compose([
            transforms.Resize((HEIGHT, WIDTH)),  # Example resizing, adjust based on your requirements
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.matched_files)

    def __getitem__(self, idx):
        pgm_path, seg_path, obstacle = self.matched_files[idx]

        # Load and transform PGM image
        pgm_image = Image.open(pgm_path).convert('L')  # Assuming PGM images are grayscale
        pgm_image = self.transform(pgm_image)

        # Load and transform segmentation image
        seg_image = Image.open(seg_path).convert('RGB')
        seg_image = self.transform(seg_image)

        # Label for obstacle detection (0: no obstacle, 1: obstacle)
        label = torch.tensor(obstacle, dtype=torch.float32)

        return pgm_image, seg_image, label