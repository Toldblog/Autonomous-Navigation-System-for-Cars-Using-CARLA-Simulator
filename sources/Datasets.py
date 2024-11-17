import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from PIL import Image
import os

from .utils import *

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
    

# Data Preprocessing
class AutoNaviDataset(Dataset):
    def __init__(self, image_files):
        self.image_files = image_files
        self.transform = transforms.Compose([
            transforms.Resize((HEIGHT, WIDTH)),
            transforms.ToTensor(),  # Converts to [0, 1] range
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label = float(os.path.basename(image_path).split('.png')[0].split('_')[2])
        if label > MAX_STEER_DEGREES:
            label = MAX_STEER_DEGREES
        elif label < -MAX_STEER_DEGREES:
            label = -MAX_STEER_DEGREES
        label = label / MAX_STEER_DEGREES  # Normalize label

        input_2 = int(os.path.basename(image_path).split('.png')[0].split('_')[1])

        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, torch.tensor(input_2, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)