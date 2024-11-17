import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import *

class ObstacleDetectorModel(nn.Module):
    def __init__(self):
        super(ObstacleDetectorModel, self).__init__()

        # Convolutional layers for PGM input
        self.pgm_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Assuming PGM is grayscale (1 channel)
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        # Convolutional layers for segmentation input
        self.seg_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Assuming segmentation is RGB (3 channels)
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        # Fully connected layers after concatenation
        self.fc = nn.Sequential(
            nn.Linear(128 * ((HEIGHT // 8) * (WIDTH // 8)) * 2, 128),  # Adjust based on the final size of the flattened layers
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output a probability for obstacle presence
        )

    def forward(self, pgm_input, seg_input):
        pgm_features = self.pgm_branch(pgm_input)
        seg_features = self.seg_branch(seg_input)

        # Concatenate the features from both branches
        combined_features = torch.cat((pgm_features, seg_features), dim=1)

        # Pass through fully connected layers
        output = self.fc(combined_features)
        return output
