import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import *

class AutoNaviModel(nn.Module):
    def __init__(self):
        super(AutoNaviModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (HEIGHT // 8) * (WIDTH // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),
            nn.ReLU(),
            nn.Flatten()
        )
        self.output_layer = nn.Linear(5, 1)  # 4 from image + 1 from input_2

    def forward(self, image, navigational_direction):
        image_features = self.conv_layers(image)
        combined = torch.cat((image_features, navigational_direction.unsqueeze(1)), dim=1)
        output = self.output_layer(combined)
        return output