"""
DnCNN-style denoiser for document image noise removal.

Residual learning: the network predicts the NOISE in the image,
then subtracts it from the input to get the clean image.
This is easier to train than predicting the clean image directly.

Architecture: Conv -> ReLU -> (Conv -> BN -> ReLU) x N -> Conv
Output: input - predicted_noise (clamped to [0,1])
"""

import torch
import torch.nn as nn

class Denoiser(nn.Module):
    def __init__(self, num_layers = 10, num_features = 64):
        super().__init__()
        layers = [
            nn.Conv2d(1, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(num_features, 1, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.clamp(x - self.net(x), 0.0, 1.0)
