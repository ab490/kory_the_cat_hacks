# DnCNN-style denoiser CNN: predict residual noise, subtract, clamp to [0, 1].

import torch
import torch.nn as nn


class Denoiser(nn.Module):
    def __init__(self, num_layers=10, num_features=64):
        super().__init__()
        convolution_stack = [
            nn.Conv2d(1, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            convolution_stack += [
                nn.Conv2d(num_features, num_features,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
            ]
        convolution_stack.append(
            nn.Conv2d(num_features, 1, kernel_size=3, padding=1))
        self.convolutional_network = nn.Sequential(*convolution_stack)

    def forward(self, noisy_image_tensor):
        predicted_noise = self.convolutional_network(noisy_image_tensor)
        return torch.clamp(noisy_image_tensor - predicted_noise, 0.0, 1.0)
