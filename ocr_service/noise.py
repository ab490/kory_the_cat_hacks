"""
Three noise functions for OCR evaluation and benchmarking.

gaussian noise - additive Gaussian noise N(0, std_2).
salt_and_pepper - random pixels forced to pure black or white. 
sidd noise - camera-realistic noise
"""

import numpy as np
import torch


def gaussian_noise(image, std = 0.15):
    return torch.clamp(image + torch.randn_like(image) * std, 0.0, 1.0)


def salt_and_pepper_noise(image, prob = 0.05):
    mask = torch.rand_like(image)
    noisy = image.clone()
    noisy[mask < prob / 2] = 0.0
    noisy[mask > 1 - prob / 2] = 1.0
    return noisy


def sidd_noise(image, sidd_params):
    sigma = sidd_params.get("sigma", 0.1)
    beta = sidd_params.get("beta", 0.02)
    # Shot noise (signal-dependent) + read noise (signal-independent)
    shot = torch.randn_like(image) * torch.sqrt(image.clamp(min=0) * beta)
    read = torch.randn_like(image) * sigma
    return torch.clamp(image + shot + read, 0.0, 1.0)


NOISE_PROFILES = {
    "gaussian": gaussian_noise,
    "salt_and_pepper": salt_and_pepper_noise,
    "sidd": sidd_noise,
    "clean": lambda x, **kw: x,
}