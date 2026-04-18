import numpy as np
import torch


def gaussian_noise(image_tensor, noise_standard_deviation=0.15):
    return torch.clamp(
        image_tensor +
        torch.randn_like(image_tensor) * noise_standard_deviation,
        0.0,
        1.0,
    )


def salt_and_pepper_noise(image_tensor, impulse_probability=0.05):
    uniform_mask = torch.rand_like(image_tensor)
    noisy_tensor = image_tensor.clone()
    noisy_tensor[uniform_mask < impulse_probability / 2] = 0.0
    noisy_tensor[uniform_mask > 1 - impulse_probability / 2] = 1.0
    return noisy_tensor


def sidd_noise(image_tensor, sidd_sensor_noise_parameters):
    """SIDD-style toy mix: shot (signal-dependent) + read (signal-independent)."""
    sigma_read_noise = sidd_sensor_noise_parameters["sigma"]
    beta_shot_noise_scale = sidd_sensor_noise_parameters["beta"]
    shot_noise = torch.randn_like(image_tensor) * torch.sqrt(
        image_tensor.clamp(min=0) * beta_shot_noise_scale
    )
    read_noise = torch.randn_like(image_tensor) * sigma_read_noise
    return torch.clamp(image_tensor + shot_noise + read_noise, 0.0, 1.0)


NOISE_PROFILES = {
    "gaussian": gaussian_noise,
    "salt_and_pepper": salt_and_pepper_noise,
    "sidd": sidd_noise,
    "clean": lambda image_tensor, **unused_keyword_arguments: image_tensor,
}
