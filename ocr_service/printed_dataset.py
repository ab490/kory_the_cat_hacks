# Random printed glyphs -> binarize/pad/resize like segment.py -> 28x28 tensor.
# TrueType fonts from Hugging Face (not system paths): model repo jonathang/fonts-ttf

import random
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from torch.utils.data import Dataset

from segment import binarize
from noise import gaussian_noise, salt_and_pepper_noise

LABEL_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

# Open TTF collection on the Hub (DejaVu Sans Mono, Vogue, etc.); files cached locally after first download.
HUGGINGFACE_FONTS_REPOSITORY_ID = "jonathang/fonts-ttf"
HUGGINGFACE_FONTS_REPOSITORY_TYPE = "model"


def collect_font_file_paths_from_huggingface():
    """Download and return local paths to every .ttf in the Hub repo (HF cache)."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as import_error:
        raise ImportError(
            "Install huggingface_hub (see requirements.txt) to load training fonts from Hugging Face."
        ) from import_error

    truetype_filenames_in_repository = sorted(
        repository_path
        for repository_path in list_repo_files(
            HUGGINGFACE_FONTS_REPOSITORY_ID,
            repo_type=HUGGINGFACE_FONTS_REPOSITORY_TYPE,
        )
        if repository_path.lower().endswith(".ttf")
    )
    assert len(truetype_filenames_in_repository) > 0, (
        "No .ttf files listed in " + HUGGINGFACE_FONTS_REPOSITORY_ID
    )
    return [
        hf_hub_download(
            HUGGINGFACE_FONTS_REPOSITORY_ID,
            filename,
            repo_type=HUGGINGFACE_FONTS_REPOSITORY_TYPE,
        )
        for filename in truetype_filenames_in_repository
    ]


class PrintedCharDataset(Dataset):
    def __init__(self, length=100000, train=True):
        self.length = length
        self.train = train
        self.fonts = collect_font_file_paths_from_huggingface()

    def __len__(self):
        return self.length

    def __getitem__(self, sample_index):
        class_index = random.randint(0, len(LABEL_MAP) - 1)
        character_label = LABEL_MAP[class_index]

        font_path = random.choice(self.fonts)
        font_size_pixels = random.randint(22, 44)
        font = ImageFont.truetype(font_path, font_size_pixels)

        canvas = Image.new("L", (64, 64), 255)
        draw = ImageDraw.Draw(canvas)
        text_bounding_box = draw.textbbox((0, 0), character_label, font=font)
        text_width_pixels = text_bounding_box[2] - text_bounding_box[0]
        text_height_pixels = text_bounding_box[3] - text_bounding_box[1]
        horizontal_jitter_pixels = random.randint(-2, 2)
        vertical_jitter_pixels = random.randint(-2, 2)
        text_origin_x = (64 - text_width_pixels) // 2 - \
            text_bounding_box[0] + horizontal_jitter_pixels
        text_origin_y = (64 - text_height_pixels) // 2 - \
            text_bounding_box[1] + vertical_jitter_pixels
        draw.text((text_origin_x, text_origin_y),
                  character_label, fill=0, font=font)

        binary = binarize(np.array(canvas))
        assert binary.any()

        pil = Image.fromarray(binary)
        if self.train:
            morph_random_value = random.random()
            if morph_random_value < 0.33:
                pil = pil.filter(ImageFilter.MinFilter(3))
            elif morph_random_value > 0.66:
                pil = pil.filter(ImageFilter.MaxFilter(3))

        ink_paper_array = np.array(pil)
        assert ink_paper_array.any()
        nonzero_row_indices, nonzero_column_indices = np.nonzero(
            ink_paper_array)
        crop_row_start, crop_row_end = nonzero_row_indices.min(), nonzero_row_indices.max() + 1
        crop_col_start, crop_col_end = nonzero_column_indices.min(
        ), nonzero_column_indices.max() + 1
        cropped_character_array = ink_paper_array[crop_row_start:crop_row_end,
                                                  crop_col_start:crop_col_end]

        border_padding_pixels = 4
        padded_character_array = np.zeros(
            (
                cropped_character_array.shape[0] + 2 * border_padding_pixels,
                cropped_character_array.shape[1] + 2 * border_padding_pixels,
            ),
            dtype=np.uint8,
        )
        padded_character_array[
            border_padding_pixels: border_padding_pixels + cropped_character_array.shape[0],
            border_padding_pixels: border_padding_pixels + cropped_character_array.shape[1],
        ] = cropped_character_array
        resized_pil = Image.fromarray(
            padded_character_array).resize((28, 28), Image.LANCZOS)

        image_tensor = torch.from_numpy(np.array(resized_pil).astype(
            np.float32) / 255.0).unsqueeze(0)

        if self.train:
            if random.random() < 0.5:
                image_tensor = gaussian_noise(image_tensor)
            else:
                image_tensor = salt_and_pepper_noise(image_tensor)

        return image_tensor, class_index
