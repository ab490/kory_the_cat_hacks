# Otsu binarize + horizontal lines + vertical chars -> 28x28 PIL crops for OCRNet.

import numpy as np
from PIL import Image


def binarize(grayscale_uint8_array):
    # Otsu: maximize between-class variance between ink/paper (histogram on 256 bins)
    intensity_histogram = np.bincount(grayscale_uint8_array.flatten(), minlength=256)
    total_pixel_count = grayscale_uint8_array.size
    weighted_intensity_sum = 0
    for intensity_level in range(256):
        weighted_intensity_sum += intensity_level * intensity_histogram[intensity_level]

    best_intensity_threshold = 0
    best_between_class_variance = 0.0
    class0_pixel_count = 0
    class0_weighted_sum = 0
    for intensity_threshold in range(256):
        class0_pixel_count += intensity_histogram[intensity_threshold]
        class1_pixel_count = total_pixel_count - class0_pixel_count
        if class0_pixel_count == 0 or class1_pixel_count == 0:
            continue
        class0_weighted_sum += intensity_threshold * intensity_histogram[intensity_threshold]
        mean_intensity_class0 = class0_weighted_sum / class0_pixel_count
        mean_intensity_class1 = (
            weighted_intensity_sum - class0_weighted_sum
        ) / class1_pixel_count
        between_class_variance = (
            class0_pixel_count
            * class1_pixel_count
            * (mean_intensity_class0 - mean_intensity_class1) ** 2
        )
        if between_class_variance > best_between_class_variance:
            best_between_class_variance = between_class_variance
            best_intensity_threshold = intensity_threshold

    # ink dark -> 255 (white strokes), paper -> 0, same convention as training tensors
    binary_mask = np.where(
        grayscale_uint8_array <= best_intensity_threshold, 255, 0
    ).astype(np.uint8)
    return binary_mask


def find_row_ranges(binary_mask, minimum_row_height=8):
    row_ink_sums = []
    for row_index in range(binary_mask.shape[0]):
        row_ink_sums.append(int(np.sum(binary_mask[row_index, :])))

    row_index_ranges = []
    inside_text_band = False
    text_band_start_row = 0
    for row_index in range(len(row_ink_sums)):
        if not inside_text_band and row_ink_sums[row_index] > 0:
            inside_text_band = True
            text_band_start_row = row_index
        elif inside_text_band and row_ink_sums[row_index] == 0:
            inside_text_band = False
            if (row_index - text_band_start_row) >= minimum_row_height:
                row_index_ranges.append((text_band_start_row, row_index))

    if inside_text_band and (binary_mask.shape[0] - text_band_start_row) >= minimum_row_height:
        row_index_ranges.append((text_band_start_row, binary_mask.shape[0]))

    return row_index_ranges


def find_char_ranges(text_row_binary_strip, minimum_character_width=3):
    column_ink_sums = []
    for column_index in range(text_row_binary_strip.shape[1]):
        column_ink_sums.append(int(np.sum(text_row_binary_strip[:, column_index])))

    column_index_ranges = []
    inside_character = False
    character_start_column = 0
    for column_index in range(len(column_ink_sums)):
        if not inside_character and column_ink_sums[column_index] > 0:
            inside_character = True
            character_start_column = column_index
        elif inside_character and column_ink_sums[column_index] == 0:
            inside_character = False
            if (column_index - character_start_column) >= minimum_character_width:
                column_index_ranges.append((character_start_column, column_index))

    if inside_character and (
        text_row_binary_strip.shape[1] - character_start_column
    ) >= minimum_character_width:
        column_index_ranges.append(
            (character_start_column, text_row_binary_strip.shape[1])
        )

    return column_index_ranges


def segment_characters(pil_image):
    grayscale_uint8_array = np.array(pil_image.convert("L"))
    binary_mask = binarize(grayscale_uint8_array)
    text_row_ranges = find_row_ranges(binary_mask)

    character_pil_images = []
    for row_start_index, row_end_index in text_row_ranges:
        text_row_binary_strip = binary_mask[row_start_index:row_end_index, :]
        character_column_ranges = find_char_ranges(text_row_binary_strip)

        for column_start_index, column_end_index in character_column_ranges:
            character_crop = text_row_binary_strip[
                :, column_start_index:column_end_index
            ]
            border_padding_pixels = 4
            padded_uint8 = np.zeros(
                (
                    character_crop.shape[0] + 2 * border_padding_pixels,
                    character_crop.shape[1] + 2 * border_padding_pixels,
                ),
                dtype=np.uint8,
            )
            padded_uint8[
                border_padding_pixels : border_padding_pixels + character_crop.shape[0],
                border_padding_pixels : border_padding_pixels + character_crop.shape[1],
            ] = character_crop
            character_pil = Image.fromarray(padded_uint8)
            character_pil = character_pil.resize((28, 28), Image.LANCZOS)
            character_pil_images.append(character_pil)

    return character_pil_images
