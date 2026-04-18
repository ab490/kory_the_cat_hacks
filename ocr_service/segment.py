"""
Connected-component character segmentation.

Steps:
  1. Binarize image with Otsu's threshold
  2. Label connected ink blobs
  3. Filter blobs by size
  4. Sort blobs reading-order (top-to-bottom, left-to-right by line)
  5. Return list of individual character crops as PIL Images
"""

import numpy as np
from PIL import Image
from scipy import ndimage


def segment_characters(image: Image.Image,
                       min_area: int = None,
                       min_width: int = None,
                       max_area: int = None) -> list:
    gray = np.array(image.convert("L"), dtype=np.float32)
    h, w = gray.shape

    total_pixels = h * w
    if min_area is None:
        min_area = max(20, int(total_pixels * 0.00015))
    if min_width is None:
        min_width = max(3, int(w * 0.003))
    if max_area is None:
        max_area = int(total_pixels * 0.02)

    threshold = _otsu_threshold(gray)
    if gray.mean() > 127:
        binary = (gray < threshold).astype(np.uint8)
    else:
        binary = (gray > threshold).astype(np.uint8)

    labeled, num_features = ndimage.label(binary)
    if num_features == 0:
        return []

    objects = ndimage.find_objects(labeled)

    boxes = []
    for slc in objects:
        if slc is None:
            continue
        row_slice, col_slice = slc
        bh = row_slice.stop - row_slice.start
        bw = col_slice.stop - col_slice.start
        area = int(binary[slc].sum())

        if area < min_area or bw < min_width:
            continue
        if area > max_area:
            continue
        if bw > bh * 8:
            continue

        boxes.append((row_slice.start, col_slice.start,
                      row_slice.stop, col_slice.stop))

    if not boxes:
        return []

    boxes = _sort_reading_order(boxes, img_height=h)

    crops = []
    for top, left, bottom, right in boxes:
        crop = gray[top:bottom, left:right]
        crops.append(Image.fromarray(crop.astype(np.uint8)))

    return crops


def _sort_reading_order(boxes: list, img_height: int = None) -> list:
    """Sort boxes into reading order: lines top-to-bottom, chars left-to-right."""
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[0])

    heights = [b[2] - b[0] for b in boxes]
    median_h = sorted(heights)[len(heights) // 2]
    line_tolerance = median_h * 0.6

    lines = []
    for box in boxes:
        top, left, bottom, right = box
        mid_y = (top + bottom) / 2
        placed = False
        for line in lines:
            line_mid = sum((b[0] + b[2]) / 2 for b in line) / len(line)
            if abs(mid_y - line_mid) < line_tolerance:
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])

    result = []
    for line in lines:
        result.extend(sorted(line, key=lambda b: b[1]))

    return result


def _otsu_threshold(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    count_b = 0
    max_var = 0.0
    threshold = 128.0
    for t in range(256):
        count_b += hist[t]
        if count_b == 0:
            continue
        count_f = total - count_b
        if count_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / count_b
        mean_f = (sum_total - sum_b) / count_f
        var = count_b * count_f * (mean_b - mean_f) ** 2
        if var > max_var:
            max_var = var
            threshold = t
    return threshold
