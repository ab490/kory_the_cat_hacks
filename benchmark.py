# OCR benchmark: SimulatedNoisyOffice TE clean vs noisy. Needs OCR server on :5001.

import os
import requests

DATASET = "SimulatedNoisyOffice"
CLEAN_DIR = os.path.join(DATASET, "clean_images_grayscale")
NOISY_DIR = os.path.join(DATASET, "simulated_noisy_images_grayscale")
OCR_URL = "http://localhost:5001/ocr"

FONTS = [
    "FontLre", "FontLrm", "FontLse", "FontLsm", "FontLte", "FontLtm",
    "Fontfre", "Fontfrm", "Fontfse", "Fontfsm", "Fontfte", "Fontftm",
    "Fontnre", "Fontnrm", "Fontnse", "Fontnsm", "Fontnte", "Fontntm",
]

NOISE_TYPES = ["Noisec", "Noisef", "Noisep", "Noisew"]
NOISE_NAMES = {
    "Noisec": "coffee stains",
    "Noisef": "folded sheets",
    "Noisep": "footprints",
    "Noisew": "wrinkled sheets",
}


def post_image_to_ocr_service_and_get_text(image_path):
    with open(image_path, "rb") as image_file:
        http_response = requests.post(OCR_URL, files={"image": image_file})
    http_response.raise_for_status()
    return http_response.json()["text"]


def character_wise_match_rate(reference_text, predicted_text):
    if len(reference_text) == 0:
        return 0.0
    match_count = 0
    comparison_length = min(len(reference_text), len(predicted_text))
    for character_index in range(comparison_length):
        if reference_text[character_index] == predicted_text[character_index]:
            match_count += 1
    return match_count / len(reference_text)


def print_summary(title, results):
    print("\n" + "=" * 70)
    print(title)
    print(f"{'Noise type':<20} {'Avg clean chars':>16} {'Avg noisy chars':>16} {'Avg match rate':>15}")
    print("-" * 70)
    for noise in NOISE_TYPES:
        entries = results[noise]
        avg_clean = sum(e[0] for e in entries) / len(entries)
        avg_noisy = sum(e[1] for e in entries) / len(entries)
        avg_match = sum(e[2] for e in entries) / len(entries)
        print(
            f"{NOISE_NAMES[noise]:<20} {avg_clean:>16.1f} {avg_noisy:>16.1f} {avg_match:>14.1%}")
    print("=" * 70)


def main():
    assert os.path.isdir(
        CLEAN_DIR), "need SimulatedNoisyOffice/clean_images_grayscale"
    assert os.path.isdir(
        NOISY_DIR), "need SimulatedNoisyOffice/simulated_noisy_images_grayscale"

    results = {n: [] for n in NOISE_TYPES}

    for font in FONTS:
        clean_path = os.path.join(CLEAN_DIR, font + "_Clean_TE.png")
        clean_text = post_image_to_ocr_service_and_get_text(clean_path)
        print(font + " clean (" + str(len(clean_text)) + " chars)")

        for noise in NOISE_TYPES:
            noisy_path = os.path.join(
                NOISY_DIR, font + "_" + noise + "_TE.png")
            noisy_text = post_image_to_ocr_service_and_get_text(noisy_path)
            rate = character_wise_match_rate(clean_text, noisy_text)
            results[noise].append((len(clean_text), len(noisy_text), rate))
            print("  " + noise + " match=" + str(round(rate * 100, 2)) + "%")

    print_summary("SIMULATED TE (avg over fonts)", results)


if __name__ == "__main__":
    main()
