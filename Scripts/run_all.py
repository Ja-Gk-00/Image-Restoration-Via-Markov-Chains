#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import pandas as pd

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Utils.utils import generate_and_save_denoised_images


def main(clean_images_dir, noisy_images_dir, output_dir):
    clean_images_path = Path(clean_images_dir)
    noisy_images_path = Path(noisy_images_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for clean_image_path in clean_images_path.glob("*.png"):
        print(f"Processing {clean_image_path}...")
        noisy_image_path = list(noisy_images_path.glob(f"{clean_image_path.stem}*"))[0]
        if not noisy_image_path.exists():
            print(f"Noisy image for {clean_image_path} not found, skipping.")
            continue
        denoised_image_paths = generate_and_save_denoised_images(
            str(clean_image_path), str(output_path)
        )

        clean_image = cv2.imread(str(clean_image_path), cv2.IMREAD_COLOR)
        clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
        row = {}
        for name, path in (denoised_image_paths | {"noisy": noisy_image_path}).items():
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            l1 = np.mean(np.abs(clean_image - image))
            l2 = np.mean((clean_image - image) ** 2)
            psnr = cv2.PSNR(clean_image, image)
            ssim = cv2.SSIM(clean_image, image, multichannel=True)
            row[f"{name}_l1"] = l1
            row[f"{name}_l2"] = l2
            row[f"{name}_psnr"] = psnr
            row[f"{name}_ssim"] = ssim
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path / "denoising_results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise images and compute metrics")
    parser.add_argument(
        "clean_images_dir", help="Directory containing clean PNG images"
    )
    parser.add_argument("noisy_images_dir", help="Directory containing noisy images")
    parser.add_argument(
        "output_dir", help="Directory to save denoised images and results"
    )
    args = parser.parse_args()

    main(args.clean_images_dir, args.noisy_images_dir, args.output_dir)
