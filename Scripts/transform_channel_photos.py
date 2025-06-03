import os
import argparse
import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import img_as_ubyte


def to_grayscale(img):
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        gray = rgb2gray(img)
        return img_as_ubyte(gray)
    else:
        return img if img.dtype == np.uint8 else img_as_ubyte(img)


def to_bichromatic(img, threshold=128):
    gray_uint8 = to_grayscale(img)
    binary = np.zeros_like(gray_uint8, dtype=np.uint8)
    binary[gray_uint8 >= threshold] = 255
    return binary


def main():
    parser = argparse.ArgumentParser(
        description="Converts images to grayscale or to bichromatic maps."
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        required=True,
        help="Directory with input images (RGB or grayscale).",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Directory, to which the images are saved.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        required=True,
        choices=["one_channel", "bichromatic"],
        help='Convertion mode: "one_channel" -> grayscale, "bichromatic" -> binary.',
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=128,
        help="Binray threshold (0-255); used only in the bichromatic mode.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    mode = args.mode
    threshold = args.threshold

    if mode == "bichromatic" and not (0 <= threshold <= 255):
        raise ValueError("Threshold must be in range 0–255.")

    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not (
            fname.lower().endswith(".png")
            or fname.lower().endswith(".jpg")
            or fname.lower().endswith(".jpeg")
            or fname.lower().endswith(".bmp")
        ):
            continue

        in_path = os.path.join(input_dir, fname)
        img = imread(in_path)

        if mode == "one_channel":
            out_img = to_grayscale(img)
        else:  # mode == "bichromatic"
            out_img = to_bichromatic(img, threshold)

        out_path = os.path.join(output_dir, fname)
        imsave(out_path, out_img)

    print(
        f"Processed images from '{input_dir}' in mode '{mode}',saved to '{output_dir}'."
    )


if __name__ == "__main__":
    main()
