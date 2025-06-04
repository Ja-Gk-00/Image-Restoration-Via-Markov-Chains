import os
import argparse
import numpy as np
from skimage.io import imread, imsave

def flip_random_pixels(img, flip_prob):
    binary = (img > 128).astype(np.uint8)  # 1 for white, 0 for black
    mask = np.random.rand(*binary.shape) < flip_prob
    binary[mask] = 1 - binary[mask]
    # Convert back to 0/255
    return (binary * 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(
        description="Apply salt-and-pepper flipping to bichromatic images."
    )
    parser.add_argument(
        "--input_dir", "-i", type=str, required=True,
        help="Directory containing input bichromatic images"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, required=True,
        help="Directory to save flipped output images"
    )
    parser.add_argument(
        "--flip_prob", "-p", type=float, required=True,
        help="Probability (between 0 and 1) to flip each pixel"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    flip_prob = args.flip_prob

    if not 0 <= flip_prob <= 1:
        raise ValueError("flip_prob must be between 0 and 1")

    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not (fname.lower().endswith(".png")
                or fname.lower().endswith(".jpg")
                or fname.lower().endswith(".jpeg")
                or fname.lower().endswith(".bmp")):
            continue

        in_path = os.path.join(input_dir, fname)
        img = imread(in_path, as_gray=True)
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # Flip pixels
        flipped = flip_random_pixels(img, flip_prob)

        out_path = os.path.join(output_dir, fname)
        imsave(out_path, flipped)

    print(f"Processed all images from '{input_dir}', saved to '{output_dir}'")

if __name__ == "__main__":
    main()
