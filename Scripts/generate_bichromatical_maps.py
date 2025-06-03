import os
import argparse
import numpy as np
from skimage.io import imsave


def generate_checkerboard(width, height, num_tiles=8):
    tile_w = width // num_tiles
    tile_h = height // num_tiles
    board = np.zeros((height, width), dtype=np.uint8)
    for i in range(num_tiles):
        for j in range(num_tiles):
            if (i + j) % 2 == 0:
                board[i * tile_h : (i + 1) * tile_h, j * tile_w : (j + 1) * tile_w] = (
                    255
                )
    return board


def generate_vertical_stripes(width, height, stripe_width=10):
    img = np.zeros((height, width), dtype=np.uint8)
    for x in range(width):
        if (x // stripe_width) % 2 == 0:
            img[:, x] = 255
    return img


def generate_horizontal_stripes(width, height, stripe_height=10):
    img = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        if (y // stripe_height) % 2 == 0:
            img[y, :] = 255
    return img


def generate_diagonal_stripes(width, height, stripe_width=10):
    img = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if ((x + y) // stripe_width) % 2 == 0:
                img[y, x] = 255
    return img


def generate_border(width, height, border_thickness=10):
    img = np.zeros((height, width), dtype=np.uint8)
    img[:border_thickness, :] = 255
    img[-border_thickness:, :] = 255
    img[:, :border_thickness] = 255
    img[:, -border_thickness:] = 255
    return img


def generate_cross(width, height, line_thickness=10):
    img = np.zeros((height, width), dtype=np.uint8)
    cy, cx = height // 2, width // 2
    img[cy - line_thickness // 2 : cy + line_thickness // 2 + 1, :] = 255
    img[:, cx - line_thickness // 2 : cx + line_thickness // 2 + 1] = 255
    return img


def generate_random_noise(width, height):
    return (np.random.rand(height, width) > 0.5).astype(np.uint8) * 255


def generate_concentric_squares(width, height, num_squares=5):
    img = np.zeros((height, width), dtype=np.uint8)
    min_dim = min(width, height)
    step = min_dim // (2 * num_squares)
    for i in range(num_squares):
        start = i * step
        end_x = width - i * step
        end_y = height - i * step
        img[start:end_y, start : start + step] = 255
        img[start:end_y, end_x - step : end_x] = 255
        img[start : start + step, start:end_x] = 255
        img[end_y - step : end_y, start:end_x] = 255
    return img


def generate_circle(width, height, radius=None):
    if radius is None:
        radius = min(width, height) // 4
    img = np.zeros((height, width), dtype=np.uint8)
    cy, cx = height // 2, width // 2
    Y, X = np.ogrid[:height, :width]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius**2
    img[mask] = 255
    return img


def generate_random_blocks(width, height, block_size=20):
    img = np.zeros((height, width), dtype=np.uint8)
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            if np.random.rand() > 0.5:
                img[y : y + block_size, x : x + block_size] = 255
    return img


PATTERNS = {
    "checkerboard": generate_checkerboard,
    "vertical_stripes": generate_vertical_stripes,
    "horizontal_stripes": generate_horizontal_stripes,
    "diagonal_stripes": generate_diagonal_stripes,
    "border": generate_border,
    "cross": generate_cross,
    "random_noise": generate_random_noise,
    "concentric_squares": generate_concentric_squares,
    "circle": generate_circle,
    "random_blocks": generate_random_blocks,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate black-and-white pattern images without PIL."
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        nargs=2,
        type=int,
        required=True,
        metavar=("WIDTH", "HEIGHT"),
        help="Resolution (width height) of generated images",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    width, height = args.resolution
    os.makedirs(output_dir, exist_ok=True)

    for name, func in PATTERNS.items():
        img_array = func(width, height)
        imsave(os.path.join(output_dir, f"{name}.png"), img_array)
    print(f"Generated {len(PATTERNS)} pattern images in '{output_dir}'")


if __name__ == "__main__":
    main()
