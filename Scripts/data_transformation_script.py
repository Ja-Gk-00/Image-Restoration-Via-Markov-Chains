import os
import argparse
import yaml
import cv2
import numpy as np


# Registry for transformations
def registered_transformation(func):
    TRANSFORMATIONS[func.__name__] = func
    return func


# Dictionary to hold transformation functions
TRANSFORMATIONS = {}


@registered_transformation
def add_noise(image: np.ndarray, mean: float = 0.0, var: float = 0.01) -> np.ndarray:
    img = image.astype(np.float32) / 255.0
    noise = np.random.normal(mean, var**0.5, img.shape)
    noisy = img + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    return (noisy * 255).astype(np.uint8)


@registered_transformation
def add_camera_noise(
    image: np.ndarray, shot_noise_scale: float = 0.01, read_noise_var: float = 0.001
) -> np.ndarray:
    img = image.astype(np.float32) / 255.0
    shot = np.random.poisson(img * shot_noise_scale) / shot_noise_scale
    read = np.random.normal(0.0, read_noise_var**0.5, img.shape)
    noisy = img + shot + read
    noisy = np.clip(noisy, 0.0, 1.0)
    return (noisy * 255).astype(np.uint8)


@registered_transformation
def make_old(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    # Convert to grayscale using luminance weights
    if image.ndim == 3 and image.shape[2] >= 3:
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = image.astype(np.float32)
    arr = gray.astype(np.float32) / 255.0
    sepia = np.zeros_like(image, dtype=np.uint8)
    tr = (arr * (1 - intensity) + arr * intensity * 0.393) * 255
    tg = (arr * (1 - intensity) + arr * intensity * 0.769) * 255
    tb = (arr * (1 - intensity) + arr * intensity * 0.189) * 255
    sepia[..., 0] = np.clip(tr, 0, 255).astype(np.uint8)
    sepia[..., 1] = np.clip(tg, 0, 255).astype(np.uint8)
    sepia[..., 2] = np.clip(tb, 0, 255).astype(np.uint8)
    return sepia


def load_image(path: str) -> np.ndarray:
    print(f"[DEBUG] Loading image from {path}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read image {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(array: np.ndarray, path: str):
    print(f"[DEBUG] Saving image to {path}")
    if array.ndim == 3 and array.shape[2] == 3:
        out = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    else:
        out = array
    cv2.imwrite(path, out)


def save_numpy(array: np.ndarray, path: str):
    print(f"[DEBUG] Saving numpy array to {path}")
    np.save(path, array)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch image transformer")
    parser.add_argument(
        "--input_dir", required=True, help="Path to input images folder"
    )
    parser.add_argument(
        "--output_dir_img", required=True, help="Folder to save transformed images"
    )
    parser.add_argument(
        "--output_dir_np", required=True, help="Folder to save numpy arrays"
    )
    parser.add_argument(
        "--transform", required=True, help="Name of the transformation to apply"
    )
    parser.add_argument(
        "--config", required=False, help="Path to YAML config file with parameters"
    )
    parser.add_argument(
        "--mode",
        choices=["gray", "rgb"],
        default="rgb",
        help="Output mode: grayscale (gray) or 3-channel RGB (rgb)",
    )
    return parser.parse_args()


def main():
    print("[DEBUG] Starting main()")
    args = parse_args()
    print(f"[DEBUG] Parsed arguments: {args}")

    params = {}
    if args.config:
        print(f"[DEBUG] Loading config from {args.config}")
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
            params = cfg.get(args.transform, {})
        print(f"[DEBUG] Parameters for '{args.transform}': {params}")

    if args.transform not in TRANSFORMATIONS:
        print(
            f"Error: transformation '{args.transform}' not found. Available: {list(TRANSFORMATIONS.keys())}"
        )
        return
    func = TRANSFORMATIONS[args.transform]

    print(f"[DEBUG] Creating output dirs: {args.output_dir_img}, {args.output_dir_np}")
    os.makedirs(args.output_dir_img, exist_ok=True)
    os.makedirs(args.output_dir_np, exist_ok=True)

    files = [
        f
        for f in os.listdir(args.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]
    print(f"[DEBUG] Found {len(files)} image(s) in {args.input_dir}")
    if not files:
        print("Warning: No images to process. Check your input directory.")
        return

    for fname in files:
        in_path = os.path.join(args.input_dir, fname)
        print(f"[DEBUG] Processing file: {fname}")
        try:
            array = load_image(in_path)
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue

        out_array = func(array, **params)
        print(f"[DEBUG] Applied transformation: {args.transform}")

        if args.mode == "gray":
            if out_array.ndim == 3 and out_array.shape[2] >= 3:
                gray = np.dot(out_array[..., :3], [0.2989, 0.5870, 0.1140])
                out_array = gray.astype(np.uint8)
                print(f"[DEBUG] Converted to grayscale mode")
        else:
            if out_array.ndim == 2:
                out_array = np.stack([out_array] * 3, axis=-1)
                print(f"[DEBUG] Converted single-channel to RGB stack")

        param_str = ""
        if params:
            parts = []
            for k, v in params.items():
                val = str(v).replace(".", "p")
                parts.append(f"{k}{val}")
            param_str = "_" + "_".join(parts)
        suffix = f"_{args.transform}" + param_str
        if args.mode == "gray":
            suffix += "_gray"

        base, ext = os.path.splitext(fname)
        img_out_path = os.path.join(args.output_dir_img, f"{base}{suffix}{ext}")
        np_out_path = os.path.join(args.output_dir_np, f"{base}{suffix}.npy")
        save_image(out_array, img_out_path)
        save_numpy(out_array, np_out_path)

        print(f"Processed {fname} -> {img_out_path}, {np_out_path}")


if __name__ == "__main__":
    main()
