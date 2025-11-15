#!/usr/bin/env python3
"""Batch preprocessing for signature images.

The aim of this script is to standardise the appearance of signature
images before feeding them into machine learning models. It performs
the following operations on every image found under the input
directory:

1. Converts the image to grayscale (if not already).
2. Resizes it to a fixed square size (default 224×224) while
   maintaining aspect ratio by padding.
3. Optionally applies Gaussian blur to remove small specks of noise.
4. Optionally applies Otsu thresholding to produce a binary image.

Processed images are saved to an output directory that mirrors the
structure of the input directory. For example, if your input data is
stored under ``datasets/custom_signature_dataset/train/genuine``, the
processed images will be saved under ``processed/train/genuine``.

Usage:
    python preprocess_batch.py --src datasets/custom_signature_dataset/train/genuine \
                               --dest processed/train/genuine \
                               --size 224 --blur 3 --threshold
"""

import argparse
import cv2
import numpy as np
import os
from pathlib import Path


def resize_and_pad(image, size: int):
    """Resize a grayscale image to a square canvas with padding.

    The original aspect ratio is preserved by scaling the longer
    side down to the target size and padding the shorter side with
    white pixels (value 255).

    Args:
        image (numpy.ndarray): Input grayscale image (2D array).
        size (int): Desired width and height of the output.

    Returns:
        numpy.ndarray: A ``size×size`` grayscale image.
    """
    h, w = image.shape[:2]
    # compute scaling factor
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    # resize image while preserving aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # create a white canvas
    canvas = 255 * np.ones((size, size), dtype=resized.dtype)
    # compute top-left coordinates to center the resized image
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    # paste resized image onto the canvas
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def process_image(input_path: Path, output_path: Path, size: int, blur_kernel: int, threshold: bool) -> None:
    """Process a single image and save the result.

    Args:
        input_path: Path to the input image.
        output_path: Path to save the processed image.
        size: Desired output square size.
        blur_kernel: Kernel size for Gaussian blur (use 0 to disable).
        threshold: Whether to apply Otsu thresholding after blur.
    """
    img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: could not read image {input_path}")
        return
    # resize & pad
    processed = resize_and_pad(img, size)
    # blur
    if blur_kernel > 0:
        processed = cv2.GaussianBlur(processed, (blur_kernel, blur_kernel), 0)
    # threshold
    if threshold:
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), processed)


def walk_and_process(src_dir: Path, dest_dir: Path, size: int, blur_kernel: int, threshold: bool) -> None:
    """Recursively process all images under ``src_dir``.

    Args:
        src_dir: Input directory.
        dest_dir: Output directory.
        size: Target size for images.
        blur_kernel: Gaussian blur kernel size.
        threshold: Whether to apply thresholding.
    """
    for root, dirs, files in os.walk(src_dir):
        rel_root = Path(root).relative_to(src_dir)
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                in_path = Path(root) / file
                out_path = dest_dir / rel_root / file
                process_image(in_path, out_path, size, blur_kernel, threshold)


def main():
    parser = argparse.ArgumentParser(description="Batch preprocess signature images.")
    parser.add_argument("--src", type=str, required=True, help="Input directory containing images.")
    parser.add_argument("--dest", type=str, required=True, help="Destination directory for processed images.")
    parser.add_argument("--size", type=int, default=224, help="Output square dimension.")
    parser.add_argument("--blur", type=int, default=3,
                        help="Gaussian blur kernel size (odd integer, 0 to disable).")
    parser.add_argument("--threshold", action="store_true",
                        help="Apply Otsu thresholding to binarise the image.")
    args = parser.parse_args()

    src_path = Path(args.src)
    dest_path = Path(args.dest)
    if not src_path.exists():
        raise ValueError(f"Input directory {src_path} does not exist")
    walk_and_process(src_path, dest_path, args.size, args.blur, args.threshold)
    print(f"Processing complete. Output saved to {dest_path}")


if __name__ == "__main__":
    main()
