#!/usr/bin/env python3
"""Interactive tool to extract emblem and stamp reference crops.

Usage:
  python extract_refs.py --image "path/to/card.jpg"
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


def _resize_for_ui(image, resize_width):
    h, w = image.shape[:2]
    if w <= resize_width:
        return image, 1.0
    scale = resize_width / float(w)
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (resize_width, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _select_roi(window_title, display_image):
    x, y, w, h = cv2.selectROI(window_title, display_image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_title)
    return int(x), int(y), int(w), int(h)


def _map_roi_to_original(roi, scale, image_shape):
    x, y, w, h = roi
    if scale != 1.0:
        inv = 1.0 / scale
        x = int(round(x * inv))
        y = int(round(y * inv))
        w = int(round(w * inv))
        h = int(round(h * inv))

    max_w = image_shape[1]
    max_h = image_shape[0]
    x = max(0, min(x, max_w - 1))
    y = max(0, min(y, max_h - 1))
    w = max(0, min(w, max_w - x))
    h = max(0, min(h, max_h - y))
    return x, y, w, h


def _crop(image, roi):
    x, y, w, h = roi
    return image[y:y + h, x:x + w].copy()


def _tighten_stamp_crop(stamp_bgr):
    if stamp_bgr is None or stamp_bgr.size == 0:
        return stamp_bgr, False

    if len(stamp_bgr.shape) == 2:
        stamp_bgr = cv2.cvtColor(stamp_bgr, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(stamp_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 60, 40], dtype=np.uint8)
    upper = np.array([140, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return stamp_bgr, False

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w == 0 or h == 0:
        return stamp_bgr, False

    pad = 8
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(stamp_bgr.shape[1], x + w + pad)
    y1 = min(stamp_bgr.shape[0], y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return stamp_bgr, False

    return stamp_bgr[y0:y1, x0:x1].copy(), True


def main():
    parser = argparse.ArgumentParser(description="Extract emblem and stamp references from a card image.")
    parser.add_argument("--image", required=True, help="Path to a card photo (phone image).")
    parser.add_argument(
        "--out_dir",
        default="ai/card_verification/registry/templates/national_id_yemen_v1/refs",
        help="Output directory for emblem.png and stamp.png."
    )
    parser.add_argument("--resize_width", type=int, default=1200, help="Resize width for UI display.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Input image not found: {image_path}")
        return 1

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return 1

    display_image, scale = _resize_for_ui(image, args.resize_width)

    emblem_roi = _select_roi("Select emblem ROI then press ENTER or SPACE", display_image)
    if emblem_roi[2] == 0 or emblem_roi[3] == 0:
        print("Emblem selection canceled. No output was written.")
        cv2.destroyAllWindows()
        return 1

    stamp_roi = _select_roi("Select stamp ROI then press ENTER or SPACE", display_image)
    if stamp_roi[2] == 0 or stamp_roi[3] == 0:
        print("Stamp selection canceled. No output was written.")
        cv2.destroyAllWindows()
        return 1

    emblem_roi_full = _map_roi_to_original(emblem_roi, scale, image.shape)
    stamp_roi_full = _map_roi_to_original(stamp_roi, scale, image.shape)

    emblem_crop = _crop(image, emblem_roi_full)
    stamp_crop = _crop(image, stamp_roi_full)

    stamp_tight, tightened = _tighten_stamp_crop(stamp_crop)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emblem_path = out_dir / "emblem.png"
    stamp_path = out_dir / "stamp.png"

    cv2.imwrite(str(emblem_path), emblem_crop)
    cv2.imwrite(str(stamp_path), stamp_tight)

    cv2.destroyAllWindows()

    print("Saved:")
    print(f"  {emblem_path}")
    print(f"  {stamp_path}")
    if not tightened:
        print("Note: No blue mask refinement found; stamp crop saved as selected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
