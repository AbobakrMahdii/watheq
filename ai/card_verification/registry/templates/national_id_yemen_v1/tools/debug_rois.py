#!/usr/bin/env python3
"""Deterministic ROI debug tool for a card template.

Draws template ROIs on a rectified image (or the original image if
rectification fails) and writes debug outputs.
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

try:
    import yaml
except Exception:
    print("PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


COLOR_MAP = {
    "emblem": (0, 200, 255),              # orange
    "photo": (0, 255, 0),                 # green
    "stamp_under_photo_band": (255, 0, 0),# blue
    "stamp_expected": (255, 0, 255),      # magenta
    "name_text": (255, 255, 0),           # cyan
    "national_id_number": (0, 165, 255),  # orange
}


def load_template(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_card_quad(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:10]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None


def rectify_image(image: np.ndarray, output_size: tuple | None) -> tuple[np.ndarray, bool]:
    quad = detect_card_quad(image)
    if quad is None:
        return image, False

    quad = order_quad_points(quad)

    if output_size is not None:
        out_w, out_h = output_size
    else:
        w1 = np.linalg.norm(quad[1] - quad[0])
        w2 = np.linalg.norm(quad[2] - quad[3])
        h1 = np.linalg.norm(quad[3] - quad[0])
        h2 = np.linalg.norm(quad[2] - quad[1])
        out_w = int(round(max(w1, w2)))
        out_h = int(round(max(h1, h2)))

    if out_w <= 0 or out_h <= 0:
        return image, False

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))
    return warped, True


def roi_to_pixels(roi: dict, image_shape: tuple) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    x = int(round(roi["x"] * w))
    y = int(round(roi["y"] * h))
    rw = int(round(roi["w"] * w))
    rh = int(round(roi["h"] * h))

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(0, min(rw, w - x))
    rh = max(0, min(rh, h - y))
    return x, y, rw, rh


def draw_rois(image: np.ndarray, elements: dict, overlay_rois: list, print_boxes: bool) -> np.ndarray:
    overlay = image.copy()
    img_h, img_w = overlay.shape[:2]
    for key in overlay_rois:
        if key not in elements:
            print(f"Warning: element '{key}' listed in overlay_rois but not found in elements.")
            continue
        if key not in elements:
            continue
        roi = elements[key].get("roi")
        if not roi:
            print(f"Warning: element '{key}' has no roi defined.")
            continue
        x, y, w, h = roi_to_pixels(roi, overlay.shape)
        color = COLOR_MAP.get(key, (255, 255, 0))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 4)
        label = key
        if print_boxes:
            x1 = x + w
            y1 = y + h
            print(
                f"{key} | ratio={roi} | pixels=({x},{y},{x1},{y1}) | image_size=({img_w},{img_h})"
            )
        cv2.putText(
            overlay,
            label,
            (x, max(10, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Debug template ROIs on a card image.")
    parser.add_argument("--image", required=True, help="Path to a card photo (phone image).")
    parser.add_argument("--template", required=True, help="Path to template layout.yaml.")
    parser.add_argument("--print_boxes", action="store_true", default=True,
                        help="Print ratio and pixel boxes for overlay_rois.")
    parser.add_argument("--no_print_boxes", action="store_false", dest="print_boxes",
                        help="Disable printing of ratio and pixel boxes.")
    args = parser.parse_args()

    image_path = Path(args.image)
    template_path = Path(args.template)

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 1
    if not template_path.exists():
        print(f"Template not found: {template_path}")
        return 1

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return 1

    template = load_template(template_path)
    elements = template.get("elements", {})
    debug_cfg = template.get("debug", {})
    overlay_rois = debug_cfg.get(
        "overlay_rois",
        ["emblem", "photo", "stamp_under_photo_band", "stamp_expected"],
    )

    output_size = None
    rect_cfg = template.get("rectification", {})
    if isinstance(rect_cfg, dict):
        out = rect_cfg.get("output_size")
        if isinstance(out, dict) and "width" in out and "height" in out:
            output_size = (int(out["width"]), int(out["height"]))

    rectified, used_rectify = rectify_image(image, output_size)
    if not used_rectify:
        print("Rectification failed; using original image for ROI overlay.")

    overlay = draw_rois(rectified, elements, overlay_rois, args.print_boxes)

    debug_dir = Path("ai/card_verification/registry/templates/national_id_yemen_v1/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    rectified_path = debug_dir / "rectified.png"
    overlay_path = debug_dir / "overlay_rois.png"

    cv2.imwrite(str(rectified_path), rectified)
    cv2.imwrite(str(overlay_path), overlay)

    print("Saved:")
    print(f"  {rectified_path}")
    print(f"  {overlay_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
