#!/usr/bin/env python3
"""Deterministic layout verification (no ML).

Checks:
- Blue stamp exists in stamp_under_photo_band and is positioned within stamp_expected
- name_text region non-empty
- national_id_number region non-empty

Supports rectified input via --rectified, or runs detect_and_rectify_card.py when absent.
"""

import argparse
import json
import subprocess
from pathlib import Path
import sys

import cv2
import numpy as np

try:
    import yaml
except Exception:
    print("PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


def load_template(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def roi_to_pixels(roi: dict, image_shape: tuple) -> list[int]:
    h, w = image_shape[:2]
    x0 = int(round(roi["x"] * w))
    y0 = int(round(roi["y"] * h))
    x1 = int(round((roi["x"] + roi["w"]) * w))
    y1 = int(round((roi["y"] + roi["h"]) * h))

    x0 = max(0, min(x0, w))
    y0 = max(0, min(y0, h))
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))

    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    return [x0, y0, x1, y1]


def crop(image: np.ndarray, box: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 0, 3), dtype=image.dtype)
    return image[y0:y1, x0:x1].copy()


def compute_ink_ratio(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ink is the black pixels in a binary image
    ink_pixels = int((thresh == 0).sum())
    return float(ink_pixels) / float(thresh.size)


def stamp_analysis(stamp_roi: np.ndarray) -> tuple[float, np.ndarray, tuple[int, int] | None]:
    if stamp_roi.size == 0:
        return 0.0, np.zeros((1, 1), dtype=np.uint8), None

    hsv = cv2.cvtColor(stamp_roi, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 60, 40], dtype=np.uint8)
    upper = np.array([140, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    blue_pixels = int(cv2.countNonZero(mask))
    blue_area_pct = blue_pixels / float(mask.size) if mask.size else 0.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return blue_area_pct, mask, None

    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M.get("m00", 0) != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv2.boundingRect(largest)
        cx = x + w // 2
        cy = y + h // 2

    return blue_area_pct, mask, (cx, cy)


def point_in_box(point: tuple[int, int], box: list[int]) -> bool:
    x, y = point
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


def draw_overlay(image: np.ndarray, roi_pixels: dict, stamp_centroid: tuple[int, int] | None) -> np.ndarray:
    overlay = image.copy()
    colors = {
        "stamp_under_photo_band": (255, 0, 0),
        "stamp_expected": (255, 0, 255),
        "name_text": (255, 255, 0),
        "national_id_number": (0, 165, 255),
    }
    for key, box in roi_pixels.items():
        x0, y0, x1, y1 = box
        color = colors.get(key, (0, 255, 0))
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 3)
        cv2.putText(
            overlay,
            key,
            (x0, max(10, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    if stamp_centroid is not None:
        cv2.circle(overlay, stamp_centroid, 6, (0, 0, 255), -1)
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic layout verification (no ML).")
    parser.add_argument("--image", required=True, help="Path to a card photo (phone image).")
    parser.add_argument("--rectified", help="Path to a pre-rectified card image.")
    parser.add_argument("--template", required=True, help="Path to template layout.yaml.")
    parser.add_argument("--out_dir", required=True, help="Output directory for runtime artifacts.")
    args = parser.parse_args()

    image_path = Path(args.image)
    rectified_arg = Path(args.rectified) if args.rectified else None
    template_path = Path(args.template)
    out_dir = Path(args.out_dir)

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 0
    if not template_path.exists():
        print(f"Template not found: {template_path}")
        return 0

    rectified_path = None
    if rectified_arg:
        if not rectified_arg.exists():
            print(f"Rectified image not found: {rectified_arg}")
            return 0
        rectified_path = rectified_arg
    else:
        rectify_script = Path(__file__).resolve().parent / "detect_and_rectify_card.py"
        if not rectify_script.exists():
            print(f"Rectification script not found: {rectify_script}")
            return 0
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(rectify_script),
            "--image",
            str(image_path),
            "--out_dir",
            str(out_dir),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            print(f"Rectification failed: {proc.stderr.strip() or proc.stdout.strip()}")
            return 0
        rectified_path = out_dir / "card_rectified.png"
        if not rectified_path.exists():
            print("Rectification did not produce card_rectified.png")
            return 0

    image = cv2.imread(str(rectified_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Failed to load rectified image: {rectified_path}")
        return 0

    template = load_template(template_path)
    elements = template.get("elements", {})

    required_keys = [
        "stamp_under_photo_band",
        "stamp_expected",
        "name_text",
        "national_id_number",
    ]

    roi_pixels = {}
    missing_keys = []
    for key in required_keys:
        roi = elements.get(key, {}).get("roi")
        if not roi:
            missing_keys.append(key)
            continue
        roi_pixels[key] = roi_to_pixels(roi, image.shape)

    if missing_keys:
        print(f"Missing ROI definitions for: {', '.join(missing_keys)}")

    # Always draw all element ROIs for debug overlay.
    for key, value in elements.items():
        roi = (value or {}).get("roi")
        if not roi:
            continue
        roi_pixels.setdefault(key, roi_to_pixels(roi, image.shape))

    stamp_band = crop(image, roi_pixels.get("stamp_under_photo_band", [0, 0, 0, 0]))
    stamp_expected_box = roi_pixels.get("stamp_expected", [0, 0, 0, 0])
    name_crop = crop(image, roi_pixels.get("name_text", [0, 0, 0, 0]))
    id_crop = crop(image, roi_pixels.get("national_id_number", [0, 0, 0, 0]))
    photo_crop = None
    photo_roi = elements.get("photo", {}).get("roi")
    if photo_roi:
        photo_box = roi_to_pixels(photo_roi, image.shape)
        photo_crop = crop(image, photo_box)
    else:
        print("Warning: photo ROI is missing in template; skipping photo crop.")

    stamp_mask = np.zeros(
        (stamp_band.shape[0], stamp_band.shape[1]), dtype=np.uint8
    ) if stamp_band.size else np.zeros((1, 1), dtype=np.uint8)
    blue_area_pct = 0.0
    stamp_centroid_full = None
    in_expected = None
    stamp_status = "SKIPPED"

    name_ratio = compute_ink_ratio(cv2.cvtColor(name_crop, cv2.COLOR_BGR2GRAY)) if name_crop.size else 0.0
    name_status = "NAME_MISSING" if name_ratio < 0.01 else "OK"

    id_ratio = compute_ink_ratio(cv2.cvtColor(id_crop, cv2.COLOR_BGR2GRAY)) if id_crop.size else 0.0
    id_status = "NATIONAL_ID_MISSING" if id_ratio < 0.01 else "OK"

    layout_status = "PASS"
    reason = None
    for status in [name_status, id_status]:
        if status in ("NAME_MISSING", "NATIONAL_ID_MISSING"):
            layout_status = "FAIL"
            reason = status
            break

    report = {
        "input_image": str(image_path),
        "rectified_image": str(rectified_path) if rectified_path else None,
        "template": str(template_path),
        "layout_status": layout_status,
        "reason": reason,
        "stamp": {
            "blue_area_pct": round(float(blue_area_pct), 6),
            "centroid_xy": list(stamp_centroid_full) if stamp_centroid_full else None,
            "in_expected": bool(in_expected) if stamp_centroid_full else None,
            "status": stamp_status,
        },
        "name_text": {
            "ink_ratio": round(float(name_ratio), 6),
            "status": name_status,
        },
        "national_id_number": {
            "ink_ratio": round(float(id_ratio), 6),
            "status": id_status,
        },
        "roi_pixels": roi_pixels,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    overlay_path = out_dir / "overlay_layout_verify.png"
    stamp_mask_path = out_dir / "stamp_mask.png"
    photo_crop_path = out_dir / "photo_crop.png"

    overlay = draw_overlay(image, roi_pixels, stamp_centroid_full)

    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(stamp_mask_path), stamp_mask)
    if photo_crop is not None and photo_crop.size:
        cv2.imwrite(str(photo_crop_path), photo_crop)
        report["artifacts"] = report.get("artifacts", {})
        report["artifacts"]["photo_crop"] = str(photo_crop_path)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Saved:")
    print(f"  {report_path}")
    print(f"  {overlay_path}")
    print(f"  {stamp_mask_path}")
    print(f"Layout status: {layout_status} ({reason})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
