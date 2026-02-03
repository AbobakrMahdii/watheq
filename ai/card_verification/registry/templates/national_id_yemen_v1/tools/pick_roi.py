#!/usr/bin/env python3
"""Interactive ROI picker for updating layout.yaml elements.

Uses cv2.selectROI to pick a pixel rectangle and converts it to ratio ROI.
Optionally updates the template YAML with --write.
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


def load_template(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_template(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive ROI picker.")
    parser.add_argument("--image", required=True, help="Path to image for ROI selection.")
    parser.add_argument("--template", required=True, help="Path to layout.yaml.")
    parser.add_argument("--element", required=True, help="Element name to update.")
    parser.add_argument("--write", action="store_true", help="Write ROI into layout.yaml.")
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

    h, w = image.shape[:2]
    window_title = f"Select ROI for: {args.element}"
    x, y, rw, rh = cv2.selectROI(window_title, image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_title)

    if rw == 0 or rh == 0:
        print("Selection canceled. No changes made.")
        return 1

    x0, y0, x1, y1 = int(x), int(y), int(x + rw), int(y + rh)
    rx = round(x0 / w, 6)
    ry = round(y0 / h, 6)
    rw_r = round(rw / w, 6)
    rh_r = round(rh / h, 6)

    print(f"Selected pixels: [{x0},{y0},{x1},{y1}]")
    print(f"ROI ratios: {{x: {rx}, y: {ry}, w: {rw_r}, h: {rh_r}}}")

    if args.write:
        data = load_template(template_path)
        elements = data.get("elements", {})
        if args.element not in elements:
            print(f"Element '{args.element}' not found in template elements.")
            return 1
        elem = elements.get(args.element, {})
        elem["roi"] = {"x": rx, "y": ry, "w": rw_r, "h": rh_r}
        elements[args.element] = elem
        data["elements"] = elements
        save_template(template_path, data)
        print("Updated layout.yaml")

    return 0


if __name__ == "__main__":
    sys.exit(main())
