#!/usr/bin/env python3
"""Extract photo ROI and verify face match (no ML in this script).

Workflow:
- Run detect_and_rectify_card.py
- Crop photo ROI from rectified image
- Call Biometric face verification
- Write JSON report + overlay
"""

import argparse
import json
from pathlib import Path
import subprocess
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


def draw_overlay(image: np.ndarray, box: list[int]) -> np.ndarray:
    overlay = image.copy()
    x0, y0, x1, y1 = box
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 3)
    cv2.putText(
        overlay,
        "photo",
        (x0, max(10, y0 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return overlay


def encode_image_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        return b""
    return buf.tobytes()


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract photo ROI and verify face match.")
    parser.add_argument("--image", required=True, help="Path to original card image.")
    parser.add_argument("--template", required=True, help="Path to layout.yaml.")
    parser.add_argument("--selfie", required=True, help="Path to selfie image.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    args = parser.parse_args()

    image_path = Path(args.image)
    template_path = Path(args.template)
    selfie_path = Path(args.selfie)
    out_dir = Path(args.out_dir)

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 0
    if not template_path.exists():
        print(f"Template not found: {template_path}")
        return 0
    if not selfie_path.exists():
        print(f"Selfie not found: {selfie_path}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # Run detect_and_rectify_card
    rectify_script = Path(__file__).resolve().parent / "detect_and_rectify_card.py"
    if rectify_script.exists():
        subprocess.run(
            [sys.executable, str(rectify_script), "--image", str(image_path), "--out_dir", str(out_dir)],
            check=False,
        )
    else:
        print("Warning: detect_and_rectify_card.py not found; using original image.")

    rectified_path = out_dir / "card_rectified.png"
    rectified = cv2.imread(str(rectified_path), cv2.IMREAD_COLOR) if rectified_path.exists() else None
    if rectified is None:
        print("Warning: rectified image not found; using original image for ROI.")
        rectified = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    template = load_template(template_path)
    elements = template.get("elements", {})
    photo_roi = elements.get("photo", {}).get("roi")

    report = {
        "photo_crop": None,
        "face_match": False,
        "similarity": 0.0,
        "decision": "INCONCLUSIVE",
    }

    if not photo_roi:
        print("Warning: photo ROI missing in template.")
        report["status"] = "ELEMENT_MISSING"
    else:
        photo_box = roi_to_pixels(photo_roi, rectified.shape)
        photo_crop = crop(rectified, photo_box)
        photo_crop_path = out_dir / "photo_crop.png"
        overlay_path = out_dir / "overlay_photo.png"

        if photo_crop.size:
            cv2.imwrite(str(photo_crop_path), photo_crop)
            overlay = draw_overlay(rectified, photo_box)
            cv2.imwrite(str(overlay_path), overlay)
            report["photo_crop"] = str(photo_crop_path)

            # Face verification
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            try:
                from Biometric.face_service import FaceService

                service = FaceService()
                photo_bytes = encode_image_bytes(photo_crop)
                selfie_img = cv2.imread(str(selfie_path), cv2.IMREAD_COLOR)
                selfie_bytes = encode_image_bytes(selfie_img) if selfie_img is not None else b""

                if not photo_bytes or not selfie_bytes:
                    report["decision"] = "INCONCLUSIVE"
                else:
                    result = service.verify_faces(photo_bytes, selfie_bytes)
                    report["face_match"] = bool(result.get("match", False))
                    report["similarity"] = float(result.get("similarity", 0.0))
                    report["decision"] = "FACE_MATCH" if report["face_match"] else "FACE_MISMATCH"
            except Exception as exc:
                print(f"Face verification failed: {exc}")
                report["decision"] = "INCONCLUSIVE"
        else:
            print("Warning: photo crop is empty; skipping face verification.")
            report["decision"] = "INCONCLUSIVE"

    report_path = out_dir / "face_verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
