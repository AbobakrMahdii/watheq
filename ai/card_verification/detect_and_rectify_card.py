#!/usr/bin/env python3
"""Detect and rectify an ID card from a full image (no ML).

Extracts the largest 4-point contour (if it covers >10% of image area),
warps it to a frontal view, and writes outputs + a JSON report.
Falls back to the largest contour bounding box when no 4-point contour is found.
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def contour_confidence(contour_area: float, image_area: float, approx: np.ndarray) -> float:
    area_ratio = contour_area / image_area if image_area > 0 else 0.0
    # Confidence boosts for near-rectangular shape and good area coverage
    shape_bonus = 0.15 if approx is not None and len(approx) == 4 else 0.0
    conf = min(1.0, area_ratio + shape_bonus)
    return float(max(0.0, conf))


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect and rectify an ID card from an image (no ML).")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.out_dir)

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 0

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return 0

    h, w = image.shape[:2]
    image_area = float(h * w)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5
    )
    edges = cv2.Canny(thresh, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    card_contour = None
    card_approx = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.10 * image_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            card_contour = cnt
            card_approx = approx
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    debug_contours_path = out_dir / "debug_contours.png"
    rectified_path = out_dir / "card_rectified.png"
    report_path = out_dir / "rectification_report.json"

    if card_contour is None and not contours:
        debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(debug_contours_path), debug)

        report = {
            "status": "NOT_A_CARD",
            "method": None,
            "corners": None,
            "original_size": {"width": w, "height": h},
            "rectified_size": None,
            "aspect_ratio": None,
            "confidence": 0.0,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print("Status: NOT_A_CARD")
        print(f"Saved debug contours: {debug_contours_path}")
        print(f"Saved report: {report_path}")
        return 0

    used_fallback = False
    if card_contour is None and contours:
        card_contour = contours[0]
        used_fallback = True

    debug = image.copy()
    for cnt in contours[:10]:
        cv2.drawContours(debug, [cnt], -1, (0, 0, 255), 2)

    if used_fallback:
        x, y, bw, bh = cv2.boundingRect(card_contour)
        ordered = np.array(
            [[x, y], [x + bw - 1, y], [x + bw - 1, y + bh - 1], [x, y + bh - 1]],
            dtype=np.float32,
        )
        max_width, max_height = bw, bh
    else:
        ordered = order_points(card_approx)
        (tl, tr, br, bl) = ordered

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)

        max_width = int(round(max(width_a, width_b)))
        max_height = int(round(max(height_a, height_b)))

    if max_width <= 0 or max_height <= 0:
        max_width = w
        max_height = h

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype=np.float32)

    if used_fallback:
        x0, y0 = int(ordered[0][0]), int(ordered[0][1])
        x1, y1 = int(ordered[2][0]) + 1, int(ordered[2][1]) + 1
        warped = image[y0:y1, x0:x1].copy()
        cv2.rectangle(debug, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
    else:
        M = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        cv2.drawContours(debug, [card_approx], -1, (0, 255, 0), 2)

    for i, pt in enumerate(ordered):
        cv2.circle(debug, (int(pt[0]), int(pt[1])), 6, (255, 0, 0), -1)
        cv2.putText(
            debug,
            f"{i}",
            (int(pt[0]) + 8, int(pt[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    cv2.imwrite(str(rectified_path), warped)
    cv2.imwrite(str(debug_contours_path), debug)

    aspect_ratio = max_width / max_height if max_height > 0 else None
    if used_fallback:
        confidence = 0.4
        status = "RECTIFIED_FALLBACK"
        method = "fallback_bbox"
    else:
        confidence = contour_confidence(cv2.contourArea(card_contour), image_area, card_approx)
        status = "RECTIFIED"
        method = "quad"

    report = {
        "status": status,
        "method": method,
        "corners": ordered.astype(int).tolist(),
        "original_size": {"width": w, "height": h},
        "rectified_size": {"width": max_width, "height": max_height},
        "aspect_ratio": round(float(aspect_ratio), 6) if aspect_ratio else None,
        "confidence": round(float(confidence), 6),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Status: {status}")
    if used_fallback:
        print("Fallback used due to weak card edges")
    print(f"Saved rectified: {rectified_path}")
    print(f"Saved debug contours: {debug_contours_path}")
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
