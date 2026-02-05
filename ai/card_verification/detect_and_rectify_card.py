#!/usr/bin/env python3
"""Detect and rectify an ID card from a full image (no ML).

Finds a card-like contour, orders the quad, applies perspective warp to a fixed
top-down size, and writes outputs + a JSON report.
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

def _try_rectify(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, float, np.ndarray, np.ndarray] | tuple[None, str]:
    h, w = image.shape[:2]
    image_area = float(h * w)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 180)
    kernel = np.ones((7, 7), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if not contours:
        return None, "NO_CARD_QUAD"

    card_contour = contours[0]
    contour_area = cv2.contourArea(card_contour)
    area_ratio = contour_area / image_area if image_area > 0 else 0.0
    if area_ratio < 0.15:
        return None, "NO_CARD_QUAD"

    peri = cv2.arcLength(card_contour, True)
    approx = cv2.approxPolyDP(card_contour, 0.02 * peri, True)
    if len(approx) == 4:
        quad = approx
        method = "quad"
    else:
        rect = cv2.minAreaRect(card_contour)
        quad = cv2.boxPoints(rect)
        quad = quad.reshape(4, 2).astype(np.float32)
        method = "min_area_rect"

    ordered = order_points(quad)
    out_w, out_h = 856, 540
    if out_w <= 0 or out_h <= 0:
        return None, "BAD_RECTIFIED"

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))
    if warped is None or warped.size == 0:
        return None, "BAD_RECTIFIED"

    debug = image.copy()
    cv2.drawContours(debug, [card_contour], -1, (0, 0, 255), 2)
    cv2.polylines(debug, [ordered.astype(int)], True, (0, 255, 0), 2)
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

    return warped, ordered, method, contour_area, edges, debug


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
    print(f"IN: {w}x{h}")

    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_input_path = debug_dir / "input.jpg"
    debug_edges_path = debug_dir / "edges.jpg"
    debug_contours_path = debug_dir / "contours.jpg"
    rectified_path = out_dir / "card_rectified.png"
    debug_rectified_path = debug_dir / "rectified.jpg"
    report_path = out_dir / "rectification_report.json"

    cv2.imwrite(str(debug_input_path), image)

    warped = None
    ordered = None
    method = None
    contour_area = 0.0
    edges = None
    debug = None
    last_reason = "NO_CARD_QUAD"

    for angle in (0, 90, 180, 270):
        rotated = image if angle == 0 else cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE if angle == 90 else
            cv2.ROTATE_180 if angle == 180 else
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )
        result = _try_rectify(rotated)
        if result and result[0] is not None:
            warped, ordered, method, contour_area, edges, debug = result
            break
        if isinstance(result, tuple):
            last_reason = result[1]

    if warped is None or ordered is None or edges is None or debug is None:
        print(last_reason)
        return 1

    cv2.imwrite(str(debug_edges_path), edges)
    cv2.imwrite(str(debug_contours_path), debug)
    cv2.imwrite(str(rectified_path), warped)
    cv2.imwrite(str(debug_rectified_path), warped)

    print("RECT: 856x540")

    aspect_ratio = 856 / 540
    confidence = contour_confidence(contour_area, image_area, ordered)
    report = {
        "status": "RECTIFIED",
        "method": method,
        "corners": ordered.astype(int).tolist(),
        "original_size": {"width": w, "height": h},
        "rectified_size": {"width": 856, "height": 540},
        "aspect_ratio": round(float(aspect_ratio), 6) if aspect_ratio else None,
        "confidence": round(float(confidence), 6),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved rectified: {rectified_path}")
    print(f"Saved debug input: {debug_input_path}")
    print(f"Saved debug edges: {debug_edges_path}")
    print(f"Saved debug contours: {debug_contours_path}")
    print(f"Saved debug rectified: {debug_rectified_path}")
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
