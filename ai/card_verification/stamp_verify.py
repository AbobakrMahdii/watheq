#!/usr/bin/env python3
"""Deterministic stamp authenticity verification (no ML).

Uses template ROI to crop stamp_expected, tightens with blue mask,
then compares against a reference stamp using template matching and ORB.
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


TEMPLATE_SCORE_STRONG = 0.70
TEMPLATE_SCORE_WEAK = 0.50
ORB_RATIO_STRONG = 0.18
ORB_RATIO_WEAK = 0.10
ORB_RATIO_STRONG_AUTH = 0.35
ORB_GOOD_MATCHES_STRONG = 8


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


def blue_mask_tighten(stamp_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool]:
    if stamp_bgr.size == 0:
        empty_mask = np.zeros((1, 1), dtype=np.uint8)
        return stamp_bgr, empty_mask, 0.0, False

    hsv = cv2.cvtColor(stamp_bgr, cv2.COLOR_BGR2HSV)
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
        return stamp_bgr, mask, blue_area_pct, False

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w <= 0 or h <= 0:
        return stamp_bgr, mask, blue_area_pct, False

    pad = 6
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(stamp_bgr.shape[1], x + w + pad)
    y1 = min(stamp_bgr.shape[0], y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return stamp_bgr, mask, blue_area_pct, False

    return stamp_bgr[y0:y1, x0:x1].copy(), mask, blue_area_pct, True


def preprocess_gray(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(resized)


def template_match_score(stamp_gray: np.ndarray, ref_gray: np.ndarray) -> float:
    stamp_norm = cv2.normalize(stamp_gray, None, 0, 255, cv2.NORM_MINMAX)
    ref_norm = cv2.normalize(ref_gray, None, 0, 255, cv2.NORM_MINMAX)
    result = cv2.matchTemplate(stamp_norm, ref_norm, cv2.TM_CCOEFF_NORMED)
    score = float(result.max()) if result.size else -1.0
    score = max(0.0, min(1.0, score))
    return score


def orb_match(stamp_gray: np.ndarray, ref_gray: np.ndarray) -> tuple[float, int, np.ndarray]:
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(stamp_gray, None)

    if des1 is None or des2 is None or not kp1 or not kp2:
        vis = cv2.cvtColor(stamp_gray, cv2.COLOR_GRAY2BGR)
        return 0.0, 0, vis

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m in matches:
        if len(m) == 2:
            m0, m1 = m
            if m0.distance < 0.75 * m1.distance:
                good.append(m0)

    min_kp = min(len(kp1), len(kp2))
    ratio = len(good) / min_kp if min_kp > 0 else 0.0
    ratio = max(0.0, min(1.0, ratio))

    vis = cv2.drawMatches(
        ref_gray,
        kp1,
        stamp_gray,
        kp2,
        sorted(good, key=lambda x: x.distance)[:40],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return ratio, len(good), vis


def decide(template_score: float, orb_ratio: float, good_matches: int) -> tuple[str, list[str]]:
    reasons = []
    if orb_ratio >= ORB_RATIO_STRONG_AUTH and good_matches >= ORB_GOOD_MATCHES_STRONG:
        return "STAMP_AUTHENTIC", ["strong ORB match"]
    if template_score >= TEMPLATE_SCORE_STRONG and orb_ratio >= ORB_RATIO_STRONG:
        return "STAMP_AUTHENTIC", ["template_score strong", "orb_ratio strong"]
    if (TEMPLATE_SCORE_WEAK <= template_score < TEMPLATE_SCORE_STRONG) or (
        ORB_RATIO_WEAK <= orb_ratio < ORB_RATIO_STRONG
    ):
        reasons.append("borderline match")
        return "STAMP_SUSPICIOUS", reasons
    reasons.append("low template_score or orb_ratio")
    return "STAMP_FORGED", reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic stamp authenticity verification.")
    parser.add_argument("--image", required=True, help="Path to a card photo (phone image).")
    parser.add_argument("--rectified", help="Path to a pre-rectified card image.")
    parser.add_argument("--template", required=True, help="Path to template layout.yaml.")
    parser.add_argument(
        "--ref",
        default="ai/card_verification/registry/templates/national_id_yemen_v1/refs/stamp.png",
        help="Reference stamp image path.",
    )
    parser.add_argument(
        "--out_dir",
        default="ai/card_verification/registry/templates/national_id_yemen_v1/runtime",
        help="Output directory for report and debug images.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    rectified_arg = Path(args.rectified) if args.rectified else None
    template_path = Path(args.template)
    ref_path = Path(args.ref)
    out_dir = Path(args.out_dir)

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 0
    if not template_path.exists():
        print(f"Template not found: {template_path}")
        return 0
    if not ref_path.exists():
        print(f"Reference stamp not found: {ref_path}")
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

    ref_bgr = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    if ref_bgr is None:
        print(f"Failed to load reference stamp: {ref_path}")
        return 0

    template = load_template(template_path)
    elements = template.get("elements", {})
    roi = elements.get("stamp_expected", {}).get("roi")
    if not roi:
        print("Missing stamp_expected ROI in template.")
        return 0

    roi_pixels = roi_to_pixels(roi, image.shape)
    stamp_crop = crop(image, roi_pixels)

    stamp_tight, stamp_mask, blue_area_pct, tightened = blue_mask_tighten(stamp_crop)

    ref_size = (256, 256)
    ref_gray = preprocess_gray(ref_bgr, ref_size)
    stamp_gray = preprocess_gray(stamp_tight, ref_size)

    tmpl_score = template_match_score(stamp_gray, ref_gray)
    orb_ratio, good_matches, orb_vis = orb_match(stamp_gray, ref_gray)

    decision, reasons = decide(tmpl_score, orb_ratio, good_matches)

    report = {
        "input_image": str(image_path),
        "rectified_image": str(rectified_path) if rectified_path else None,
        "ref_image": str(ref_path),
        "roi_pixels": {"stamp_expected": roi_pixels},
        "blue_area_pct": round(float(blue_area_pct), 6),
        "scores": {
            "template_score": round(float(tmpl_score), 6),
            "orb_ratio": round(float(orb_ratio), 6),
            "good_matches": int(good_matches),
        },
        "decision": decision,
        "reasons": reasons,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "stamp_auth_report.json"
    crop_path = out_dir / "stamp_crop.png"
    ref_path_out = out_dir / "stamp_ref.png"
    mask_path = out_dir / "stamp_mask.png"
    vis_path = out_dir / "stamp_match_vis.png"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    cv2.imwrite(str(crop_path), stamp_tight)
    cv2.imwrite(str(ref_path_out), ref_gray)
    cv2.imwrite(str(mask_path), stamp_mask)
    cv2.imwrite(str(vis_path), orb_vis)

    print(
        "Thresholds: orb_strong_ratio={:.2f}, orb_strong_matches={}, template_strong={:.2f}, orb_strong={:.2f}, template_weak={:.2f}, orb_weak={:.2f}".format(
            ORB_RATIO_STRONG_AUTH,
            ORB_GOOD_MATCHES_STRONG,
            TEMPLATE_SCORE_STRONG,
            ORB_RATIO_STRONG,
            TEMPLATE_SCORE_WEAK,
            ORB_RATIO_WEAK,
        )
    )
    print(
        "Stamp decision: {} (template_score={:.3f}, orb_ratio={:.3f}, matches={})".format(
            decision, tmpl_score, orb_ratio, good_matches
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
