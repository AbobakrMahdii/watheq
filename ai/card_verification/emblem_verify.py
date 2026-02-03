#!/usr/bin/env python3
"""Deterministic emblem authenticity verification (no ML).

Uses template ROI to crop emblem, compares to reference via template matching
and ORB, then outputs a JSON report and debug images.
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


TEMPLATE_SCORE_STRONG = 0.68
TEMPLATE_SCORE_WEAK = 0.52
ORB_RATIO_STRONG = 0.15
ORB_RATIO_WEAK = 0.10


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


def preprocess_gray(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(resized)


def multi_scale_template_score(crop_gray: np.ndarray, ref_gray: np.ndarray) -> tuple[float, float]:
    best_score = -1.0
    best_scale = 1.0
    for scale in np.arange(0.80, 1.21, 0.05):
        new_w = max(8, int(round(ref_gray.shape[1] * scale)))
        new_h = max(8, int(round(ref_gray.shape[0] * scale)))
        ref_scaled = cv2.resize(ref_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if crop_gray.shape[0] < ref_scaled.shape[0] or crop_gray.shape[1] < ref_scaled.shape[1]:
            continue
        result = cv2.matchTemplate(crop_gray, ref_scaled, cv2.TM_CCOEFF_NORMED)
        if result.size == 0:
            continue
        score = float(result.max())
        if score > best_score:
            best_score = score
            best_scale = float(scale)
    best_score = max(0.0, min(1.0, best_score)) if best_score >= 0 else 0.0
    return best_score, best_scale


def orb_match(crop_gray: np.ndarray, ref_gray: np.ndarray) -> tuple[float, int, np.ndarray]:
    orb_factory = getattr(cv2, "ORB_create", None)
    if orb_factory is not None:
        orb = orb_factory(nfeatures=500)
    else:
        orb_class = getattr(cv2, "ORB", None)
        if orb_class is None:
            raise RuntimeError("OpenCV ORB is not available in this build.")
        orb = orb_class.create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(crop_gray, None)

    if des1 is None or des2 is None or not kp1 or not kp2:
        vis = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
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
    ratio = len(good) / max(1, min_kp)
    ratio = max(0.0, min(1.0, ratio))

    # Safe visualization
    if crop_gray.dtype != np.uint8:
        crop_tmp = np.empty_like(crop_gray)
        crop_gray = cv2.normalize(crop_gray, crop_tmp, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if ref_gray.dtype != np.uint8:
        ref_tmp = np.empty_like(ref_gray)
        ref_gray = cv2.normalize(ref_gray, ref_tmp, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    img1 = cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2BGR) if ref_gray.ndim == 2 else ref_gray
    img2 = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR) if crop_gray.ndim == 2 else crop_gray
    vis = img2.copy()
    valid_matches = []
    kp1_len = len(kp1)
    kp2_len = len(kp2)
    for m in good:
        if 0 <= m.queryIdx < kp1_len and 0 <= m.trainIdx < kp2_len:
            valid_matches.append(m)
    valid_matches = sorted(valid_matches, key=lambda x: x.distance)[:40]
    if not valid_matches:
        return ratio, len(good), vis
    try:
        out_img = np.zeros((1, 1, 3), dtype=np.uint8)
        vis = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            valid_matches,
            out_img,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    except Exception:
        vis = img2

    return ratio, len(good), vis


def decide(score: float, orb_ratio: float) -> tuple[str, list[str]]:
    reasons = []
    if score >= TEMPLATE_SCORE_STRONG and orb_ratio >= ORB_RATIO_STRONG:
        return "EMBLEM_AUTHENTIC", ["template_score strong", "orb_ratio strong"]
    if score >= TEMPLATE_SCORE_WEAK or orb_ratio >= ORB_RATIO_WEAK:
        reasons.append("borderline match")
        return "EMBLEM_SUSPICIOUS", reasons
    reasons.append("low template_score or orb_ratio")
    return "EMBLEM_FORGED", reasons


def draw_overlay(image: np.ndarray, box: list[int]) -> np.ndarray:
    overlay = image.copy()
    x0, y0, x1, y1 = box
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 200, 255), 3)
    cv2.putText(
        overlay,
        "emblem",
        (x0, max(10, y0 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic emblem authenticity verification.")
    parser.add_argument("--image", required=True, help="Path to a card image.")
    parser.add_argument("--rectified", help="Path to a pre-rectified card image.")
    parser.add_argument("--template", required=True, help="Path to layout.yaml.")
    parser.add_argument(
        "--ref",
        default="ai/card_verification/registry/templates/national_id_yemen_v1/refs/emblem.png",
        help="Reference emblem image path.",
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
        print(f"Reference emblem not found: {ref_path}")
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
        print(f"Failed to load reference emblem: {ref_path}")
        return 0

    template = load_template(template_path)
    elements = template.get("elements", {})
    roi = elements.get("emblem", {}).get("roi")

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "emblem_auth_report.json"

    if not roi:
        report = {
            "input_image": str(image_path),
            "rectified_image": str(rectified_path) if rectified_path else None,
            "ref_image": str(ref_path),
            "roi_pixels": None,
            "scores": {
                "best_template_score": 0.0,
                "orb_ratio": 0.0,
                "good_matches": 0,
                "best_scale": None,
            },
            "decision": "ELEMENT_MISSING",
            "reasons": ["emblem ROI missing"],
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print("Emblem decision: ELEMENT_MISSING (score=0.000, orb_ratio=0.000, matches=0)")
        return 0

    roi_pixels = roi_to_pixels(roi, image.shape)
    emblem_crop = crop(image, roi_pixels)

    ref_size = (256, 256)
    ref_gray = preprocess_gray(ref_bgr, ref_size)
    crop_gray = preprocess_gray(emblem_crop, ref_size)

    best_score, best_scale = multi_scale_template_score(crop_gray, ref_gray)
    orb_ratio, good_matches, match_vis = orb_match(crop_gray, ref_gray)

    decision, reasons = decide(best_score, orb_ratio)

    report = {
        "input_image": str(image_path),
        "rectified_image": str(rectified_path) if rectified_path else None,
        "ref_image": str(ref_path),
        "roi_pixels": {"emblem": roi_pixels},
        "scores": {
            "best_template_score": round(float(best_score), 6),
            "orb_ratio": round(float(orb_ratio), 6),
            "good_matches": int(good_matches),
            "best_scale": round(float(best_scale), 6),
        },
        "decision": decision,
        "reasons": reasons,
    }

    crop_path = out_dir / "emblem_crop.png"
    ref_out_path = out_dir / "emblem_ref.png"
    vis_path = out_dir / "emblem_match_vis.png"
    overlay_path = out_dir / "overlay_emblem.png"

    cv2.imwrite(str(crop_path), emblem_crop)
    cv2.imwrite(str(ref_out_path), ref_gray)
    cv2.imwrite(str(vis_path), match_vis)
    cv2.imwrite(str(overlay_path), draw_overlay(image, roi_pixels))

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(
        "Emblem decision: {} (score={:.3f}, orb_ratio={:.3f}, matches={})".format(
            decision, best_score, orb_ratio, good_matches
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
