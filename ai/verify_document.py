#!/usr/bin/env python3
"""
Watheq Document Verification Script (v3 — Trained Classifiers + Font Analysis)

التحقق من أصالة الوثائق باستخدام:
1. كشف العناصر بناءً على التخطيط المحفوظ (layout_config.yaml)
2. مصنف ثنائي مدرب لكل عنصر — القرار من التعلم لا من المراجع
3. تحليل خصائص الخطوط للمناطق النصية
4. التحقق من مواقع العناصر وأحجامها
5. قرار نهائي مرجح — الحد الأدنى للنجاح 85%

Usage:
    python ai/verify_document.py --image doc.jpg --type identity --json
    python ai/verify_document.py --image doc.jpg --type passport

Returns detailed JSON:
{
    "decision": "PASSED|SUSPICIOUS|FAILED",
    "overall_confidence": 0.92,
    "elements": { ... per-element details ... },
    "missing_elements": [],
    "anomalies": [],
    "processing_time_ms": 1200
}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

AI_DIR = Path(__file__).parent.resolve()
MODELS_DIR = AI_DIR / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
FONTS_DIR = MODELS_DIR / "fonts"
REFERENCES_DIR = AI_DIR / "data" / "refrences"
TRAINING_DIR = AI_DIR / "data" / "training"

# Ensure ai/ directory is importable
if str(AI_DIR.parent) not in sys.path:
    sys.path.insert(0, str(AI_DIR.parent))


# ─────────────── Layout Config Loading ───────────────

def _load_layout_config(doc_type: str) -> Optional[Dict[str, Any]]:
    """Load layout_config.yaml for a document type."""
    config_path = REFERENCES_DIR / doc_type / "layout_config.yaml"
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_training_config(doc_type: str) -> Optional[Dict[str, Any]]:
    """Load training config to get learned layout positions."""
    config_path = TRAINING_DIR / doc_type / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return None


def _get_all_elements(layout_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all element definitions from layout config."""
    elements = []
    for ref_stem, elem_data in layout_config.get("elements", {}).items():
        elements.append({
            "ref_stem": ref_stem,
            "class_name": elem_data["class_name"],
            "roi": elem_data.get("roi", {}),
            "tolerance": elem_data.get("tolerance", 0.10),
            "weight": elem_data.get("weight", 1.0),
            "critical": elem_data.get("critical", False),
            "type": elem_data.get("type", "visual"),
        })
    for text_name, text_data in layout_config.get("text_regions", {}).items():
        elements.append({
            "ref_stem": text_name,
            "class_name": text_data["class_name"],
            "roi": text_data.get("roi", {}),
            "tolerance": text_data.get("tolerance", 0.12),
            "weight": text_data.get("weight", 1.0),
            "critical": text_data.get("critical", False),
            "type": "text",
        })
    return elements


# ─────────────── Element Detection (Template-based) ───────────────

def _detect_elements(
    image: np.ndarray, elements: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Detect elements using layout ROI positions from config.
    Returns list of detections with pixel and normalized bounding boxes.
    """
    h, w = image.shape[:2]
    detections = []

    for elem in elements:
        roi = elem.get("roi", {})
        if not roi:
            continue

        rx = roi.get("x", 0)
        ry = roi.get("y", 0)
        rw = roi.get("w", 0.1)
        rh = roi.get("h", 0.1)

        px = int(rx * w)
        py = int(ry * h)
        pw = int(rw * w)
        ph = int(rh * h)

        detections.append({
            "class_name": elem["class_name"],
            "ref_stem": elem["ref_stem"],
            "confidence": 0.95,  # Template-based confidence
            "bbox": [px, py, pw, ph],
            "bbox_norm": [round(rx, 4), round(ry, 4), round(rw, 4), round(rh, 4)],
            "elem_type": elem["type"],
            "weight": elem["weight"],
            "critical": elem["critical"],
            "tolerance": elem["tolerance"],
            "expected_roi": roi,
        })

    return detections


def _crop_element(image: np.ndarray, bbox: List[int], padding: float = 0.05) -> np.ndarray:
    """Crop an element from an image using bounding box with padding."""
    h, w = image.shape[:2]
    x, y, bw, bh = bbox
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + bw + pad_x)
    y1 = min(h, y + bh + pad_y)
    return image[y0:y1, x0:x1].copy()


# ─────────────── Position Validation ───────────────

def _validate_position(
    detected_bbox_norm: List[float],
    expected_roi: Dict[str, float],
    tolerance: float,
) -> bool:
    """Check if detected position is within expected tolerance."""
    dx, dy = detected_bbox_norm[0], detected_bbox_norm[1]
    ex, ey = expected_roi.get("x", 0), expected_roi.get("y", 0)
    return abs(dx - ex) <= tolerance and abs(dy - ey) <= tolerance


def _validate_size(
    detected_bbox_norm: List[float],
    expected_roi: Dict[str, float],
    size_tolerance: float = 0.40,
) -> bool:
    """Check if detected element size is within expected range."""
    dw, dh = detected_bbox_norm[2], detected_bbox_norm[3]
    ew = expected_roi.get("w", 0.1)
    eh = expected_roi.get("h", 0.1)
    w_ratio = dw / (ew + 1e-6)
    h_ratio = dh / (eh + 1e-6)
    return (
        (1.0 - size_tolerance) <= w_ratio <= (1.0 + size_tolerance)
        and (1.0 - size_tolerance) <= h_ratio <= (1.0 + size_tolerance)
    )


def _position_score(
    detected_bbox_norm: List[float],
    expected_roi: Dict[str, float],
    tolerance: float,
) -> float:
    """Compute a position accuracy score 0-1."""
    dx, dy = detected_bbox_norm[0], detected_bbox_norm[1]
    ex, ey = expected_roi.get("x", 0), expected_roi.get("y", 0)
    dist = ((dx - ex) ** 2 + (dy - ey) ** 2) ** 0.5
    max_dist = tolerance * 2
    return max(0.0, 1.0 - dist / max(max_dist, 1e-6))


# ─────────────── Classifier Loading ───────────────

_classifier_cache: Dict[str, Any] = {}


def _load_classifier(doc_type: str, class_name: str):
    """Load a trained ElementClassifier for a specific element."""
    cache_key = f"{doc_type}_{class_name}"
    if cache_key in _classifier_cache:
        return _classifier_cache[cache_key]

    weight_path = WEIGHTS_DIR / f"{doc_type}_{class_name}.pt"
    if not weight_path.exists():
        logger.warning(f"No trained classifier for {doc_type}/{class_name}")
        _classifier_cache[cache_key] = None
        return None

    try:
        from ai.models.element_classifier import ElementClassifier
        classifier = ElementClassifier.load(weight_path)
        _classifier_cache[cache_key] = classifier
        return classifier
    except Exception as e:
        logger.error(f"Failed to load classifier {cache_key}: {e}")
        _classifier_cache[cache_key] = None
        return None


# ─────────────── Font Profile Loading ───────────────

_font_cache: Dict[str, Any] = {}


def _load_font_profile(doc_type: str, class_name: str):
    """Load a learned font profile for a text element."""
    cache_key = f"{doc_type}_{class_name}"
    if cache_key in _font_cache:
        return _font_cache[cache_key]

    profile_path = FONTS_DIR / f"{doc_type}_{class_name}.json"
    if not profile_path.exists():
        logger.warning(f"No font profile for {doc_type}/{class_name}")
        _font_cache[cache_key] = None
        return None

    try:
        from ai.models.font_analyzer import FontAnalyzer
        profile = FontAnalyzer.load_profile(profile_path)
        _font_cache[cache_key] = profile
        return profile
    except Exception as e:
        logger.error(f"Failed to load font profile {cache_key}: {e}")
        _font_cache[cache_key] = None
        return None


# ─────────────── Main Verification Pipeline ───────────────

def verify(image_path: str, doc_type: str) -> Dict[str, Any]:
    """
    Full document verification pipeline.

    Args:
        image_path: Path to the rectified document image
        doc_type: Document type folder name (e.g., 'identity', 'passport')

    Returns:
        Detailed verification result with per-element scores
    """
    start_time = time.time()

    # Load image
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "document_type": doc_type,
            "decision": "ERROR",
            "overall_confidence": 0.0,
            "elements": {},
            "missing_elements": [],
            "failed_elements": [],
            "anomalies": ["Could not load image"],
            "processing_time_ms": 0,
            "error": f"Could not load image: {image_path}",
            "element_results": {},
        }

    # Load layout config
    layout_config = _load_layout_config(doc_type)
    if layout_config is None:
        return {
            "document_type": doc_type,
            "decision": "ERROR",
            "overall_confidence": 0.0,
            "elements": {},
            "missing_elements": [],
            "failed_elements": [],
            "anomalies": [f"No layout_config.yaml for doc type: {doc_type}"],
            "processing_time_ms": 0,
            "error": f"No layout_config.yaml for {doc_type}",
            "element_results": {},
        }

    thresholds = layout_config.get("thresholds", {})
    pass_score = thresholds.get("pass_score", 0.95)
    suspicious_score = thresholds.get("suspicious_score", 0.70)

    # Get all elements from config
    all_elements = _get_all_elements(layout_config)

    # Step 1: Detect elements using layout positions
    detections = _detect_elements(image, all_elements)

    # Step 2: Process each detection
    elements_result: Dict[str, Dict[str, Any]] = {}
    anomalies: List[str] = []

    for det in detections:
        class_name = det["class_name"]
        elem_type = det["elem_type"]

        # Crop element from image
        element_crop = _crop_element(image, det["bbox"])
        if element_crop.size == 0:
            elements_result[class_name] = {
                "detected": False,
                "status": "MISSING",
                "details": f"{class_name}: empty crop region",
                "score": 0.0,
            }
            continue

        # Position validation
        pos_valid = _validate_position(
            det["bbox_norm"], det["expected_roi"], det["tolerance"]
        )
        size_valid = _validate_size(det["bbox_norm"], det["expected_roi"])
        pos_score = _position_score(
            det["bbox_norm"], det["expected_roi"], det["tolerance"]
        )

        # ─── Visual element: use trained binary classifier ───
        if elem_type == "visual":
            classifier = _load_classifier(doc_type, class_name)
            if classifier is not None:
                classifier_score = classifier.predict(element_crop)
                has_trained_model = True
            else:
                # No trained model — fallback with warning
                classifier_score = 0.5
                has_trained_model = False
                anomalies.append(f"{class_name}: no trained classifier (fallback)")

            # Combine: 70% classifier + 30% position for visual elements
            combined_score = classifier_score * 0.70 + pos_score * 0.30

            elements_result[class_name] = {
                "detected": True,
                "position": {
                    "x": det["bbox"][0],
                    "y": det["bbox"][1],
                    "w": det["bbox"][2],
                    "h": det["bbox"][3],
                },
                "detection_confidence": det["confidence"],
                "position_valid": pos_valid,
                "size_valid": size_valid,
                "position_score": round(pos_score, 4),
                "classifier_score": round(classifier_score, 4),
                "has_trained_model": has_trained_model,
                "score": round(combined_score, 4),
                "status": "PASSED" if combined_score >= pass_score else "FAILED",
                "details": (
                    f"{class_name}: classifier={classifier_score:.1%} pos={pos_score:.1%}"
                    if has_trained_model
                    else f"{class_name}: NO TRAINED MODEL (fallback={classifier_score:.1%})"
                ),
            }

        # ─── Text element: use font profile analysis ───
        elif elem_type == "text":
            font_profile = _load_font_profile(doc_type, class_name)
            if font_profile is not None:
                from ai.models.font_analyzer import FontAnalyzer
                analyzer = FontAnalyzer()
                font_score, font_details = analyzer.verify_font(element_crop, font_profile)
                has_font_profile = True
            else:
                font_score = 0.5
                font_details = {"warning": "no font profile"}
                has_font_profile = False
                anomalies.append(f"{class_name}: no font profile (fallback)")

            # For text: 60% font + 40% position
            combined_score = font_score * 0.60 + pos_score * 0.40

            elements_result[class_name] = {
                "detected": True,
                "position": {
                    "x": det["bbox"][0],
                    "y": det["bbox"][1],
                    "w": det["bbox"][2],
                    "h": det["bbox"][3],
                },
                "detection_confidence": det["confidence"],
                "position_valid": pos_valid,
                "size_valid": size_valid,
                "position_score": round(pos_score, 4),
                "font_score": round(font_score, 4),
                "font_details": font_details,
                "has_font_profile": has_font_profile,
                "score": round(combined_score, 4),
                "status": "PASSED" if combined_score >= pass_score else "FAILED",
                "details": (
                    f"{class_name}: font={font_score:.1%} pos={pos_score:.1%}"
                    if has_font_profile
                    else f"{class_name}: NO FONT PROFILE (fallback={font_score:.1%})"
                ),
            }

    # Step 3: Weighted overall confidence
    total_weight = 0.0
    weighted_score = 0.0
    for det in detections:
        class_name = det["class_name"]
        elem_data = elements_result.get(class_name, {})
        weight = det["weight"]
        score = elem_data.get("score", 0.0)
        if elem_data.get("status") == "MISSING":
            score = 0.0
        weighted_score += score * weight
        total_weight += weight

    overall_confidence = weighted_score / total_weight if total_weight > 0 else 0.0

    # Step 4: Determine missing/failed elements
    missing_elements = [
        cn for cn, data in elements_result.items()
        if data.get("status") == "MISSING"
    ]
    failed_elements = [
        cn for cn, data in elements_result.items()
        if data.get("status") in ("FAILED", "MISSING")
    ]

    if missing_elements:
        anomalies.append(f"Missing elements: {', '.join(missing_elements)}")

    # Step 5: Final decision — 85% to pass
    if overall_confidence >= pass_score and not missing_elements:
        decision = "PASSED"
    elif overall_confidence >= suspicious_score:
        decision = "SUSPICIOUS"
    else:
        decision = "FAILED"

    # Critical elements must be present and pass
    critical_failed = [
        cn for cn, data in elements_result.items()
        if any(
            d["class_name"] == cn and d.get("critical", False)
            for d in detections
        ) and data.get("status") != "PASSED"
    ]
    if critical_failed:
        decision = "FAILED"
        anomalies.append(f"Critical elements failed: {', '.join(critical_failed)}")

    processing_time = int((time.time() - start_time) * 1000)

    return {
        "document_type": doc_type,
        "decision": decision,
        "overall_confidence": round(overall_confidence, 4),
        "pass_threshold": pass_score,
        "elements": elements_result,
        "missing_elements": missing_elements,
        "failed_elements": failed_elements,
        "anomalies": anomalies,
        "processing_time_ms": processing_time,
        # Backward-compatible fields for pipeline integration
        "element_results": {
            name: {
                "status": data.get("status", "ERROR"),
                "score": data.get("score", 0.0),
                "threshold": pass_score,
                "message": data.get("details", ""),
            }
            for name, data in elements_result.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Watheq Document Verification (v3 — Trained Classifiers)",
    )
    parser.add_argument(
        "--image", "-i", type=str, required=True, help="Path to document image"
    )
    parser.add_argument(
        "--type", "-t", type=str, required=True, help="Document type folder name"
    )
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = verify(args.image, args.type)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        decision = result["decision"]
        confidence = result["overall_confidence"]
        threshold = result.get("pass_threshold", 0.85)
        print(f"\n{'='*60}")
        print(f"  Document Verification: {result.get('document_type', 'unknown')}")
        print(f"{'='*60}")
        print(f"  Decision: {decision}  |  Confidence: {confidence:.1%}  |  Threshold: {threshold:.0%}")
        print(f"  Processing time: {result['processing_time_ms']}ms")
        print(f"{'─'*60}")

        for name, data in result["elements"].items():
            status = data.get("status", "?")
            icon = "✓" if status == "PASSED" else "✗" if status == "FAILED" else "?"
            score = data.get("score", 0.0)
            det = "detected" if data.get("detected") else "MISSING"
            trained = "" if data.get("has_trained_model", data.get("has_font_profile", True)) else " [NO MODEL]"
            print(f"  {icon} {name:25s} {score:6.1%}  ({det}){trained}")
            if data.get("details"):
                print(f"    └ {data['details']}")

        if result["missing_elements"]:
            print(f"\n  Missing: {', '.join(result['missing_elements'])}")
        if result["anomalies"]:
            for a in result["anomalies"]:
                print(f"  ⚠ {a}")
        print()


if __name__ == "__main__":
    main()
