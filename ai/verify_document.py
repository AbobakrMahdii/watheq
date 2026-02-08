#!/usr/bin/env python3
"""
Watheq Document Verification Script (v2 — YOLOv8 + Siamese Network)

التحقق من أصالة الوثائق باستخدام:
1. YOLOv8 لكشف العناصر (الشعار، الختم، الصورة، الباركود...)
2. شبكة سيامية للتحقق من أصالة كل عنصر
3. تحليل الألوان والمواقع والأحجام
4. قرار نهائي مرجح

Usage:
    python ai/verify_document.py --image doc.jpg --type identity --json
    python ai/verify_document.py --image doc.jpg --type passport

Returns detailed JSON:
{
    "decision": "PASSED|FAILED",
    "overall_confidence": 0.94,
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

AI_DIR = Path(__file__).parent.resolve()
MODELS_DIR = AI_DIR / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
REFERENCES_DIR = AI_DIR / "data" / "refrences"
TRAINING_DIR = AI_DIR / "data" / "training"

# Ensure ai/ directory is importable
if str(AI_DIR.parent) not in sys.path:
    sys.path.insert(0, str(AI_DIR.parent))

# Expected elements per document type (position tolerances in normalized coords)
EXPECTED_LAYOUTS: Dict[str, Dict[str, Dict[str, float]]] = {
    "identity": {
        "logo_main": {"x": 0.02, "y": 0.02, "w": 0.15, "h": 0.20, "tolerance": 0.10},
        "photo_primary": {
            "x": 0.75,
            "y": 0.15,
            "w": 0.22,
            "h": 0.55,
            "tolerance": 0.10,
        },
        "stamp": {"x": 0.60, "y": 0.70, "w": 0.15, "h": 0.25, "tolerance": 0.15},
        "text_name": {"x": 0.20, "y": 0.20, "w": 0.50, "h": 0.10, "tolerance": 0.15},
        "text_national_id": {
            "x": 0.20,
            "y": 0.35,
            "w": 0.50,
            "h": 0.08,
            "tolerance": 0.15,
        },
        "barcode": {"x": 0.05, "y": 0.80, "w": 0.50, "h": 0.15, "tolerance": 0.15},
    },
    "passport": {
        "logo_main": {"x": 0.35, "y": 0.02, "w": 0.30, "h": 0.15, "tolerance": 0.10},
        "photo_primary": {
            "x": 0.05,
            "y": 0.25,
            "w": 0.30,
            "h": 0.45,
            "tolerance": 0.10,
        },
        "stamp": {"x": 0.65, "y": 0.60, "w": 0.20, "h": 0.20, "tolerance": 0.15},
        "barcode": {"x": 0.05, "y": 0.85, "w": 0.90, "h": 0.12, "tolerance": 0.10},
    },
}

# Element weights for final score (higher = more important for authenticity)
ELEMENT_WEIGHTS = {
    "logo_main": 1.5,
    "logo_secondary": 0.8,
    "stamp": 1.5,
    "photo_primary": 1.0,
    "photo_ghost": 0.7,
    "text_name": 1.0,
    "text_national_id": 1.2,
    "text_dob": 0.8,
    "text_issue_date": 0.8,
    "text_expiry_date": 0.8,
    "barcode": 1.3,
    "background_pattern": 0.5,
}


def _load_detector(doc_type: str):
    """Load or create a YOLODetector for the given document type."""
    from ai.models.yolo_detector import YOLODetector

    model_path = WEIGHTS_DIR / f"yolo_{doc_type}.pt"
    if not model_path.exists():
        model_path = WEIGHTS_DIR / "yolo_document.pt"
    return YOLODetector(model_path)


def _load_verifier(doc_type: str):
    """Load or create a SiameseVerifier for the given document type."""
    from ai.models.siamese_verifier import SiameseVerifier

    model_path = WEIGHTS_DIR / f"siamese_{doc_type}.pt"
    if not model_path.exists():
        model_path = WEIGHTS_DIR / "siamese_document.pt"
    return SiameseVerifier(
        model_path=model_path if model_path.exists() else None,
        embeddings_dir=EMBEDDINGS_DIR,
    )


def _validate_position(
    detected_bbox_norm: List[float],
    expected: Dict[str, float],
) -> bool:
    """Check if detected position is within expected tolerance."""
    dx, dy, dw, dh = detected_bbox_norm
    ex, ey = expected["x"], expected["y"]
    tol = expected.get("tolerance", 0.10)

    return abs(dx - ex) <= tol and abs(dy - ey) <= tol


def _validate_size(
    detected_bbox_norm: List[float],
    expected: Dict[str, float],
    size_tolerance: float = 0.50,
) -> bool:
    """Check if detected element size is within expected range."""
    _, _, dw, dh = detected_bbox_norm
    ew, eh = expected["w"], expected["h"]

    w_ratio = dw / (ew + 1e-6)
    h_ratio = dh / (eh + 1e-6)

    return (1.0 - size_tolerance) <= w_ratio <= (1.0 + size_tolerance) and (
        1.0 - size_tolerance
    ) <= h_ratio <= (1.0 + size_tolerance)


def _check_ghost_image(image: np.ndarray, ghost_bbox: List[int]) -> Dict[str, Any]:
    """Check for ghost/watermark image presence and opacity."""
    x, y, w, h = ghost_bbox
    ih, iw = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(iw, x + w), min(ih, y + h)

    if x1 <= x0 or y1 <= y0:
        return {"detected": False, "opacity": 0.0}

    region = image[y0:y1, x0:x1]
    gray = (
        cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    )
    std_dev = float(np.std(gray))

    if 10 < std_dev < 50:
        opacity = min(1.0, std_dev / 50.0)
        return {
            "detected": True,
            "opacity": round(opacity, 2),
            "std_dev": round(std_dev, 2),
        }
    return {"detected": False, "opacity": 0.0, "std_dev": round(std_dev, 2)}


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

    expected_layout = EXPECTED_LAYOUTS.get(doc_type, {})

    # Step 1: Detect elements with YOLOv8
    detector = _load_detector(doc_type)
    detections = detector.detect(image_path, conf_threshold=0.4)

    # Step 2: Load Siamese verifier
    verifier = _load_verifier(doc_type)

    # Step 3: Process each detection
    from ai.models.yolo_detector import crop_element

    elements_result: Dict[str, Dict[str, Any]] = {}
    detected_classes: set = set()

    for det in detections:
        class_name = det["class_name"]
        detected_classes.add(class_name)

        # Crop element from image
        element_crop = crop_element(image, det["bbox"])
        if element_crop.size == 0:
            continue

        # Position validation
        expected = expected_layout.get(class_name)
        position_valid = (
            _validate_position(det["bbox_norm"], expected) if expected else True
        )
        size_valid = _validate_size(det["bbox_norm"], expected) if expected else True

        # Siamese verification (authenticity check)
        siamese_result = verifier.verify_element(element_crop, class_name, doc_type)

        # Color analysis (for logos and stamps)
        color_result = {"color_match": 1.0, "skipped": True}
        if class_name in ("logo_main", "logo_secondary", "stamp"):
            color_result = verifier.verify_color(element_crop)

        # Ghost image check
        ghost_result = None
        if class_name == "photo_ghost":
            ghost_result = _check_ghost_image(image, det["bbox"])

        # Build element result
        elem_data: Dict[str, Any] = {
            "detected": True,
            "position": {
                "x": det["bbox"][0],
                "y": det["bbox"][1],
                "w": det["bbox"][2],
                "h": det["bbox"][3],
            },
            "detection_confidence": det["confidence"],
            "position_valid": position_valid,
            "size_valid": size_valid,
            "authenticity_score": siamese_result["authenticity_score"],
            "color_match": color_result.get("color_match", 1.0),
        }

        if ghost_result:
            elem_data["opacity_detected"] = ghost_result.get("opacity", 0.0)
            elem_data["ghost_present"] = ghost_result.get("detected", False)

        # Overall element status
        element_passed = (
            siamese_result["passed"]
            and position_valid
            and size_valid
            and color_result.get("color_match", 1.0) >= 0.5
        )
        elem_data["status"] = "PASSED" if element_passed else "FAILED"

        # Detail message
        issues = []
        if not siamese_result["passed"]:
            issues.append(f"authenticity {siamese_result['authenticity_score']:.0%}")
        if not position_valid:
            issues.append("wrong position")
        if not size_valid:
            issues.append("wrong size")
        if color_result.get("color_match", 1.0) < 0.5:
            issues.append("color mismatch")

        elem_data["details"] = (
            f"{class_name}: " + ", ".join(issues)
            if issues
            else f"{class_name} verified"
        )
        elem_data["score"] = siamese_result["authenticity_score"]

        elements_result[class_name] = elem_data

    # Step 4: Check for missing expected elements
    missing_elements = []
    for expected_name in expected_layout:
        if expected_name not in detected_classes:
            missing_elements.append(expected_name)
            elements_result[expected_name] = {
                "detected": False,
                "status": "MISSING",
                "details": f"{expected_name} not detected in document",
                "score": 0.0,
            }

    # Step 5: Anomaly detection
    anomalies: List[str] = []
    if missing_elements:
        anomalies.append(f"Missing elements: {', '.join(missing_elements)}")

    # Step 6: Weighted overall confidence
    total_weight = 0.0
    weighted_score = 0.0
    for elem_name, elem_data in elements_result.items():
        weight = ELEMENT_WEIGHTS.get(elem_name, 1.0)
        score = elem_data.get("score", 0.0)
        if elem_data.get("status") == "MISSING":
            score = 0.0
        weighted_score += score * weight
        total_weight += weight

    overall_confidence = weighted_score / total_weight if total_weight > 0 else 0.0

    # Step 7: Final decision
    failed_elements = [
        name
        for name, data in elements_result.items()
        if data.get("status") in ("FAILED", "MISSING")
    ]

    if overall_confidence >= 0.75 and not missing_elements:
        decision = "PASSED"
    elif overall_confidence >= 0.50:
        decision = "SUSPICIOUS"
    else:
        decision = "FAILED"

    # Critical elements must be present
    critical_missing = [
        m for m in missing_elements if m in ("logo_main", "stamp", "photo_primary")
    ]
    if critical_missing:
        decision = "FAILED"

    processing_time = int((time.time() - start_time) * 1000)

    return {
        "document_type": doc_type,
        "decision": decision,
        "overall_confidence": round(overall_confidence, 4),
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
                "threshold": 0.50,
                "message": data.get("details", ""),
            }
            for name, data in elements_result.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Watheq Document Verification (v2 — YOLOv8 + Siamese)",
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
        print(f"\n{'='*60}")
        print(f"  Document Verification: {result.get('document_type', 'unknown')}")
        print(f"{'='*60}")
        print(f"  Decision: {decision}  |  Confidence: {confidence:.1%}")
        print(f"  Processing time: {result['processing_time_ms']}ms")
        print(f"{'─'*60}")

        for name, data in result["elements"].items():
            status = data.get("status", "?")
            icon = "✓" if status == "PASSED" else "✗" if status == "FAILED" else "?"
            score = data.get("score", 0.0)
            det = "detected" if data.get("detected") else "MISSING"
            print(f"  {icon} {name:25s} {score:6.1%}  ({det})")
            if data.get("details"):
                print(f"    └ {data['details']}")

        if result["missing_elements"]:
            print(f"\n  Missing: {', '.join(result['missing_elements'])}")
        if result["anomalies"]:
            print(f"  Anomalies: {'; '.join(result['anomalies'])}")
        print()


if __name__ == "__main__":
    main()
