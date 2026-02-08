"""
YOLOv8-based Document Element Detector for Watheq.

كشف العناصر في الوثائق باستخدام YOLOv8:
- الشعار الرئيسي، الختم، الصورة الشخصية، الصورة الشفافة
- مناطق النص (الاسم، الرقم الوطني، تواريخ)
- الباركود، نمط الخلفية

Usage:
    detector = YOLODetector("ai/models/weights/yolo_identity.pt")
    results = detector.detect("document.jpg", conf_threshold=0.5)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Annotation class labels for document elements
ELEMENT_CLASSES = [
    "logo_main",  # 0  — الشعار الرئيسي
    "logo_secondary",  # 1  — الشعار في مواقع أخرى
    "stamp",  # 2  — الختم
    "photo_primary",  # 3  — صورة الشخص الرئيسية
    "photo_ghost",  # 4  — صورة الشخص الشفافة (watermark)
    "text_name",  # 5  — منطقة الاسم
    "text_national_id",  # 6  — منطقة الرقم الوطني
    "text_dob",  # 7  — تاريخ الميلاد
    "text_issue_date",  # 8  — تاريخ الإصدار
    "text_expiry_date",  # 9  — تاريخ الانتهاء
    "barcode",  # 10 — الباركود
    "background_pattern",  # 11 — نمط الخلفية
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(ELEMENT_CLASSES)}


class YOLODetector:
    """
    YOLOv8 Object Detection wrapper for document element detection.
    Loads a trained YOLO model and runs inference on document images.
    """

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        """
        Initialize the YOLO detector.

        Args:
            model_path: Path to the trained YOLOv8 .pt weights file
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLOv8 model from weights file."""
        if not self.model_path.exists():
            logger.warning(
                f"YOLO model not found at {self.model_path}. "
                "Using fallback template-based detection."
            )
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            self.model.to(self.device)
            logger.info(f"Loaded YOLO model from {self.model_path}")
        except ImportError:
            logger.warning("ultralytics not installed. Using fallback detection.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")

    def detect(
        self,
        image_path: str | Path,
        conf_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Detect document elements in an image.

        Args:
            image_path: Path to the document image
            conf_threshold: Minimum confidence threshold

        Returns:
            List of detections, each with:
                - class_name: element class name
                - class_id: element class index
                - confidence: detection confidence
                - bbox: [x, y, w, h] bounding box (pixel coords)
                - bbox_norm: [x, y, w, h] normalized to image size
        """
        if self.model is not None:
            return self._detect_yolo(image_path, conf_threshold)
        return self._detect_fallback(image_path)

    def _detect_yolo(
        self, image_path: str | Path, conf_threshold: float
    ) -> List[Dict[str, Any]]:
        """Run YOLO model inference."""
        results = self.model(str(image_path), conf=conf_threshold, verbose=False)
        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                img_h, img_w = r.orig_shape
                class_name = (
                    ELEMENT_CLASSES[cls_id]
                    if cls_id < len(ELEMENT_CLASSES)
                    else f"class_{cls_id}"
                )
                detections.append(
                    {
                        "class_name": class_name,
                        "class_id": cls_id,
                        "confidence": round(conf, 4),
                        "bbox": [round(x1), round(y1), round(w), round(h)],
                        "bbox_norm": [
                            round(x1 / img_w, 4),
                            round(y1 / img_h, 4),
                            round(w / img_w, 4),
                            round(h / img_h, 4),
                        ],
                    }
                )

        return detections

    def _detect_fallback(self, image_path: str | Path) -> List[Dict[str, Any]]:
        """
        Fallback template-based detection when YOLO model is unavailable.
        Reads ROI positions from layout_config.yaml for the document type.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return []

        h, w = image.shape[:2]

        # Try to load ROIs from layout_config.yaml
        rois = self._load_layout_rois()
        if not rois:
            # Ultimate fallback: hardcoded ROIs for Yemeni National ID
            rois = {
                "logo_main": {"x": 0.02, "y": 0.02, "w": 0.15, "h": 0.20},
                "photo_primary": {"x": 0.75, "y": 0.15, "w": 0.22, "h": 0.55},
                "stamp": {"x": 0.60, "y": 0.70, "w": 0.15, "h": 0.25},
                "text_name": {"x": 0.20, "y": 0.20, "w": 0.50, "h": 0.10},
                "text_national_id": {"x": 0.20, "y": 0.35, "w": 0.50, "h": 0.08},
                "barcode": {"x": 0.05, "y": 0.80, "w": 0.50, "h": 0.15},
            }

        detections = []
        for class_name, roi in rois.items():
            px = int(roi["x"] * w)
            py = int(roi["y"] * h)
            pw = int(roi["w"] * w)
            ph = int(roi["h"] * h)
            detections.append(
                {
                    "class_name": class_name,
                    "class_id": CLASS_TO_IDX.get(class_name, -1),
                    "confidence": 0.85,  # Layout-config confidence
                    "bbox": [px, py, pw, ph],
                    "bbox_norm": [
                        round(roi["x"], 4),
                        round(roi["y"], 4),
                        round(roi["w"], 4),
                        round(roi["h"], 4),
                    ],
                }
            )

        return detections

    def _load_layout_rois(self) -> Dict[str, Dict[str, float]]:
        """
        Load element ROIs from layout_config.yaml files.
        Scans all doc type folders for configs and merges their element ROIs.
        """
        ai_dir = Path(__file__).resolve().parents[1]
        refs_dir = ai_dir / "data" / "refrences"
        if not refs_dir.exists():
            return {}

        rois = {}
        try:
            import yaml
            for doc_dir in refs_dir.iterdir():
                if not doc_dir.is_dir():
                    continue
                config_path = doc_dir / "layout_config.yaml"
                if not config_path.exists():
                    continue
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                # Visual elements
                for ref_stem, elem_data in config.get("elements", {}).items():
                    class_name = elem_data.get("class_name", ref_stem)
                    roi = elem_data.get("roi", {})
                    if roi:
                        rois[class_name] = roi
                # Text regions
                for text_name, text_data in config.get("text_regions", {}).items():
                    class_name = text_data.get("class_name", text_name)
                    roi = text_data.get("roi", {})
                    if roi:
                        rois[class_name] = roi
        except Exception as e:
            logger.warning(f"Failed to load layout configs: {e}")

        return rois


def crop_element(
    image: np.ndarray, bbox: List[int], padding: float = 0.05
) -> np.ndarray:
    """
    Crop an element from an image using bounding box.

    Args:
        image: Source image (BGR)
        bbox: [x, y, w, h] pixel coordinates
        padding: Fractional padding around the crop

    Returns:
        Cropped element image
    """
    h, w = image.shape[:2]
    x, y, bw, bh = bbox
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + bw + pad_x)
    y1 = min(h, y + bh + pad_y)

    return image[y0:y1, x0:x1].copy()
