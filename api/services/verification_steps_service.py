"""Verification Step Implementations — تنفيذ مراحل التحقق

Contains the concrete functions called by the orchestrator for each
pipeline stage: image quality checks, document cropping, face extraction,
face matching, OCR, AI verification, data verification against citizen
records, and blockchain recording via MultiChain + IPFS.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ledger.ipfs_service import IPFSService
from Biometric.face_service import FaceService
from ocr.vision_service_ocr import ocr_image, ocr_pdf
from api.services.multichain_service import publish_to_stream, json_to_hex

QUALITY_BRIGHTNESS_MIN = 40
QUALITY_BRIGHTNESS_MAX = 220
QUALITY_BLUR_MIN = 70.0
QUALITY_MIN_AREA_RATIO = 0.15
QUALITY_MAX_AREA_RATIO = 0.95
QUALITY_MIN_ASPECT = 1.2
QUALITY_MAX_ASPECT = 2.1


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def document_image_quality(document_front: Path) -> dict[str, Any]:
    image = cv2.imdecode(np.fromfile(document_front, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Invalid document image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    ok_brightness = QUALITY_BRIGHTNESS_MIN <= brightness <= QUALITY_BRIGHTNESS_MAX
    ok_blur = blur_score >= QUALITY_BLUR_MIN
    reason_code = None
    if brightness < QUALITY_BRIGHTNESS_MIN:
        reason_code = "LOW_BRIGHTNESS"
    elif brightness > QUALITY_BRIGHTNESS_MAX:
        reason_code = "HIGH_BRIGHTNESS"
    elif blur_score < QUALITY_BLUR_MIN:
        reason_code = "BLURRY"

    return {
        "brightness": brightness,
        "blur_score": blur_score,
        "brightness_ok": ok_brightness,
        "blur_ok": ok_blur,
        "reason_code": reason_code,
        "message": None,
    }


def document_crop(document_front: Path, output_path: Path) -> dict[str, Any]:
    """Detect and perspective-rectify the document card using OpenCV contour detection."""
    image = cv2.imdecode(np.fromfile(document_front, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Invalid document image")

    h, w = image.shape[:2]
    print(f"IN: {w}x{h}")

    # Pre-process for contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.dilate(edged, kernel, iterations=2)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    min_area = h * w * QUALITY_MIN_AREA_RATIO
    max_area = h * w * QUALITY_MAX_AREA_RATIO

    # Sort contours by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            best_quad = approx
            break

    if best_quad is None:
        # Fallback: use the largest contour's bounding rect as a quad
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            best_quad = np.int32(box).reshape(4, 1, 2)
            break

    if best_quad is None:
        # Last fallback: crop a central region
        margin_x, margin_y = int(w * 0.05), int(h * 0.05)
        cropped = image[margin_y : h - margin_y, margin_x : w - margin_x]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".jpg", cropped)[1].tofile(str(output_path))
        print(f"DOCUMENT_CROPPING: PASS method=center_crop")
        return {
            "cropped_path": str(output_path),
            "rectified_path": str(output_path),
            "method": "center_crop",
        }

    # Order points: top-left, top-right, bottom-right, bottom-left
    pts = best_quad.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    ordered = np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmin(d)],
            pts[np.argmax(s)],
            pts[np.argmax(d)],
        ],
        dtype=np.float32,
    )

    # Compute target width/height
    wA = np.linalg.norm(ordered[2] - ordered[3])
    wB = np.linalg.norm(ordered[1] - ordered[0])
    target_w = int(max(wA, wB))
    hA = np.linalg.norm(ordered[1] - ordered[2])
    hB = np.linalg.norm(ordered[0] - ordered[3])
    target_h = int(max(hA, hB))

    # Ensure reasonable aspect ratio for an ID card (~1.58)
    if target_w < target_h:
        target_w, target_h = target_h, target_w

    dst = np.array(
        [
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(ordered, dst)
    rectified = cv2.warpPerspective(image, M, (target_w, target_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", rectified)[1].tofile(str(output_path))

    print(f"RECT: {target_w}x{target_h}")
    print(f"DOCUMENT_CROPPING: PASS method=rectify_warp size={target_w}x{target_h}")
    return {
        "cropped_path": str(output_path),
        "rectified_path": str(output_path),
        "method": "rectify_warp",
    }


def document_face_extraction(cropped_path: Path, output_path: Path) -> dict[str, Any]:
    """Extract the face from the rectified document using face detection."""
    image = cv2.imdecode(np.fromfile(cropped_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Invalid cropped document image")

    h, w = image.shape[:2]

    # Try layout report + template ROI first (preferred)
    repo_root = Path(__file__).resolve().parents[2]
    layout_report = cropped_path.parent / "layout" / "report.json"
    template_candidates = [
        repo_root
        / "ai"
        / "card_verification"
        / "registry"
        / "templates"
        / "national_id_yemen_v1"
        / "layout.yaml",
    ]
    roi = None
    for tpl_path in template_candidates:
        if not tpl_path.exists() or not layout_report.exists():
            continue
        try:
            report = json.loads(layout_report.read_text(encoding="utf-8"))
            if (report.get("layout_status") or "").upper() != "PASS":
                continue
            import yaml

            with open(tpl_path, "r", encoding="utf-8") as f:
                template = yaml.safe_load(f)
            roi = (template.get("elements") or {}).get("photo", {}).get("roi")
        except Exception:
            pass

    if roi:
        x0 = max(0, min(int(round(roi["x"] * w)), w))
        y0 = max(0, min(int(round(roi["y"] * h)), h))
        x1 = max(0, min(int(round((roi["x"] + roi["w"]) * w)), w))
        y1 = max(0, min(int(round((roi["y"] + roi["h"]) * h)), h))
        if x1 > x0 and y1 > y0:
            face = image[y0:y1, x0:x1].copy()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), face)
            return {"document_face_path": str(output_path), "source": "layout_roi"}

    # Fallback: use OpenCV Haar cascade face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        raise RuntimeError("No face detected in document")

    # Pick the largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    fx, fy, fw, fh = faces[0]

    # Add margin
    margin = int(max(fw, fh) * 0.15)
    fx = max(0, fx - margin)
    fy = max(0, fy - margin)
    fw = min(w - fx, fw + 2 * margin)
    fh = min(h - fy, fh + 2 * margin)

    face = image[fy : fy + fh, fx : fx + fw].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), face)

    return {
        "document_face_path": str(output_path),
        "source": "haar_cascade",
        "bbox": {"x": int(fx), "y": int(fy), "w": int(fw), "h": int(fh)},
    }


def face_matching(document_face_path: Path, person_image: Path) -> dict[str, Any]:
    if not document_face_path.exists():
        raise RuntimeError(
            "Document face image not found — face extraction may have failed"
        )
    if not person_image.exists():
        raise RuntimeError("Person/selfie image not found")

    service = FaceService()
    # Both images go through RetinaFace detection + alignment inside compare_faces,
    # ensuring consistent face embeddings regardless of crop quality.
    result = service.verify_id_vs_live(
        document_face_path.read_bytes(), person_image.read_bytes()
    )
    return result


def layout_gating_verify(rectified_image: Path) -> dict[str, Any]:
    """Inline layout verification: checks aspect ratio, edge density, and face presence."""
    image = cv2.imdecode(np.fromfile(rectified_image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Invalid rectified image")

    h, w = image.shape[:2]
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    # 1. Aspect ratio check (ID cards are typically ~1.4–1.7 landscape)
    aspect = w / h if h > 0 else 0
    checks["aspect_ratio"] = QUALITY_MIN_ASPECT <= aspect <= QUALITY_MAX_ASPECT
    if not checks["aspect_ratio"]:
        reasons.append(f"ASPECT_RATIO_OUT_OF_RANGE ({aspect:.2f})")

    # 2. Edge density — a real document should have meaningful edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / (h * w) if (h * w) > 0 else 0
    checks["edge_density"] = edge_density > 0.02
    if not checks["edge_density"]:
        reasons.append(f"LOW_EDGE_DENSITY ({edge_density:.4f})")

    # 3. Face presence — at least one face should be detectable
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
    )
    checks["face_detected"] = len(faces) > 0
    if not checks["face_detected"]:
        reasons.append("NO_FACE_DETECTED")

    # 4. Minimum resolution
    checks["min_resolution"] = w >= 200 and h >= 120
    if not checks["min_resolution"]:
        reasons.append(f"LOW_RESOLUTION ({w}x{h})")

    passed = all(checks.values())
    layout_status = "PASS" if passed else "FAIL"
    reason_str = "; ".join(reasons) if reasons else None

    # Write report for downstream steps
    out_dir = rectified_image.parent / "layout"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "layout_status": layout_status,
        "reason": reason_str,
        "checks": checks,
        "metrics": {
            "aspect_ratio": round(aspect, 3),
            "edge_density": round(edge_density, 4),
            "faces_found": len(faces),
            "width": w,
            "height": h,
        },
    }
    report_path = out_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"[LAYOUT] status={layout_status} aspect={aspect:.2f} edges={edge_density:.4f} faces={len(faces)}"
    )

    return {
        "layout_status": layout_status,
        "reason": reason_str,
        "artifacts": {
            "report_json": str(report_path),
        },
    }


def ml_verify(
    document_front: Path, doc_type_folder: str = "identity"
) -> dict[str, Any]:
    """
    Run AI verification using the new dynamic verify_document.py script.

    Args:
        document_front: Path to the document image
        doc_type_folder: The folder_name from document_types table (e.g., 'identity', 'passport')

    Returns:
        Verification result with decision, failed elements, and per-element scores
    """
    repo_root = Path(__file__).resolve().parents[2]
    verify_script = repo_root / "ai" / "verify_document.py"

    if not verify_script.exists():
        raise RuntimeError("ai/verify_document.py not found")

    import subprocess
    import sys

    cmd = [
        sys.executable,
        str(verify_script),
        "--image",
        str(document_front),
        "--type",
        doc_type_folder,
        "--json",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"AI verification failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )

    result = json.loads(proc.stdout)

    # Calculate authenticity percent from element scores
    element_results = result.get("element_results", {})
    scores = [
        r.get("score", 0)
        for r in element_results.values()
        if r.get("status") != "ERROR"
    ]
    authenticity_percent = (sum(scores) / len(scores) * 100) if scores else None

    return {
        "final_decision": result.get("decision"),
        "authenticity_percent": authenticity_percent,
        "failed_elements": result.get("failed_elements", []),
        "element_results": element_results,
    }


def ocr_verify(document_front: Path, max_pages: int = 10) -> dict[str, Any]:
    suffix = document_front.suffix.lower()
    data = document_front.read_bytes()
    if suffix == ".pdf":
        return ocr_pdf(data, max_pages=max_pages)
    return ocr_image(data)


# ---------------------------------------------------------------------------
# DATA_VERIFICATION — cross-check OCR fields against citizen_records DB
# ---------------------------------------------------------------------------
import re


def _parse_ocr_fields(ocr_result: dict[str, Any]) -> dict[str, str]:
    """استخراج الحقول المهيكلة من نتيجة OCR."""
    fields: dict[str, str] = {}
    text = ocr_result.get("text", "") or ""

    # Try to extract national ID (Yemeni format: digits, typically 8-12)

    id_match = re.search(r"\b(\d{8,12})\b", text)
    if id_match:
        fields["national_id"] = id_match.group(1)

    # Try to extract name (Arabic name pattern)
    arabic_name = re.search(r"[\u0600-\u06FF][\u0600-\u06FF\s]{4,}", text)
    if arabic_name:
        fields["full_name_ar"] = arabic_name.group(0).strip()

    # Date pattern (dd/mm/yyyy or dd-mm-yyyy)
    dates = re.findall(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", text)
    if dates:
        fields["date_of_birth"] = dates[0]
        if len(dates) > 1:
            fields["issue_date"] = dates[1]
        if len(dates) > 2:
            fields["expiry_date"] = dates[2]

    return fields


def _build_citizen_insert_data(fields: dict[str, str]) -> dict[str, Any]:
    """Build a complete citizen record dict from parsed OCR fields, defaulting missing fields to None."""
    all_columns = [
        "national_id",
        "full_name_ar",
        "full_name_en",
        "date_of_birth",
        "address",
        "issue_date",
        "expiry_date",
        "gender",
        "nationality",
        "document_type",
    ]
    return {col: fields.get(col) for col in all_columns}


async def data_verification(
    *,
    ocr_result: dict[str, Any],
    document_type_id: int,
) -> dict[str, Any]:
    """التحقق من البيانات — مقارنة حقول OCR مع سجلات المواطنين في قاعدة البيانات.

    Logic:
      - If national_id cannot be extracted from OCR → raise (pipeline fails).
      - If citizen NOT found → store extracted data as new record, pass.
      - If citizen found and fields match → pass.
      - If citizen found but fields mismatch → raise as fraud (محاولة احتيال).
    """
    from api.database import get_citizen_records_collection

    fields = _parse_ocr_fields(ocr_result)
    national_id = fields.get("national_id")

    if not national_id:
        raise RuntimeError(
            "National_id not extracted from OCR — cannot verify citizen data"
        )

    citizens_col = get_citizen_records_collection()
    citizen = await citizens_col.get_by_national_id(national_id)

    # ── Case C: citizen does not exist → store new record, PASS ──
    if citizen is None:
        insert_data = _build_citizen_insert_data(fields)
        await citizens_col.create(insert_data)
        return {
            "citizen_found": False,
            "new_record_created": True,
            "data_match": True,
            "fraud_suspected": False,
            "national_id": national_id,
            "parsed_fields": fields,
            "match_details": {},
            "match_count": 0,
            "total_compared": 0,
            "message": "سجل مواطن جديد — تم حفظ البيانات المستخرجة لأول مرة",
        }

    # ── Citizen exists → compare fields ──
    match_details: dict[str, Any] = {}

    if "full_name_ar" in fields and citizen.get("full_name_ar"):
        ocr_name = fields["full_name_ar"].replace(" ", "")
        db_name = citizen["full_name_ar"].replace(" ", "")
        name_match = ocr_name in db_name or db_name in ocr_name
        match_details["full_name_ar"] = {
            "ocr": fields["full_name_ar"],
            "db": citizen["full_name_ar"],
            "match": name_match,
        }

    if "date_of_birth" in fields and citizen.get("date_of_birth"):
        db_dob = str(citizen["date_of_birth"])
        dob_match = fields["date_of_birth"].replace("-", "/") in db_dob.replace(
            "-", "/"
        )
        match_details["date_of_birth"] = {
            "ocr": fields["date_of_birth"],
            "db": db_dob,
            "match": dob_match,
        }

    match_count = sum(1 for v in match_details.values() if v.get("match"))
    total_compared = len(match_details)
    all_matched = total_compared > 0 and match_count == total_compared

    # ── Case B: citizen exists but data mismatches → FRAUD ──
    if total_compared > 0 and not all_matched:
        raise RuntimeError(
            f"Fraud suspected — data mismatch: "
            f"{match_count}/{total_compared} fields matched for national_id {national_id}"
        )

    # ── Case A: citizen exists and data matches → PASS ──
    return {
        "citizen_found": True,
        "new_record_created": False,
        "data_match": True,
        "fraud_suspected": False,
        "national_id": national_id,
        "parsed_fields": fields,
        "match_details": match_details,
        "match_count": match_count,
        "total_compared": total_compared,
        "message": f"تم مطابقة {match_count}/{total_compared} حقول مع سجل المواطن بنجاح",
    }


def blockchain_verify(
    document_front: Path,
    *,
    document_type_id: int,
    owner: str,
) -> dict[str, Any]:
    """سجل الوثيقة على IPFS و MultiChain."""
    import time as _time

    sha = _sha256_file(document_front)

    ipfs = IPFSService()
    cid = ipfs.pin_file(str(document_front))

    timestamp = int(_time.time())
    doc_id = f"DOC-{timestamp}-{document_type_id}"

    # Complete metadata for on-chain record
    metadata = {
        "doc_id": doc_id,
        "cid": cid,
        "filename": document_front.name,
        "owner": owner,
        "sha256": sha,
        "document_type_id": document_type_id,
        "timestamp": timestamp,
        "source": "verification_pipeline",
    }
    data_hex = json_to_hex(json.dumps(metadata))
    publish_to_stream(doc_id, data_hex)

    return {
        "doc_id": doc_id,
        "cid": cid,
        "sha256": sha,
        "filename": document_front.name,
        "timestamp": timestamp,
        "ledger_recorded": True,
    }


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
