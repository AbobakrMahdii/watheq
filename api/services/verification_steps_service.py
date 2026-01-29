from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ledger.ipfs_service import IPFSService
from ai.Biometric.face_service import FaceService
from ai.ocr.vision_service_ocr import ocr_image, ocr_pdf
from api.services.fabric_service import fabric_invoke

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


def _compute_percent(report: dict[str, Any]) -> Optional[float]:
    try:
        verification_results = report.get("verification_results") or {}
        ssim = float(verification_results.get("ssim_score", 0.0))
        orb = float(verification_results.get("orb_match_ratio", 0.0))
        resnet_conf = float(verification_results.get("resnet_confidence", 0.0))

        base = max(0.0, min(1.0, (ssim + orb + resnet_conf) / 3.0)) * 100.0
        decision = (report.get("decision") or "").upper()
        if decision == "FORGED":
            return float(min(base, 30.0))
        if decision == "SUSPICIOUS":
            return float(min(max(base, 40.0), 75.0))
        if decision == "AUTHENTIC":
            return float(min(max(base, 80.0), 99.0))
        return float(base)
    except Exception:
        return None


def biometric_verify(document_front: Path, person_image: Path) -> dict[str, Any]:
    service = FaceService()
    result = service.verify_faces(document_front.read_bytes(), person_image.read_bytes())
    return result


def document_image_quality(document_front: Path) -> dict[str, Any]:
    image = cv2.imdecode(np.fromfile(document_front, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Invalid document image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    ok_brightness = QUALITY_BRIGHTNESS_MIN <= brightness <= QUALITY_BRIGHTNESS_MAX
    ok_blur = blur_score >= QUALITY_BLUR_MIN

    return {
        "brightness": brightness,
        "blur_score": blur_score,
        "brightness_ok": ok_brightness,
        "blur_ok": ok_blur,
        "message": None,
    }


def document_crop(document_front: Path, output_path: Path) -> dict[str, Any]:
    image = cv2.imdecode(np.fromfile(document_front, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Invalid document image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Document edges not found")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    doc_contour = None
    img_area = float(image.shape[0] * image.shape[1])

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area_ratio = cv2.contourArea(approx) / img_area
            if QUALITY_MIN_AREA_RATIO <= area_ratio <= QUALITY_MAX_AREA_RATIO:
                doc_contour = approx
                break

    if doc_contour is None:
        raise RuntimeError("Document contour not detected")

    pts = doc_contour.reshape(4, 2).astype("float32")
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), warped)

    aspect_ratio = maxWidth / maxHeight if maxHeight else 0
    return {
        "cropped_path": str(output_path),
        "aspect_ratio": aspect_ratio,
        "width": maxWidth,
        "height": maxHeight,
    }


def document_face_extraction(cropped_path: Path, output_path: Path) -> dict[str, Any]:
    image = cv2.imdecode(np.fromfile(cropped_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Invalid cropped document image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if faces is None or len(faces) == 0:
        raise RuntimeError("No face detected in document")

    x, y, w, h = faces[0]
    face = image[y : y + h, x : x + w]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), face)

    return {
        "document_face_path": str(output_path),
        "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
    }


def selfie_liveness_check(liveness_payload: Optional[dict[str, Any]]) -> dict[str, Any]:
    if liveness_payload is None:
        return {
            "passed": True,
            "skipped": True,
            "message": "Liveness data not provided",
        }

    passed = bool(liveness_payload.get("passed", False))
    if not passed:
        raise RuntimeError(liveness_payload.get("message") or "Liveness check failed")

    return {
        "passed": True,
        "skipped": False,
        "details": liveness_payload,
    }


def face_matching(document_face_path: Path, person_image: Path) -> dict[str, Any]:
    service = FaceService()
    result = service.verify_faces(document_face_path.read_bytes(), person_image.read_bytes())
    return result


def ml_verify(document_front: Path) -> dict[str, Any]:
    # Reuse Logo/Stamp verification pipeline by calling the unified script.
    repo_root = Path(__file__).resolve().parents[2]
    module_dir = repo_root / "ai" / "LogoAndStamp"
    if not module_dir.exists():
        raise RuntimeError("ai/LogoAndStamp not found")

    config_path = module_dir / "config.yaml"
    if not config_path.exists():
        raise RuntimeError("ai/LogoAndStamp/config.yaml not found")

    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory(prefix="watheq_doc_verify_") as tmpdir:
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(module_dir / "main_unified.py"),
            "--input",
            str(document_front),
            "--config",
            str(config_path),
            "--output",
            str(out_dir),
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(module_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"LogoAndStamp failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
            )

        report_path = out_dir / f"{document_front.stem}_unified_report.json"
        if not report_path.exists():
            raise RuntimeError("Unified report not generated")

        unified = json.loads(report_path.read_text(encoding="utf-8"))
        logo_report = (unified.get("verifications") or {}).get("logo") or {}
        stamp_report = (unified.get("verifications") or {}).get("stamp") or {}

        logo_percent = _compute_percent(logo_report)
        stamp_percent = _compute_percent(stamp_report)
        available = [p for p in [logo_percent, stamp_percent] if p is not None]
        authenticity_percent = float(sum(available) / len(available)) if available else None

        return {
            "final_decision": unified.get("final_decision"),
            "authenticity_percent": authenticity_percent,
            "logo": logo_report,
            "stamp": stamp_report,
        }


def ocr_verify(document_front: Path, max_pages: int = 10) -> dict[str, Any]:
    suffix = document_front.suffix.lower()
    data = document_front.read_bytes()
    if suffix == ".pdf":
        return ocr_pdf(data, max_pages=max_pages)
    return ocr_image(data)


def blockchain_verify(
    document_front: Path,
    *,
    document_type_id: int,
    owner: str,
) -> dict[str, Any]:
    sha = _sha256_file(document_front)

    ipfs = IPFSService()
    cid = ipfs.pin_file(str(document_front))

    doc_id = f"DOC-{int(Path(document_front).stat().st_mtime)}-{document_type_id}"
    fabric_invoke(
        1,
        "mychannel",
        "watheq",
        "CreateDoc",
        [doc_id, cid, document_front.name, owner, sha],
    )

    return {
        "doc_id": doc_id,
        "cid": cid,
        "sha256": sha,
        "filename": document_front.name,
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
