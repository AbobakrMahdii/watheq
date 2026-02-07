from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ledger.ipfs_service import IPFSService
from Biometric.face_service import FaceService
from ocr.vision_service_ocr import ocr_image, ocr_pdf
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
    image = cv2.imdecode(np.fromfile(document_front, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Invalid document image")

    print(f"IN: {image.shape[1]}x{image.shape[0]}")

    import subprocess
    import sys
    import tempfile
    import shutil

    repo_root = Path(__file__).resolve().parents[2]
    rectify_script = repo_root / "ai" / "card_verification" / "detect_and_rectify_card.py"
    if not rectify_script.exists():
        raise RuntimeError("detect_and_rectify_card.py not found")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(output_path.parent), prefix="rectify_run_") as tmpdir:
        tmp_out = Path(tmpdir)
        cmd = [
            sys.executable,
            str(rectify_script),
            "--image",
            str(document_front),
            "--out_dir",
            str(tmp_out),
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            reason = "NO_CARD_QUAD"
            combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if "BAD_RECTIFIED" in combined:
                reason = "BAD_RECTIFIED"
            print(f"DOCUMENT_CROPPING: FAILED reason={reason}")
            raise RuntimeError(reason)

        rectified = tmp_out / "card_rectified.png"
        if not rectified.exists():
            print("DOCUMENT_CROPPING: FAILED reason=BAD_RECTIFIED")
            raise RuntimeError("BAD_RECTIFIED")

        shutil.copyfile(rectified, output_path)

    rectified_img = cv2.imdecode(np.fromfile(output_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if rectified_img is None:
        print("DOCUMENT_CROPPING: FAILED reason=BAD_RECTIFIED")
        raise RuntimeError("BAD_RECTIFIED")

    print(f"RECT: {rectified_img.shape[1]}x{rectified_img.shape[0]}")
    print("DOCUMENT_CROPPING: PASS method=rectify_warp size=856x540")
    return {
        "cropped_path": str(output_path),
        "rectified_path": str(output_path),
    }


def document_face_extraction(cropped_path: Path, output_path: Path) -> dict[str, Any]:
    image = cv2.imdecode(np.fromfile(cropped_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Invalid cropped document image")

    repo_root = Path(__file__).resolve().parents[2]
    template_path = (
        repo_root
        / "ai"
        / "card_verification"
        / "registry"
        / "templates"
        / "national_id_yemen_v1"
        / "layout.yaml"
    )
    layout_report = cropped_path.parent / "layout" / "report.json"
    if not layout_report.exists():
        raise RuntimeError("Layout report missing; run layout verification first")
    if not template_path.exists():
        raise RuntimeError("Layout template not found for national_id_yemen_v1")

    try:
        report = json.loads(layout_report.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("Failed to read layout report") from exc

    if (report.get("layout_status") or "").upper() != "PASS":
        raise RuntimeError("Layout verification did not pass")

    try:
        template = json.loads(template_path.read_text(encoding="utf-8"))
    except Exception:
        template = None

    if template is None:
        import yaml

        with open(template_path, "r", encoding="utf-8") as f:
            template = yaml.safe_load(f)

    roi = (template.get("elements") or {}).get("photo", {}).get("roi")
    if not roi:
        raise RuntimeError("Photo ROI missing in template")

    h, w = image.shape[:2]
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

    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("Photo ROI produced empty crop")

    face = image[y0:y1, x0:x1].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), face)

    return {
        "document_face_path": str(output_path),
        "source": "layout_roi",
        "roi": {
            "x": float(roi["x"]),
            "y": float(roi["y"]),
            "w": float(roi["w"]),
            "h": float(roi["h"]),
        },
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


def layout_gating_verify(rectified_image: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    layout_script = repo_root / "ai" / "card_verification" / "layout_verify.py"
    template_path = (
        repo_root
        / "ai"
        / "card_verification"
        / "registry"
        / "templates"
        / "national_id_yemen_v1"
        / "layout.yaml"
    )
    if not layout_script.exists():
        raise RuntimeError("ai/card_verification/layout_verify.py not found")
    if not template_path.exists():
        raise RuntimeError("layout.yaml not found for national_id_yemen_v1")

    import subprocess
    import sys

    out_dir = rectified_image.parent / "layout"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(layout_script),
        "--image",
        str(rectified_image),
        "--rectified",
        str(rectified_image),
        "--template",
        str(template_path),
        "--out_dir",
        str(out_dir),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Layout verify failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )

    report_path = out_dir / "report.json"
    if not report_path.exists():
        raise RuntimeError("Layout report not generated")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    layout_status = report.get("layout_status") or "FAIL"
    reason = report.get("reason")

    return {
        "layout_status": layout_status,
        "reason": reason,
        "artifacts": {
            "report_json": str(report_path),
            "overlay_png": str(out_dir / "overlay_layout_verify.png"),
            "stamp_mask_png": str(out_dir / "stamp_mask.png"),
        },
    }


def ml_verify(document_front: Path, doc_type_folder: str = "identity") -> dict[str, Any]:
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
        "--image", str(document_front),
        "--type", doc_type_folder,
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
    scores = [r.get("score", 0) for r in element_results.values() if r.get("status") != "ERROR"]
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
