"""Verification Step Implementations — تنفيذ مراحل التحقق

Contains the concrete functions called by the orchestrator for each
pipeline stage: image quality checks, document cropping, face extraction,
face matching, OCR, AI verification, data verification against citizen
records, and blockchain recording via MultiChain + IPFS.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger("watheq.verification_steps")

from ledger.ipfs_service import IPFSService
from Biometric.face_service import FaceService
from ocr.vision_service_ocr import ocr_image, ocr_pdf
from api.services.multichain_service import publish_to_stream, json_to_hex

QUALITY_BRIGHTNESS_MIN = 40
QUALITY_BRIGHTNESS_MAX = 220
QUALITY_BLUR_MIN = 70.0
QUALITY_MIN_AREA_RATIO = 0.20
QUALITY_MAX_AREA_RATIO = 0.95
QUALITY_MIN_ASPECT = 1.2
QUALITY_MAX_ASPECT = 2.2


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_image_exif_safe(path: Path) -> np.ndarray | None:
    """Read an image file and apply EXIF orientation.

    This is a safety net in case the upload handler didn't normalize.
    """
    try:
        pil_img = Image.open(path)
        pil_img = ImageOps.exif_transpose(pil_img)
        arr = np.array(pil_img)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return arr
    except Exception:
        # Fallback to raw cv2 decode
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def _ensure_correct_orientation(rectified: np.ndarray) -> np.ndarray:
    """Post-rectification sanity check: detect if the card is upside-down.

    ID cards have a face photo typically in the upper portion.  If we detect
    a face in the bottom half but not the top half, the card is likely
    upside-down (180° rotated).  We also check text density: real ID cards
    have more horizontal features (text) in specific areas.
    """
    h, w = rectified.shape[:2]
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)

    # Try face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
    )

    if len(faces) > 0:
        # Find the largest face
        largest = max(faces, key=lambda f: f[2] * f[3])
        face_center_y = largest[1] + largest[3] / 2

        # If the face center is in the bottom 40%, likely upside-down
        if face_center_y > h * 0.6:
            # Double-check: try rotating 180° and detect again
            rotated = cv2.rotate(rectified, cv2.ROTATE_180)
            gray_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            faces_rot = face_cascade.detectMultiScale(
                gray_rot, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
            )
            if len(faces_rot) > 0:
                largest_rot = max(faces_rot, key=lambda f: f[2] * f[3])
                face_center_y_rot = largest_rot[1] + largest_rot[3] / 2
                # If the face is now in the top 60%, the rotation fixed it
                if face_center_y_rot < h * 0.6:
                    print("[ORIENTATION] Detected upside-down card — rotating 180°")
                    return rotated
    else:
        # No face found at all — try 180° rotation
        rotated = cv2.rotate(rectified, cv2.ROTATE_180)
        gray_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        faces_rot = face_cascade.detectMultiScale(
            gray_rot, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
        )
        if len(faces_rot) > 0:
            largest_rot = max(faces_rot, key=lambda f: f[2] * f[3])
            face_center_y_rot = largest_rot[1] + largest_rot[3] / 2
            if face_center_y_rot < h * 0.6:
                print("[ORIENTATION] No face in original, found after 180° — rotating")
                return rotated

    return rectified


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
    """Accept the pre-cropped document image from the mobile app.

    The Flutter app crops the camera capture to the overlay frame before
    uploading, so the image already contains just the document card.
    This stage only normalises EXIF orientation and copies to output_path.
    No rotation or flipping is applied — the app delivers the correct
    orientation.
    """
    image = _read_image_exif_safe(document_front)
    if image is None:
        raise RuntimeError("Invalid document image")

    h, w = image.shape[:2]
    print(f"DOCUMENT_CROPPING IN: {w}x{h}")

    # Just save the EXIF-normalised image as-is — the app already
    # cropped it correctly and in the right orientation.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", image)[1].tofile(str(output_path))

    print(f"DOCUMENT_CROPPING: PASS method=app_precrop size={w}x{h}")
    return {
        "cropped_path": str(output_path),
        "rectified_path": str(output_path),
        "method": "app_precrop",
    }


def _find_document_quad(
    image: np.ndarray, h: int, w: int
) -> tuple[np.ndarray | None, str]:
    """Try multiple edge-detection strategies to find a 4-point card contour."""

    min_area = h * w * QUALITY_MIN_AREA_RATIO
    max_area = h * w * QUALITY_MAX_AREA_RATIO
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Bilateral filter preserves card edges better than Gaussian
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_lg = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # --- Strategy A: Canny edge detection ---
    edged_canny = cv2.Canny(filtered, 30, 150)
    # Use morphological CLOSE to connect edge gaps without expanding outward
    edged_canny = cv2.morphologyEx(edged_canny, cv2.MORPH_CLOSE, kernel_lg)

    # --- Strategy B: Adaptive threshold (works better on uneven lighting) ---
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    # --- Strategy C: Otsu threshold ---
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel_lg)

    for label, edge_img in [
        ("canny", edged_canny),
        ("adaptive", thresh),
        ("otsu", otsu),
    ]:
        contours, _ = cv2.findContours(
            edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        # Pass 1: look for a clean 4-point polygon
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                # Validate: aspect ratio of the bounding rect should be card-like
                rect = cv2.minAreaRect(approx)
                rw_r, rh_r = rect[1]
                if rw_r > 0 and rh_r > 0:
                    aspect = max(rw_r, rh_r) / min(rw_r, rh_r)
                    if QUALITY_MIN_ASPECT <= aspect <= QUALITY_MAX_ASPECT:
                        return approx, f"quad_{label}"

        # Pass 2: use minAreaRect of the largest valid contour
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            rect = cv2.minAreaRect(cnt)
            rw_r, rh_r = rect[1]
            if rw_r > 0 and rh_r > 0:
                aspect = max(rw_r, rh_r) / min(rw_r, rh_r)
                if QUALITY_MIN_ASPECT <= aspect <= QUALITY_MAX_ASPECT:
                    box = cv2.boxPoints(rect)
                    quad = np.int32(box).reshape(4, 1, 2)
                    return quad, f"rect_{label}"

    return None, ""


def _shrink_quad(pts: np.ndarray, fraction: float = 0.01) -> np.ndarray:
    """Shrink a 4-point polygon inward by *fraction* of its size.

    This trims 1-2 % off each edge so the warp doesn't include
    background pixels that cling to the detected contour boundary.
    """
    center = pts.mean(axis=0)
    return (center + (1.0 - fraction) * (pts - center)).astype(np.float32)


def _warp_quad(
    image: np.ndarray,
    best_quad: np.ndarray,
    output_path: Path,
    method_tag: str,
) -> dict[str, Any]:
    """Perspective-warp the image using a detected 4-point quad."""
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

    # Shrink quad inward ~1.5 % to eliminate background fringe
    ordered = _shrink_quad(ordered, fraction=0.015)

    print(f"[CROP] method={method_tag} quad={ordered.tolist()}")

    # Compute target width/height
    wA = np.linalg.norm(ordered[2] - ordered[3])
    wB = np.linalg.norm(ordered[1] - ordered[0])
    target_w = int(max(wA, wB))
    hA = np.linalg.norm(ordered[1] - ordered[2])
    hB = np.linalg.norm(ordered[0] - ordered[3])
    target_h = int(max(hA, hB))

    # Ensure landscape orientation for an ID card
    if target_w < target_h:
        target_w, target_h = target_h, target_w
        ordered = np.array(
            [ordered[3], ordered[0], ordered[1], ordered[2]], dtype=np.float32
        )

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

    # Post-rectification: ensure the card is right-side-up using face detection
    rectified = _ensure_correct_orientation(rectified)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", rectified)[1].tofile(str(output_path))

    print(f"RECT: {target_w}x{target_h}")
    print(f"DOCUMENT_CROPPING: PASS method={method_tag} size={target_w}x{target_h}")
    return {
        "cropped_path": str(output_path),
        "rectified_path": str(output_path),
        "method": method_tag,
    }


def document_face_extraction(cropped_path: Path, output_path: Path) -> dict[str, Any]:
    """Extract the face from the rectified document using face detection."""
    logger.info("[FACE_EXTRACT] === Starting face extraction ===")
    logger.info(
        "[FACE_EXTRACT] cropped_path=%s  exists=%s  size=%s bytes",
        cropped_path,
        cropped_path.exists(),
        cropped_path.stat().st_size if cropped_path.exists() else "N/A",
    )
    image = _read_image_exif_safe(cropped_path)
    if image is None:
        logger.error("[FACE_EXTRACT] FAILED — could not read cropped image")
        raise RuntimeError("Invalid cropped document image")

    h, w = image.shape[:2]
    logger.info(
        "[FACE_EXTRACT] image loaded: %dx%d  channels=%d",
        w,
        h,
        image.shape[2] if len(image.shape) > 2 else 1,
    )

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

    logger.info("[FACE_EXTRACT] ROI from layout template: %s", roi)
    if roi:
        x0 = max(0, min(int(round(roi["x"] * w)), w))
        y0 = max(0, min(int(round(roi["y"] * h)), h))
        x1 = max(0, min(int(round((roi["x"] + roi["w"]) * w)), w))
        y1 = max(0, min(int(round((roi["y"] + roi["h"]) * h)), h))
        if x1 > x0 and y1 > y0:
            face = image[y0:y1, x0:x1].copy()
            logger.info(
                "[FACE_EXTRACT] layout_roi crop: (%d,%d)->(%d,%d) = %dx%d",
                x0,
                y0,
                x1,
                y1,
                x1 - x0,
                y1 - y0,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), face)
            logger.info("[FACE_EXTRACT] SUCCESS via layout_roi -> %s", output_path)
            return {"document_face_path": str(output_path), "source": "layout_roi"}

    # Fallback 1: OpenCV Haar cascade face detection
    logger.info("[FACE_EXTRACT] Trying Haar cascade...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
    )
    logger.info("[FACE_EXTRACT] Haar cascade found %d face(s)", len(faces))

    if len(faces) > 0:
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
        logger.info(
            "[FACE_EXTRACT] Haar crop: (%d,%d,%d,%d) = %dx%d", fx, fy, fw, fh, fw, fh
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), face)
        logger.info("[FACE_EXTRACT] SUCCESS via haar_cascade -> %s", output_path)

        return {
            "document_face_path": str(output_path),
            "source": "haar_cascade",
            "bbox": {"x": int(fx), "y": int(fy), "w": int(fw), "h": int(fh)},
        }

    # Fallback 2: DeepFace RetinaFace — much more robust for ID card photos
    logger.info("[FACE_EXTRACT] Trying RetinaFace fallback...")
    try:
        from deepface import DeepFace

        df_faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend="retinaface",
            enforce_detection=False,
        )
        if df_faces and df_faces[0].get("confidence", 0) > 0:
            region = df_faces[0].get("facial_area", {})
            fx = region.get("x", 0)
            fy = region.get("y", 0)
            fw = region.get("w", 0)
            fh = region.get("h", 0)

            if fw > 0 and fh > 0:
                # Add margin
                margin = int(max(fw, fh) * 0.15)
                fx = max(0, fx - margin)
                fy = max(0, fy - margin)
                fw = min(w - fx, fw + 2 * margin)
                fh = min(h - fy, fh + 2 * margin)

                face = image[fy : fy + fh, fx : fx + fw].copy()
                logger.info(
                    "[FACE_EXTRACT] RetinaFace crop: (%d,%d,%d,%d) = %dx%d",
                    fx,
                    fy,
                    fw,
                    fh,
                    fw,
                    fh,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), face)
                logger.info("[FACE_EXTRACT] SUCCESS via retinaface -> %s", output_path)

                return {
                    "document_face_path": str(output_path),
                    "source": "retinaface",
                    "bbox": {"x": int(fx), "y": int(fy), "w": int(fw), "h": int(fh)},
                }
    except Exception as e:
        logger.warning("[FACE_EXTRACT] RetinaFace fallback error: %s", e)

    logger.error("[FACE_EXTRACT] FAILED — no face detected by any method")
    raise RuntimeError("No face detected in document")


# ── Face-matching helpers ────────────────────────────────────────────

# Models ordered by robustness for cross-domain (ID-card vs live-selfie) matching.
# ArcFace  – additive angular margin + ResNet-100 → most robust to quality gap.
# Facenet512 – 512-dim embeddings → more expressive than Facenet-128.
# VGG-Face dropped: consistently worst performer in logs (27-28% similarity).
_VERIFY_MODELS = ["ArcFace", "Facenet512"]

# Explicit acceptance threshold (percent).  Cross-domain ID-card vs live-selfie
# inherently yields lower similarity than selfie-vs-selfie due to printing
# artefacts, lighting differences, age gap, and resolution gap.  60 % is the
# industry-standard sweet-spot: high enough to reject impostors, low enough to
# accept genuine cross-domain matches.
_ACCEPT_SIMILARITY_PCT = 60.0


def _preprocess_id_face(img: np.ndarray) -> np.ndarray:
    """Enhance a document-face crop for better embedding quality.

    ID-card photos are often low-contrast and washed-out from scanning /
    photographing.  CLAHE on the L channel of LAB space equalises contrast
    without colour distortion, and mild denoising reduces scan artefacts.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    enhanced = cv2.merge([l_ch, a_ch, b_ch])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)
    logger.info("[PREPROCESS] CLAHE + denoise applied to document face")
    return enhanced


def _upscale_if_small(img: np.ndarray, label: str = "") -> np.ndarray:
    """Upscale image if too small for reliable face detection / recognition.

    Uses LANCZOS4 interpolation (sharper than CUBIC, fewer artefacts on
    small face crops) and targets at least 400 px on the shortest side.
    """
    h, w = img.shape[:2]
    min_dim = min(h, w)
    if min_dim < 400:
        scale = max(2, -(-400 // min_dim))  # ceiling division
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
        logger.info(
            "[UPSCALE:%s] %dx → %dx%d (was %dx%d, min_dim=%d, LANCZOS4)",
            label,
            scale,
            img.shape[1],
            img.shape[0],
            w,
            h,
            min_dim,
        )
    return img


def _try_verify(
    img1: np.ndarray,
    img2: np.ndarray,
    model: str,
    label: str,
) -> dict[str, Any] | None:
    """Run ``DeepFace.verify()`` with *one* model.  Returns a result dict or ``None``."""
    from deepface import DeepFace

    try:
        logger.info(
            "[VERIFY:%s] model=%s  img1=%dx%d  img2=%dx%d",
            label,
            model,
            img1.shape[1],
            img1.shape[0],
            img2.shape[1],
            img2.shape[0],
        )
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name=model,
            detector_backend="retinaface",
            enforce_detection=False,  # critical for low-res ID-card faces
            distance_metric="cosine",
        )
        distance = float(result.get("distance", 1.0))
        verified = bool(result.get("verified", False))
        threshold = float(result.get("threshold", 0.40))
        similarity_pct = round(max(0.0, (1.0 - distance)) * 100, 2)

        logger.info(
            "[VERIFY:%s] model=%s  distance=%.6f  threshold=%.4f  "
            "verified=%s  similarity=%.2f%%",
            label,
            model,
            distance,
            threshold,
            verified,
            similarity_pct,
        )
        return {
            "distance": distance,
            "verified": verified,
            "threshold": threshold,
            "similarity_pct": similarity_pct,
            "model": model,
        }
    except Exception as e:
        logger.warning("[VERIFY:%s] model=%s FAILED: %s", label, model, e)
        return None


def face_matching(
    document_face_or_rect_path: Path,
    person_image: Path,
    rectified_document_path: Path | None = None,
) -> dict[str, Any]:
    """Compare a document face against the selfie.

    Uses ``DeepFace.verify()`` with ArcFace → Facenet512 to overcome the
    **domain shift** between low-resolution printed ID-card photos and
    live selfie images.

    Acceptance is based on an **explicit similarity threshold**
    (``_ACCEPT_SIMILARITY_PCT``) rather than DeepFace's per-model
    ``verified`` flag, which was effectively only ~32 % for ArcFace.

    The document face is pre-processed with CLAHE + denoising to
    compensate for scan / photo artefacts.
    """
    logger.info("======================================================")
    logger.info("[FACE_MATCH] === Starting face matching (multi-model) ===")
    logger.info("[FACE_MATCH] doc_path=%s", document_face_or_rect_path)
    logger.info("[FACE_MATCH] person_image=%s", person_image)
    logger.info("[FACE_MATCH] rectified_path=%s", rectified_document_path)

    if not document_face_or_rect_path.exists():
        raise RuntimeError(
            "Document face / rectified image not found — previous stages may have failed"
        )
    if not person_image.exists():
        raise RuntimeError("Person/selfie image not found")

    logger.info(
        "[FACE_MATCH] doc size=%d bytes  selfie size=%d bytes",
        document_face_or_rect_path.stat().st_size,
        person_image.stat().st_size,
    )

    # Read images
    img_doc = cv2.imdecode(
        np.fromfile(document_face_or_rect_path, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    img_selfie = cv2.imdecode(
        np.fromfile(person_image, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if img_doc is None:
        raise RuntimeError("Cannot read document face image")
    if img_selfie is None:
        raise RuntimeError("Cannot read selfie image")

    logger.info(
        "[FACE_MATCH] doc: %dx%d  selfie: %dx%d",
        img_doc.shape[1],
        img_doc.shape[0],
        img_selfie.shape[1],
        img_selfie.shape[0],
    )

    # Pre-process document face: CLAHE contrast enhancement + denoise
    img_doc = _preprocess_id_face(img_doc)

    # Upscale small images for better face detection & recognition
    img_doc_up = _upscale_if_small(img_doc, "doc")
    img_selfie_up = _upscale_if_small(img_selfie, "selfie")

    best_similarity: float = 0.0
    best_model: str = ""
    accepted: bool = False

    # ── Pass 1: extracted document face vs selfie ──────────────────
    logger.info(
        "[FACE_MATCH] --- Pass 1: doc_face vs selfie (threshold=%.1f%%) ---",
        _ACCEPT_SIMILARITY_PCT,
    )
    for model in _VERIFY_MODELS:
        vr = _try_verify(img_doc_up, img_selfie_up, model, "pass1")
        if vr is None:
            continue
        if vr["similarity_pct"] > best_similarity:
            best_similarity = vr["similarity_pct"]
            best_model = vr["model"]
        if vr["similarity_pct"] >= _ACCEPT_SIMILARITY_PCT:
            accepted = True
            best_similarity = vr["similarity_pct"]
            best_model = vr["model"]
            logger.info(
                "[FACE_MATCH] ✓ ACCEPTED by %s at %.2f%% (>= %.1f%%)",
                model,
                vr["similarity_pct"],
                _ACCEPT_SIMILARITY_PCT,
            )
            break

    # ── Pass 2: full rectified document vs selfie (if pass 1 failed) ──
    if not accepted:
        can_pass2 = (
            rectified_document_path is not None
            and rectified_document_path.exists()
            and rectified_document_path != document_face_or_rect_path
        )
        logger.info(
            "[FACE_MATCH] --- Pass 2: full_doc vs selfie  (eligible=%s) ---",
            can_pass2,
        )
        if can_pass2:
            img_full = cv2.imdecode(
                np.fromfile(rectified_document_path, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if img_full is not None:
                img_full_up = _upscale_if_small(img_full, "full_doc")
                for model in _VERIFY_MODELS:
                    vr = _try_verify(img_full_up, img_selfie_up, model, "pass2")
                    if vr is None:
                        continue
                    if vr["similarity_pct"] > best_similarity:
                        best_similarity = vr["similarity_pct"]
                        best_model = vr["model"]
                    if vr["similarity_pct"] >= _ACCEPT_SIMILARITY_PCT:
                        accepted = True
                        best_similarity = vr["similarity_pct"]
                        best_model = vr["model"]
                        logger.info(
                            "[FACE_MATCH] ✓ ACCEPTED by %s at %.2f%% (pass2, >= %.1f%%)",
                            model,
                            vr["similarity_pct"],
                            _ACCEPT_SIMILARITY_PCT,
                        )
                        break

    # ── Final result ──────────────────────────────────────────────
    if accepted:
        logger.info(
            "[FACE_MATCH] === FINAL: ACCEPTED  similarity=%.2f%%  model=%s ===",
            best_similarity,
            best_model,
        )
    else:
        logger.warning(
            "[FACE_MATCH] === FINAL: REJECTED  similarity=%.2f%%  best_model=%s ===",
            best_similarity,
            best_model,
        )
    logger.info("======================================================")

    return {
        "similarity_percent": best_similarity,
        "accepted": accepted,
        "accept_threshold_percent": _ACCEPT_SIMILARITY_PCT,
    }


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
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Try to extract national ID (Yemeni format: digits, typically 8-12)
    id_match = re.search(r"\b(\d{8,12})\b", text)
    if id_match:
        fields["national_id"] = id_match.group(1)

    # Extract name: the line immediately AFTER the line containing the national ID.
    # The national ID line is e.g. "الرقم الوطني 01310001042" and the next line is the person's name.
    if fields.get("national_id"):
        for i, line in enumerate(lines):
            if fields["national_id"] in line:
                # Next line is the full name
                if i + 1 < len(lines):
                    candidate = lines[i + 1]
                    # Verify it's predominantly Arabic (the actual name, not a header)
                    arabic_chars = sum(
                        1 for c in candidate if "\u0600" <= c <= "\u06ff"
                    )
                    if arabic_chars >= 3:
                        fields["full_name_ar"] = candidate
                break

    # Fallback: if we still don't have a name, try the old generic approach
    # but skip common header words
    if "full_name_ar" not in fields:
        header_keywords = {
            "الجمهورية",
            "اليمنية",
            "وزارة",
            "الداخلية",
            "مصلحة",
            "الأحوال",
            "المدنية",
            "السجل",
            "المدني",
            "بطاقة",
            "شخصية",
            "الرقم",
            "الوطني",
            "جمهورية",
        }
        for line in lines:
            arabic_chars = sum(1 for c in line if "\u0600" <= c <= "\u06ff")
            if arabic_chars < 4:
                continue
            words = line.split()
            non_header = [w for w in words if w not in header_keywords]
            if len(non_header) >= 2:
                fields["full_name_ar"] = " ".join(non_header)
                break

    # Date pattern (yyyy/mm/dd or dd/mm/yyyy or dd-mm-yyyy)
    dates = re.findall(r"\b(\d{1,4}[/-]\d{1,2}[/-]\d{1,4})\b", text)
    if dates:
        fields["date_of_birth"] = dates[0]
        if len(dates) > 1:
            fields["issue_date"] = dates[1]
        if len(dates) > 2:
            fields["expiry_date"] = dates[2]

    # Try to extract address from "المكان وتاريخ الميلاد" or similar line
    for line in lines:
        if "الميلاد" in line or "مكان" in line:
            # Extract the part after the label
            parts = re.split(r"الميلاد", line, maxsplit=1)
            if len(parts) > 1:
                addr = parts[1].strip().lstrip("-").strip()
                # Remove the date from the address if present
                addr = re.sub(
                    r"\s*-?\s*\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\s*$", "", addr
                ).strip()
                if addr:
                    fields["address"] = addr
            break

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
    from api.database import (
        get_citizen_records_collection,
        get_document_type_collection,
    )

    fields = _parse_ocr_fields(ocr_result)
    national_id = fields.get("national_id")

    if not national_id:
        raise RuntimeError(
            "National_id not extracted from OCR — cannot verify citizen data"
        )

    # Resolve document type name from DB so it gets stored in citizen_records
    doc_type_name: str | None = None
    try:
        dt_col = get_document_type_collection()
        dt_row = await dt_col.find_one({"_id": document_type_id})
        if dt_row:
            doc_type_name = dt_row.get("name")
    except Exception:
        pass
    if doc_type_name:
        fields["document_type"] = doc_type_name

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
