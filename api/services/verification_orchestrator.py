"""Verification Orchestrator — منظم مراحل التحقق

Executes the 8-stage sequential verification pipeline:
  1. DOCUMENT_IMAGE_QUALITY — جودة الصورة
  2. DOCUMENT_CROPPING — قص الوثيقة واستخراج المخطط
  3. DOCUMENT_FACE_EXTRACTION — استخراج وجه البطاقة
  4. FACE_MATCHING — مطابقة الوجه مع السيلفي
  5. OCR — قراءة النصوص (Google Vision)
  6. AI_VERIFICATION — تحقق الذكاء الاصطناعي (ElementClassifier + FontAnalyzer v3)
  7. DATA_VERIFICATION — مطابقة البيانات مع سجلات المواطنين
  8. BLOCKCHAIN — تسجيل على MultiChain + IPFS

Each stage must succeed before the next runs. Results are stored in the
verifications table and individual steps in verification_steps.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import cv2
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class _NumpySafeEncoder(json.JSONEncoder):
    """Handle numpy scalars / arrays that are not JSON-serializable."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, cls=_NumpySafeEncoder, ensure_ascii=False)


from api.database import (
    database as _db,
    get_verifications_collection,
    get_verification_steps_collection,
)
from api.models import VerificationStage, VerificationStatus
from api.services.notification_service import notification_bus, persist_notification
from api.services.verification_steps_service import (
    layout_gating_verify,
    ml_verify,
    ocr_verify,
    data_verification,
    blockchain_verify,
    document_image_quality,
    document_crop,
    document_face_extraction,
    face_matching,
)

logger = logging.getLogger("watheq.orchestrator")


@dataclass
class VerificationInput:
    verification_id: int
    document_front_path: Path
    document_back_path: Optional[Path]
    person_image_path: Path
    document_type_id: int
    owner_email: str


class VerificationOrchestrator:
    def __init__(self) -> None:
        self.verifications = get_verifications_collection()
        self.steps = get_verification_steps_collection()

    async def _get_user_id(self, verification_id: int) -> Optional[int]:
        """Look up the user_id from the verifications row."""
        row = await _db.fetch_one(
            "SELECT user_id FROM verifications WHERE id = :vid",
            values={"vid": verification_id},
        )
        return int(row["user_id"]) if row and row["user_id"] else None

    def _arabic_message(self, code: Optional[str]) -> str:
        messages = {
            "LOW_BRIGHTNESS": "\u0627\u0644\u0625\u0636\u0627\u0621\u0629 \u0645\u0646\u062e\u0641\u0636\u0629",
            "HIGH_BRIGHTNESS": "\u0627\u0644\u0625\u0636\u0627\u0621\u0629 \u0645\u0631\u062a\u0641\u0639\u0629",
            "BLURRY": "\u0627\u0644\u0635\u0648\u0631\u0629 \u063a\u064a\u0631 \u0648\u0627\u0636\u062d\u0629",
            "DOCUMENT_FACE_MISSING": "\u0644\u0627 \u064a\u0648\u062c\u062f \u0648\u062c\u0647 \u0645\u0633\u062a\u062e\u0631\u062c \u0645\u0646 \u0627\u0644\u0628\u0637\u0627\u0642\u0629",
            "DOCUMENT_CROP_MISSING": "\u0644\u0645 \u064a\u062a\u0645 \u0642\u0635 \u0627\u0644\u0628\u0637\u0627\u0642\u0629",
            "DOCUMENT_CONTOUR_NOT_DETECTED": "\u0644\u0645 \u064a\u062a\u0645 \u0627\u0644\u0639\u062b\u0648\u0631 \u0639\u0644\u0649 \u062d\u062f\u0648\u062f \u0627\u0644\u0628\u0637\u0627\u0642\u0629",
            "DOCUMENT_EDGES_NOT_FOUND": "\u0644\u0645 \u064a\u062a\u0645 \u0627\u0644\u0639\u062b\u0648\u0631 \u0639\u0644\u0649 \u062d\u0648\u0627\u0641 \u0627\u0644\u0628\u0637\u0627\u0642\u0629",
            "INVALID_DOCUMENT_IMAGE": "\u0635\u0648\u0631\u0629 \u0627\u0644\u0628\u0637\u0627\u0642\u0629 \u063a\u064a\u0631 \u0635\u0627\u0644\u062d\u0629",
            "DOCUMENT_FACE_NOT_DETECTED": "\u0644\u0645 \u064a\u062a\u0645 \u0627\u0644\u0639\u062b\u0648\u0631 \u0639\u0644\u0649 \u0648\u062c\u0647 \u0641\u064a \u0627\u0644\u0628\u0637\u0627\u0642\u0629",
            "LAYOUT_FAILED": "\u0641\u0634\u0644 \u062a\u062d\u0642\u0642 \u062a\u0637\u0627\u0628\u0642 \u0627\u0644\u0645\u062e\u0637\u0637",
            "STAMP_MISSING": "\u0627\u0644\u062e\u062a\u0645 \u063a\u064a\u0631 \u0645\u0648\u062c\u0648\u062f",
            "STAMP_WRONG_POSITION": "\u0627\u0644\u062e\u062a\u0645 \u0641\u064a \u0645\u0643\u0627\u0646 \u063a\u064a\u0631 \u0635\u062d\u064a\u062d",
            "NAME_MISSING": "\u062d\u0642\u0644 \u0627\u0644\u0627\u0633\u0645 \u0641\u0627\u0631\u063a",
            "NATIONAL_ID_MISSING": "\u0631\u0642\u0645 \u0627\u0644\u0647\u0648\u064a\u0629 \u063a\u064a\u0631 \u0645\u0648\u062c\u0648\u062f",
            "FACE_MATCH_FAILED": "\u0641\u0634\u0644 \u0645\u0637\u0627\u0628\u0642\u0629 \u0627\u0644\u0648\u062c\u0647",
            "FACE_MISMATCH": "\u0627\u0644\u0648\u062c\u0647 \u0644\u0627 \u064a\u0637\u0627\u0628\u0642 \u0635\u0648\u0631\u0629 \u0627\u0644\u0633\u064a\u0644\u0641\u064a",
            "DATA_FRAUD_SUSPECTED": "محاولة احتيال — بيانات الوثيقة لا تطابق السجل المحفوظ",
            "NATIONAL_ID_NOT_EXTRACTED": "لم يتم استخراج رقم الهوية من الوثيقة",
            "UNKNOWN_ERROR": "\u062d\u062f\u062b \u062e\u0637\u0623 \u0623\u062b\u0646\u0627\u0621 \u0627\u0644\u062a\u062d\u0642\u0642",
        }
        return messages.get(code or "UNKNOWN_ERROR", messages["UNKNOWN_ERROR"])

    def _infer_failure_code(self, message: str) -> Optional[str]:
        msg = message or ""
        if "Document face not available" in msg and "biometric" in msg.lower():
            return "DOCUMENT_FACE_MISSING"
        if "Document face not available" in msg:
            return "DOCUMENT_FACE_MISSING"
        if "Document crop not available" in msg:
            return "DOCUMENT_CROP_MISSING"
        if "Document contour not detected" in msg:
            return "DOCUMENT_CONTOUR_NOT_DETECTED"
        if "Document edges not found" in msg:
            return "DOCUMENT_EDGES_NOT_FOUND"
        if "Invalid document image" in msg:
            return "INVALID_DOCUMENT_IMAGE"
        if "No face detected in document" in msg:
            return "DOCUMENT_FACE_NOT_DETECTED"

        if "Layout gating failed" in msg:
            return "LAYOUT_FAILED"
        if "STAMP_MISSING" in msg:
            return "STAMP_MISSING"
        if "STAMP_WRONG_POSITION" in msg:
            return "STAMP_WRONG_POSITION"
        if "NAME_MISSING" in msg:
            return "NAME_MISSING"
        if "NATIONAL_ID_MISSING" in msg:
            return "NATIONAL_ID_MISSING"
        if "DeepFace" in msg:
            return "FACE_MATCH_FAILED"
        if "Face mismatch" in msg:
            return "FACE_MISMATCH"
        if "Fraud suspected" in msg or "fraud" in msg.lower():
            return "DATA_FRAUD_SUSPECTED"
        if "National_id not extracted" in msg:
            return "NATIONAL_ID_NOT_EXTRACTED"
        return None

    async def run(self, payload: VerificationInput) -> None:
        started_at = datetime.now(timezone.utc)
        await self.verifications.update_one(
            payload.verification_id,
            {
                "status": VerificationStatus.RUNNING.value,
                "current_stage": VerificationStage.DOCUMENT_IMAGE_QUALITY.value,
                "start_time": started_at,
                "error_message": None,
            },
        )

        # ── Broadcast RUNNING event to admin dashboards ──
        try:
            await notification_bus.broadcast(
                {
                    "type": "VERIFICATION_RUNNING",
                    "verification_id": payload.verification_id,
                    "created_at": started_at.isoformat(),
                }
            )
        except Exception:
            pass

        results: dict[str, Any] = {}

        rectified_path: Optional[Path] = None
        cropped_path: Optional[Path] = None
        doc_face_path: Optional[Path] = None

        # Sequential pipeline: each stage must succeed before moving to the next.
        for stage in [
            VerificationStage.DOCUMENT_IMAGE_QUALITY,
            VerificationStage.DOCUMENT_CROPPING,
            VerificationStage.DOCUMENT_FACE_EXTRACTION,
            VerificationStage.FACE_MATCHING,
            VerificationStage.OCR,
            VerificationStage.AI_VERIFICATION,
            VerificationStage.DATA_VERIFICATION,
            VerificationStage.BLOCKCHAIN,
        ]:
            failure_reason_code: Optional[str] = None
            step_id = await self.steps.insert_one(
                {
                    "verification_id": payload.verification_id,
                    "step_name": stage.value,
                    "stage": stage.value,
                    "status": VerificationStatus.RUNNING.value,
                    "error_message": None,
                    "start_time": datetime.now(timezone.utc),
                    "end_time": None,
                    "result_data": None,
                }
            )

            try:
                await self.verifications.update_one(
                    payload.verification_id,
                    {"current_stage": stage.value},
                )

                if stage == VerificationStage.DOCUMENT_IMAGE_QUALITY:
                    result = await asyncio.to_thread(
                        document_image_quality,
                        payload.document_front_path,
                    )
                    if not result.get("brightness_ok", True):
                        failure_reason_code = (
                            result.get("reason_code") or "LOW_BRIGHTNESS"
                        )
                        raise RuntimeError(self._arabic_message(failure_reason_code))
                    if not result.get("blur_ok", True):
                        failure_reason_code = result.get("reason_code") or "BLURRY"
                        raise RuntimeError(self._arabic_message(failure_reason_code))
                elif stage == VerificationStage.DOCUMENT_CROPPING:
                    debug_dir = payload.document_front_path.parent / "debug"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    input_image = cv2.imdecode(
                        np.fromfile(payload.document_front_path, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if input_image is not None:
                        print(
                            f"[DOC_CROP] vid={payload.verification_id} IN={input_image.shape[1]}x{input_image.shape[0]}"
                        )
                    else:
                        print(f"[DOC_CROP] vid={payload.verification_id} IN=unknown")
                    rectified_path = (
                        payload.document_front_path.parent / "document_rectified.jpg"
                    )
                    cropped_path = rectified_path
                    try:
                        result = await asyncio.to_thread(
                            document_crop,
                            payload.document_front_path,
                            rectified_path,
                        )
                        rectified_image = cv2.imdecode(
                            np.fromfile(rectified_path, dtype=np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        if rectified_image is None:
                            raise RuntimeError("BAD_RECTIFIED")
                        print(
                            f"[DOC_CROP] vid={payload.verification_id} PASS method=rectify_warp "
                            f"RECT={rectified_image.shape[1]}x{rectified_image.shape[0]}"
                        )
                        cv2.imwrite(
                            str(debug_dir / "rectified_orchestrator.jpg"),
                            rectified_image,
                        )
                    except Exception as exc:
                        print(
                            f"[DOC_CROP] vid={payload.verification_id} FAIL reason={exc}"
                        )
                        if input_image is not None:
                            cv2.imwrite(
                                str(debug_dir / "input_orchestrator.jpg"),
                                input_image,
                            )
                        raise
                    layout_result = await asyncio.to_thread(
                        layout_gating_verify,
                        rectified_path,
                    )
                    results["DOCUMENT_LAYOUT"] = layout_result
                    overlay_src = (layout_result.get("artifacts") or {}).get(
                        "overlay_png"
                    )
                    if overlay_src and Path(overlay_src).exists():
                        shutil.copyfile(
                            overlay_src, debug_dir / "overlay_on_rectified.jpg"
                        )
                    if (layout_result.get("layout_status") or "").upper() == "FAIL":
                        failure_reason_code = (
                            layout_result.get("reason") or "LAYOUT_FAILED"
                        )
                        raise RuntimeError(self._arabic_message(failure_reason_code))
                elif stage == VerificationStage.DOCUMENT_FACE_EXTRACTION:
                    if rectified_path is None:
                        raise RuntimeError("Document crop not available")
                    doc_face_path = (
                        payload.document_front_path.parent / "document_face.jpg"
                    )
                    result = await asyncio.to_thread(
                        document_face_extraction,
                        rectified_path,
                        doc_face_path,
                    )
                elif stage == VerificationStage.FACE_MATCHING:
                    if rectified_path is None:
                        raise RuntimeError("Rectified document not available")
                    # Prefer the extracted document face for comparison when
                    # available.  The full rectified document is the fallback
                    # so that FaceService._id_likeness_score can still
                    # classify it as an ID card if needed.
                    face_src = (
                        doc_face_path
                        if doc_face_path is not None and doc_face_path.exists()
                        else rectified_path
                    )
                    logger.info(
                        "[ORCH:FACE_MATCHING] doc_face_path=%s  exists=%s",
                        doc_face_path,
                        doc_face_path.exists() if doc_face_path else False,
                    )
                    logger.info(
                        "[ORCH:FACE_MATCHING] rectified_path=%s", rectified_path
                    )
                    logger.info("[ORCH:FACE_MATCHING] → face_src chosen: %s", face_src)
                    logger.info(
                        "[ORCH:FACE_MATCHING] person_image=%s  exists=%s",
                        payload.person_image_path,
                        payload.person_image_path.exists(),
                    )
                    result = await asyncio.to_thread(
                        face_matching,
                        face_src,
                        payload.person_image_path,
                        rectified_path,
                    )
                    logger.info("[ORCH:FACE_MATCHING] result=%s", result)
                    if not result.get("accepted", False):
                        pct = result.get("similarity_percent", 0)
                        failure_reason_code = "FACE_MISMATCH"
                        raise RuntimeError(
                            f"Face mismatch — similarity {pct}% "
                            f"(threshold {result.get('accept_threshold_percent', 80)}%)"
                        )
                elif stage == VerificationStage.OCR:
                    if rectified_path is None:
                        raise RuntimeError("Rectified image not available")
                    src = rectified_path
                    result = await asyncio.to_thread(
                        ocr_verify,
                        src,
                    )
                elif stage == VerificationStage.AI_VERIFICATION:
                    if rectified_path is None:
                        raise RuntimeError("Rectified image not available")
                    src = rectified_path
                    # Look up folder_name from document_types for AI verification
                    dt_row = await _db.fetch_one(
                        "SELECT folder_name FROM document_types WHERE id = :dtid",
                        values={"dtid": payload.document_type_id},
                    )
                    doc_folder = dt_row["folder_name"] if dt_row else "identity"
                    result = await asyncio.to_thread(
                        ml_verify,
                        src,
                        doc_folder,
                    )
                elif stage == VerificationStage.DATA_VERIFICATION:
                    ocr_result = results.get(VerificationStage.OCR.value) or {}
                    result = await data_verification(
                        ocr_result=ocr_result,
                        document_type_id=payload.document_type_id,
                    )
                else:
                    if rectified_path is None:
                        raise RuntimeError("Rectified image not available")
                    src = rectified_path
                    result = await asyncio.to_thread(
                        blockchain_verify,
                        src,
                        document_type_id=payload.document_type_id,
                        owner=payload.owner_email,
                    )

                results[stage.value] = result

                await self.steps.update_one(
                    step_id,
                    {
                        "status": VerificationStatus.SUCCESS.value,
                        "end_time": datetime.now(timezone.utc),
                        "result_data": _safe_json(result),
                    },
                )
            except Exception as exc:
                failure_reason_code = (
                    failure_reason_code
                    or self._infer_failure_code(str(exc))
                    or "UNKNOWN_ERROR"
                )
                error_message = self._arabic_message(failure_reason_code)
                raw_error = f"[{failure_reason_code}] {exc}"
                print(
                    f"[ORCHESTRATOR] vid={payload.verification_id} stage={stage.value} ERROR: {raw_error}"
                )
                results["failure_reason_code"] = failure_reason_code
                results["raw_error"] = str(exc)
                await self.steps.update_one(
                    step_id,
                    {
                        "status": VerificationStatus.FAILED.value,
                        "error_message": f"{error_message} | {raw_error}",
                        "end_time": datetime.now(timezone.utc),
                    },
                )
                try:
                    result_json = _safe_json(results)
                except Exception:
                    result_json = json.dumps({"error": "result serialization failed"})
                await self.verifications.update_one(
                    payload.verification_id,
                    {
                        "status": VerificationStatus.FAILED.value,
                        "current_stage": stage.value,
                        "error_message": f"{error_message} | {raw_error}",
                        "end_time": datetime.now(timezone.utc),
                        "result_data": result_json,
                    },
                )

                # ── Push real-time notification to admin dashboards ──
                try:
                    # Resolve user name & document type name for the notification
                    _user_row = await _db.fetch_one(
                        "SELECT name FROM users WHERE id = :uid",
                        values={
                            "uid": await self._get_user_id(payload.verification_id)
                        },
                    )
                    _dt_row = await _db.fetch_one(
                        "SELECT name FROM document_types WHERE id = :did",
                        values={"did": payload.document_type_id},
                    )
                    _user_name = (
                        _user_row["name"] if _user_row else payload.owner_email
                    ) or payload.owner_email
                    _doc_type_name = (
                        _dt_row["name"] if _dt_row else str(payload.document_type_id)
                    )
                    _notif_message = (
                        f"فشل التحقق من وثيقة ({_doc_type_name}) للمستخدم {_user_name}"
                    )

                    notif_id = await persist_notification(
                        verification_id=payload.verification_id,
                        message=_notif_message,
                        document_type_name=_doc_type_name,
                        user_name=_user_name,
                        failure_stage=stage.value,
                        failure_reason_code=failure_reason_code,
                    )
                    await notification_bus.broadcast(
                        {
                            "type": "VERIFICATION_FAILED",
                            "id": notif_id,
                            "verification_id": payload.verification_id,
                            "message": _notif_message,
                            "document_type_name": _doc_type_name,
                            "user_name": _user_name,
                            "failure_stage": stage.value,
                            "failure_reason_code": failure_reason_code,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                except Exception as notif_exc:
                    print(f"[ORCHESTRATOR] notification push failed: {notif_exc}")

                return

        face_result = results.get(VerificationStage.FACE_MATCHING.value) or {}
        face_match = None
        if isinstance(face_result, dict):
            if "match" in face_result:
                face_match = bool(face_result.get("match"))
            elif "verified" in face_result:
                face_match = bool(face_result.get("verified"))

        layout_result = results.get("DOCUMENT_LAYOUT") or {}
        layout_status = layout_result.get("layout_status")

        ai_result = results.get(VerificationStage.AI_VERIFICATION.value) or {}
        ai_final_decision = ai_result.get("final_decision")

        data_result = results.get(VerificationStage.DATA_VERIFICATION.value) or {}
        data_match = data_result.get("data_match", False)

        ocr_done = VerificationStage.OCR.value in results

        blockchain_result = results.get(VerificationStage.BLOCKCHAIN.value) or {}
        blockchain_cid = blockchain_result.get("cid")

        results["SUMMARY"] = {
            "face_match": face_match,
            "layout_status": layout_status,
            "ai_final_decision": ai_final_decision,
            "data_match": data_match,
            "ocr_done": ocr_done,
            "blockchain_cid": blockchain_cid,
        }

        await self.verifications.update_one(
            payload.verification_id,
            {
                "status": VerificationStatus.SUCCESS.value,
                "current_stage": VerificationStage.BLOCKCHAIN.value,
                "end_time": datetime.now(timezone.utc),
                "result_data": _safe_json(results),
            },
        )

        # ── Broadcast SUCCESS event to admin dashboards ──
        try:
            await notification_bus.broadcast(
                {
                    "type": "VERIFICATION_SUCCESS",
                    "verification_id": payload.verification_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        except Exception:
            pass
