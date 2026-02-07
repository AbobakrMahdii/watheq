from __future__ import annotations

import asyncio
import json
import shutil
import cv2
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from api.database import get_verifications_collection, get_verification_steps_collection
from api.models import VerificationStage, VerificationStatus
from api.services.verification_steps_service import (
    biometric_verify,
    layout_gating_verify,
    ml_verify,
    ocr_verify,
    blockchain_verify,
    document_image_quality,
    document_crop,
    document_face_extraction,
    selfie_liveness_check,
    face_matching,
)


@dataclass
class VerificationInput:
    verification_id: int
    document_front_path: Path
    document_back_path: Optional[Path]
    person_image_path: Path
    document_type_id: int
    owner_email: str
    liveness_data: Optional[dict[str, Any]] = None


class VerificationOrchestrator:
    def __init__(self) -> None:
        self.verifications = get_verifications_collection()
        self.steps = get_verification_steps_collection()

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
            "LIVENESS_FAILED": "\u0641\u0634\u0644 \u0627\u0644\u062a\u062d\u0642\u0642 \u0627\u0644\u062d\u064a\u0648\u064a",
            "LAYOUT_FAILED": "\u0641\u0634\u0644 \u062a\u062d\u0642\u0642 \u062a\u0637\u0627\u0628\u0642 \u0627\u0644\u0645\u062e\u0637\u0637",
            "STAMP_MISSING": "\u0627\u0644\u062e\u062a\u0645 \u063a\u064a\u0631 \u0645\u0648\u062c\u0648\u062f",
            "STAMP_WRONG_POSITION": "\u0627\u0644\u062e\u062a\u0645 \u0641\u064a \u0645\u0643\u0627\u0646 \u063a\u064a\u0631 \u0635\u062d\u064a\u062d",
            "NAME_MISSING": "\u062d\u0642\u0644 \u0627\u0644\u0627\u0633\u0645 \u0641\u0627\u0631\u063a",
            "NATIONAL_ID_MISSING": "\u0631\u0642\u0645 \u0627\u0644\u0647\u0648\u064a\u0629 \u063a\u064a\u0631 \u0645\u0648\u062c\u0648\u062f",
            "BIOMETRIC_FAILED": "\u0641\u0634\u0644 \u0627\u0644\u062a\u062d\u0642\u0642 \u0627\u0644\u062d\u064a\u0648\u064a",
            "FACE_MISMATCH": "\u0627\u0644\u0648\u062c\u0647 \u0644\u0627 \u064a\u0637\u0627\u0628\u0642 \u0635\u0648\u0631\u0629 \u0627\u0644\u0633\u064a\u0644\u0641\u064a",
            "UNKNOWN_ERROR": "\u062d\u062f\u062b \u062e\u0637\u0623 \u0623\u062b\u0646\u0627\u0621 \u0627\u0644\u062a\u062d\u0642\u0642",
        }
        return messages.get(code or "UNKNOWN_ERROR", messages["UNKNOWN_ERROR"])

    def _infer_failure_code(self, message: str) -> Optional[str]:
        msg = message or ""
        if "Document face not available for biometric" in msg:
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
        if "Liveness check failed" in msg:
            return "LIVENESS_FAILED"
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
            return "BIOMETRIC_FAILED"
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

        results: dict[str, Any] = {}

        rectified_path: Optional[Path] = None
        cropped_path: Optional[Path] = None
        doc_face_path: Optional[Path] = None

        # Sequential pipeline: each stage must succeed before moving to the next.
        for stage in [
            VerificationStage.DOCUMENT_IMAGE_QUALITY,
            VerificationStage.DOCUMENT_CROPPING,
            VerificationStage.DOCUMENT_FACE_EXTRACTION,
            VerificationStage.SELFIE_LIVENESS,
            VerificationStage.FACE_MATCHING,
            VerificationStage.BIOMETRIC,
            VerificationStage.ML,
            VerificationStage.OCR,
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

                if stage == VerificationStage.BIOMETRIC:
                    if doc_face_path is None:
                        raise RuntimeError("Document face not available for biometric")
                    result = await asyncio.to_thread(
                        biometric_verify,
                        doc_face_path,
                        payload.person_image_path,
                    )
                elif stage == VerificationStage.DOCUMENT_IMAGE_QUALITY:
                    result = await asyncio.to_thread(
                        document_image_quality,
                        payload.document_front_path,
                    )
                    if not result.get("brightness_ok", True):
                        failure_reason_code = result.get("reason_code") or "LOW_BRIGHTNESS"
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
                    overlay_src = (layout_result.get("artifacts") or {}).get("overlay_png")
                    if overlay_src and Path(overlay_src).exists():
                        shutil.copyfile(overlay_src, debug_dir / "overlay_on_rectified.jpg")
                    if (layout_result.get("layout_status") or "").upper() == "FAIL":
                        failure_reason_code = layout_result.get("reason") or "LAYOUT_FAILED"
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
                elif stage == VerificationStage.SELFIE_LIVENESS:
                    result = await asyncio.to_thread(
                        selfie_liveness_check,
                        payload.liveness_data,
                    )
                elif stage == VerificationStage.FACE_MATCHING:
                    if doc_face_path is None:
                        raise RuntimeError("Document face not available")
                    result = await asyncio.to_thread(
                        face_matching,
                        doc_face_path,
                        payload.person_image_path,
                    )
                elif stage == VerificationStage.ML:
                    if rectified_path is None:
                        raise RuntimeError("Rectified image not available")
                    src = rectified_path
                    result = await asyncio.to_thread(
                        ml_verify,
                        src,
                    )
                elif stage == VerificationStage.OCR:
                    if rectified_path is None:
                        raise RuntimeError("Rectified image not available")
                    src = rectified_path
                    result = await asyncio.to_thread(
                        ocr_verify,
                        src,
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
                        "result_data": json.dumps(result),
                    },
                )
            except Exception as exc:
                failure_reason_code = failure_reason_code or self._infer_failure_code(str(exc)) or "UNKNOWN_ERROR"
                error_message = self._arabic_message(failure_reason_code)
                results["failure_reason_code"] = failure_reason_code
                await self.steps.update_one(
                    step_id,
                    {
                        "status": VerificationStatus.FAILED.value,
                        "error_message": error_message,
                        "end_time": datetime.now(timezone.utc),
                    },
                )
                await self.verifications.update_one(
                    payload.verification_id,
                    {
                        "status": VerificationStatus.FAILED.value,
                        "current_stage": stage.value,
                        "error_message": error_message,
                        "end_time": datetime.now(timezone.utc),
                        "result_data": json.dumps(results),
                    },
                )
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

        ml_result = results.get(VerificationStage.ML.value) or {}
        ml_final_decision = ml_result.get("final_decision")

        ocr_done = VerificationStage.OCR.value in results

        blockchain_result = results.get(VerificationStage.BLOCKCHAIN.value) or {}
        blockchain_cid = blockchain_result.get("cid")

        results["SUMMARY"] = {
            "face_match": face_match,
            "layout_status": layout_status,
            "ml_final_decision": ml_final_decision,
            "ocr_done": ocr_done,
            "blockchain_cid": blockchain_cid,
        }

        await self.verifications.update_one(
            payload.verification_id,
            {
                "status": VerificationStatus.SUCCESS.value,
                "current_stage": VerificationStage.BLOCKCHAIN.value,
                "end_time": datetime.now(timezone.utc),
                "result_data": json.dumps(results),
            },
        )
