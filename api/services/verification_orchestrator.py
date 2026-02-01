from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from api.database import get_verifications_collection, get_verification_steps_collection
from api.models import VerificationStage, VerificationStatus
from api.services.verification_steps_service import (
    biometric_verify,
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

    async def run(self, payload: VerificationInput) -> None:
        started_at = datetime.now(timezone.utc)
        await self.verifications.update_one(
            payload.verification_id,
            {
                "status": VerificationStatus.RUNNING.value,
                "current_stage": VerificationStage.BIOMETRIC.value,
                "start_time": started_at,
                "error_message": None,
            },
        )

        results: dict[str, Any] = {}

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
                    result = await asyncio.to_thread(
                        biometric_verify,
                        payload.document_front_path,
                        payload.person_image_path,
                    )
                elif stage == VerificationStage.DOCUMENT_IMAGE_QUALITY:
                    result = await asyncio.to_thread(
                        document_image_quality,
                        payload.document_front_path,
                    )
                    if not result.get("brightness_ok", True):
                        raise RuntimeError("الإضاءة غير مناسبة")
                    if not result.get("blur_ok", True):
                        raise RuntimeError("الصورة غير واضحة")
                elif stage == VerificationStage.DOCUMENT_CROPPING:
                    cropped_path = (
                        payload.document_front_path.parent / "document_cropped.jpg"
                    )
                    result = await asyncio.to_thread(
                        document_crop,
                        payload.document_front_path,
                        cropped_path,
                    )
                elif stage == VerificationStage.DOCUMENT_FACE_EXTRACTION:
                    if cropped_path is None:
                        raise RuntimeError("Document crop not available")
                    doc_face_path = (
                        payload.document_front_path.parent / "document_face.jpg"
                    )
                    result = await asyncio.to_thread(
                        document_face_extraction,
                        cropped_path,
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
                    src = cropped_path or payload.document_front_path
                    result = await asyncio.to_thread(
                        ml_verify,
                        src,
                    )
                elif stage == VerificationStage.OCR:
                    src = cropped_path or payload.document_front_path
                    result = await asyncio.to_thread(
                        ocr_verify,
                        src,
                    )
                else:
                    src = cropped_path or payload.document_front_path
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
                error_message = str(exc)
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

        await self.verifications.update_one(
            payload.verification_id,
            {
                "status": VerificationStatus.SUCCESS.value,
                "current_stage": VerificationStage.BLOCKCHAIN.value,
                "end_time": datetime.now(timezone.utc),
                "result_data": json.dumps(results),
            },
        )
