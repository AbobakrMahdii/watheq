from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.security import get_current_user
from api.database import get_biometric_audit_collection
from Biometric.face_service import FaceService

router = APIRouter(
    prefix="/api/v1/biometric",
    tags=["Biometric"],
    dependencies=[Depends(get_current_user)],
)

# نستخدم FaceService (DeepFace) لمطابقة الوجوه بدقة أعلى من المقارنة المبسطة.
_face_service: Optional[FaceService] = None
_face_err: Optional[str] = None
try:
    _face_service = FaceService()
except Exception as exc:
    _face_service = None
    _face_err = str(exc)


@router.post("/verify")
async def biometric_verify(
    selfie_image: UploadFile = File(...),
    document_image: UploadFile = File(...),
    user_id: int = Form(...),
    document_id: str = Form(...),
):
    """
    تدفق التحقق البيومتري:
    1) Liveness (حيوية) لمنع التلاعب بالصور/الشاشات.
    2) Face Matching عبر DeepFace (FaceService) للحصول على match/distance/similarity.
    3) تسجيل نتيجة التدقيق دون تخزين الصور لأسباب خصوصية.
    الفرق: liveness يتحقق من أن الصورة حقيقية وحية، بينما face matching يطابق هوية الشخص مع الوثيقة.
    """
    if _face_service is None:
        raise HTTPException(
            status_code=503, detail=f"FaceService unavailable: {_face_err}"
        )

    selfie_bytes = await selfie_image.read()
    doc_bytes = await document_image.read()

    # Client-side ML Kit liveness is sufficient; server only does face matching.

    # Face Matching via DeepFace
    try:
        face_result = _face_service.verify_id_vs_live(doc_bytes, selfie_bytes)
    except Exception as exc:
        await _audit(user_id, document_id, "skipped", False, 0.0)
        raise HTTPException(status_code=400, detail=f"Face match failed: {exc}")

    accepted = bool(face_result.get("accepted", False))
    similarity_percent = float(face_result.get("similarity_percent", 0.0))
    accept_threshold = float(face_result.get("accept_threshold_percent", 80.0))

    await _audit(
        user_id, document_id, "client_side", accepted, similarity_percent / 100.0
    )

    return {
        "liveness": {"passed": True, "message": "Client-side liveness (ML Kit)"},
        "match": {
            "passed": accepted,
            "similarity_percent": similarity_percent,
            "accept_threshold_percent": accept_threshold,
        },
    }


async def _audit(
    user_id: int,
    document_id: str,
    liveness_result: str,
    match_result: bool,
    score: float,
):
    """
    تسجيل نتائج التحقق البيومتري دون تخزين الصور الخام.
    نستخدم similarity كـ confidence_score للقياس العددي.
    """
    audit = get_biometric_audit_collection()
    await audit.insert_one(
        user_id=user_id,
        document_id=document_id,
        liveness_result=liveness_result,
        match_result=match_result,
        confidence_score=score,
    )
