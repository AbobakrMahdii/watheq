import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from api.services.audit_log_service import log_file_event

router = APIRouter(prefix="/face", tags=["face"])
logger = logging.getLogger("api.face")


def get_face_service():
    # lazy import to avoid heavy third-party imports at app startup
    from Biometric.face_service import FaceService

    return FaceService()


@router.post("/verify")
async def verify_face(
    request: Request,
    photo1: UploadFile = File(...),
    photo2: UploadFile = File(...),
    service: "FaceService" = Depends(get_face_service),
):
    try:
        if photo1 is None or photo2 is None:
            raise HTTPException(
                status_code=400, detail="Missing required files: photo1, photo2"
            )
        data1 = await photo1.read()
        data2 = await photo2.read()
        if not data1 or not data2:
            raise HTTPException(status_code=400, detail="Empty file upload")
        result = service.verify_id_vs_live(data1, data2)
        request.state.audit_logged = True
        await log_file_event(
            request,
            status="success",
            operation_type="Verify",
            module="face",
            file_name=",".join([p for p in [photo1.filename, photo2.filename] if p]),
            file_size=(len(data1) + len(data2)),
        )
        return result
    except HTTPException:
        request.state.audit_logged = True
        await log_file_event(
            request,
            status="failed",
            failure_reason="Validation error",
            operation_type="Verify",
            module="face",
            file_name=",".join([p for p in [photo1.filename, photo2.filename] if p]),
            file_size=(
                (len(data1) + len(data2))
                if "data1" in locals() and "data2" in locals()
                else None
            ),
        )
        raise
    except ValueError as e:
        # input/decoding errors (e.g., invalid image bytes)
        request.state.audit_logged = True
        await log_file_event(
            request,
            status="failed",
            failure_reason=str(e),
            operation_type="Verify",
            module="face",
            file_name=",".join([p for p in [photo1.filename, photo2.filename] if p]),
            file_size=(
                (len(data1) + len(data2))
                if "data1" in locals() and "data2" in locals()
                else None
            ),
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(
            "Face verify failed (%s, %s): %s", photo1.filename, photo2.filename, e
        )
        request.state.audit_logged = True
        await log_file_event(
            request,
            status="failed",
            failure_reason=f"{type(e).__name__}: {e}",
            operation_type="Verify",
            module="face",
            file_name=",".join([p for p in [photo1.filename, photo2.filename] if p]),
            file_size=(
                (len(data1) + len(data2))
                if "data1" in locals() and "data2" in locals()
                else None
            ),
        )
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}",
        )
