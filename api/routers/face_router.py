import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

router = APIRouter(prefix="/face", tags=["face"])
logger = logging.getLogger("api.face")

def get_face_service():
    # lazy import to avoid heavy third-party imports at app startup
    from ai.Biometric.face_service import FaceService
    return FaceService()


@router.post("/verify")
async def verify_face(
    photo1: UploadFile = File(...),
    photo2: UploadFile = File(...),
    service: "FaceService" = Depends(get_face_service),
):
    try:
        if photo1 is None or photo2 is None:
            raise HTTPException(status_code=400, detail="Missing required files: photo1, photo2")
        data1 = await photo1.read()
        data2 = await photo2.read()
        if not data1 or not data2:
            raise HTTPException(status_code=400, detail="Empty file upload")
        result = service.verify_faces(data1, data2)
        return result
    except HTTPException:
        raise
    except ValueError as e:
        # input/decoding errors (e.g., invalid image bytes)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Face verify failed (%s, %s): %s", photo1.filename, photo2.filename, e)
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}",
        )
