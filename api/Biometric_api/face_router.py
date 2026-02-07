from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

router = APIRouter(prefix="/face", tags=["face"])


def get_face_service():
    # lazy import to avoid heavy third-party imports at app startup
    from Biometric.face_service import FaceService
    return FaceService()


@router.post("/verify")
async def verify_face(
    photo1: UploadFile = File(...),
    photo2: UploadFile = File(...),
    service: "FaceService" = Depends(get_face_service),
):
    try:
        data1 = await photo1.read()
        data2 = await photo2.read()
        result = service.verify_faces(data1, data2)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))