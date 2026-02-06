from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from ai.Biometric.face_service import FaceService

router = APIRouter(prefix="/face", tags=["face"])

def get_face_service():
    return FaceService()

@router.post("/verify-id-live")
async def verify_id_vs_live(
    photo1: UploadFile = File(...),
    photo2: UploadFile = File(...),
    service: FaceService = Depends(get_face_service),
):
    try:
        p1 = await photo1.read()
        p2 = await photo2.read()
        return service.verify_id_vs_live(p1, p2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))