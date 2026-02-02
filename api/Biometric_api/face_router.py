from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

router = APIRouter(prefix="/face", tags=["face"])


def get_face_service():
    from ai.face_service import FaceService
    return FaceService()


@router.post("/verify-id-live")
async def verify_id_vs_live(
    id_photo: UploadFile = File(...),
    live_photo: UploadFile = File(...),
    service: "FaceService" = Depends(get_face_service),
):
    try:
        id_data = await id_photo.read()
        live_data = await live_photo.read()
        return service.verify_id_vs_live(id_data, live_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))