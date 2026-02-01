from fastapi import APIRouter, File, UploadFile, Request

from api.services.file_upload_service import handle_file_upload

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    # File upload handling is centralized in the service layer (validation + audit logging).
    return await handle_file_upload(request, file, module="files")
