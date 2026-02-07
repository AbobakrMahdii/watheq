from fastapi import APIRouter, UploadFile, File, HTTPException
from ocr.vision_service_ocr import ocr_image, ocr_pdf

router = APIRouter()

@router.post("/ocr")
async def ocr(file: UploadFile = File(...), max_pages: int = 10):
    content_type = (file.content_type or "").lower()
    data = await file.read()

    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    # Image
    if content_type.startswith("image/"):
        return ocr_image(data)

    # PDF
    if content_type in ["application/pdf", "application/x-pdf"] or (file.filename or "").lower().endswith(".pdf"):
        if max_pages < 1 or max_pages > 50:
            raise HTTPException(status_code=400, detail="max_pages must be between 1 and 50")
        return ocr_pdf(data, max_pages=max_pages)

    raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")
