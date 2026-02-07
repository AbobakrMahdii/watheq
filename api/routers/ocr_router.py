from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from ocr.vision_service_ocr import ocr_image, ocr_pdf
from api.services.audit_log_service import log_file_event

router = APIRouter()

@router.post("/ocr")
async def ocr(request: Request, file: UploadFile = File(...), max_pages: int = 10):
    content_type = (file.content_type or "").lower()
    data = await file.read()

    if not data:
        request.state.audit_logged = True
        await log_file_event(
            request,
            status="failed",
            failure_reason="Empty file",
            operation_type="OCR",
            module="ocr",
            file_name=file.filename,
            file_size=0,
        )
        raise HTTPException(status_code=400, detail="Empty file")

    # Image
    if content_type.startswith("image/"):
        request.state.audit_logged = True
        await log_file_event(
            request,
            status="success",
            operation_type="OCR",
            module="ocr",
            file_name=file.filename,
            file_size=len(data),
        )
        return ocr_image(data)

    # PDF
    if content_type in ["application/pdf", "application/x-pdf"] or (file.filename or "").lower().endswith(".pdf"):
        if max_pages < 1 or max_pages > 50:
            request.state.audit_logged = True
            await log_file_event(
                request,
                status="failed",
                failure_reason="max_pages must be between 1 and 50",
                operation_type="OCR",
                module="ocr",
                file_name=file.filename,
                file_size=len(data),
            )
            raise HTTPException(status_code=400, detail="max_pages must be between 1 and 50")
        request.state.audit_logged = True
        await log_file_event(
            request,
            status="success",
            operation_type="OCR",
            module="ocr",
            file_name=file.filename,
            file_size=len(data),
        )
        return ocr_pdf(data, max_pages=max_pages)

    request.state.audit_logged = True
    await log_file_event(
        request,
        status="failed",
        failure_reason=f"Unsupported file type: {content_type}",
        operation_type="OCR",
        module="ocr",
        file_name=file.filename,
        file_size=len(data),
    )
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")
