from fastapi import FastAPI
from api.ocr.routes_ocr import router as ocr_router

app = FastAPI(title="OCR API (Google Vision) - Image & PDF")
app.include_router(ocr_router)
