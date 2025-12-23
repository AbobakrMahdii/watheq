import os
import base64
import requests
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(title="OCR API (Google Vision) - Image & PDF")

API_KEY = "AIzaSyBowuRTs9j25JIElBgtjBehuptsspeJcVQ"

VISION_URL = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"

def google_vision_text_detection(image_bytes: bytes) -> dict:
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "requests": [
            {
                "image": {"content": img_b64},
                "features": [{"type": "TEXT_DETECTION"}],
            }
        ]
    }
    res = requests.post(VISION_URL, json=payload, timeout=30)
    if res.status_code != 200:
        raise HTTPException(status_code=502, detail={"google_status": res.status_code, "body": res.text})
    return res.json()

def extract_text_from_vision_response(data: dict) -> str | None:
    try:
        return data["responses"][0]["fullTextAnnotation"]["text"]
    except Exception:
        return None

def pdf_to_png_pages(pdf_bytes: bytes, max_pages: int = 10, zoom: float = 2.0) -> list[bytes]:
    """
    zoom=2.0 تقريبًا يعطي دقة جيدة (حوالي 144-200 DPI حسب الإعدادات).
    max_pages لتفادي ملفات ضخمة/زمن طويل.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    page_count = min(doc.page_count, max_pages)

    matrix = fitz.Matrix(zoom, zoom)
    for i in range(page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pages.append(pix.tobytes("png"))
    doc.close()
    return pages

@app.post("/ocr")
async def ocr(file: UploadFile = File(...), max_pages: int = 10):
    content_type = (file.content_type or "").lower()
    data_bytes = await file.read()
    if not data_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # صور مباشرة
    if content_type.startswith("image/"):
        vision_data = google_vision_text_detection(data_bytes)
        text = extract_text_from_vision_response(vision_data)
        return {
            "type": "image",
            "pages": 1,
            "text": text,
            "raw_if_no_text": None if text else vision_data
        }

    # PDF
    if content_type in ["application/pdf", "application/x-pdf"] or file.filename.lower().endswith(".pdf"):
        if max_pages < 1 or max_pages > 50:
            raise HTTPException(status_code=400, detail="max_pages must be between 1 and 50")

        try:
            page_images = pdf_to_png_pages(data_bytes, max_pages=max_pages, zoom=2.0)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")

        texts = []
        raws = []
        for idx, img_png in enumerate(page_images, start=1):
            vision_data = google_vision_text_detection(img_png)
            page_text = extract_text_from_vision_response(vision_data)
            texts.append(page_text or "")
            if not page_text:
                raws.append({"page": idx, "raw": vision_data})

        full_text = "\n".join(t for t in texts if t is not None).strip() or None

        return {
            "type": "pdf",
            "pages": len(page_images),
            "text": full_text,
            "page_texts": texts,     
            "raw_pages_without_text": raws
        }

    raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")
