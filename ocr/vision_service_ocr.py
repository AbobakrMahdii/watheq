import base64
import requests
import fitz  # PyMuPDF
from fastapi import HTTPException
from core.config import GOOGLE_VISION_API_KEY

if not GOOGLE_VISION_API_KEY:
    raise RuntimeError("Missing env var GOOGLE_VISION_API_KEY (check .env)")

VISION_URL = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"

def _google_vision_text_detection(image_bytes: bytes) -> dict:
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
        raise HTTPException(
            status_code=502,
            detail={"google_status": res.status_code, "body": res.text},
        )
    return res.json()

def _extract_text(data: dict) -> str | None:
    try:
        return data["responses"][0]["fullTextAnnotation"]["text"]
    except Exception:
        return None

def _pdf_to_png_pages(pdf_bytes: bytes, max_pages: int = 10, zoom: float = 2.0) -> list[bytes]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[bytes] = []
    page_count = min(doc.page_count, max_pages)
    matrix = fitz.Matrix(zoom, zoom)

    for i in range(page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pages.append(pix.tobytes("png"))

    doc.close()
    return pages

def ocr_image(image_bytes: bytes) -> dict:
    vision_data = _google_vision_text_detection(image_bytes)
    text = _extract_text(vision_data)
    return {
        "type": "image",
        "pages": 1,
        "text": text,
        "raw_if_no_text": None if text else vision_data,
    }

def ocr_pdf(pdf_bytes: bytes, max_pages: int = 10) -> dict:
    page_images = _pdf_to_png_pages(pdf_bytes, max_pages=max_pages, zoom=2.0)

    texts: list[str] = []
    raws: list[dict] = []

    for idx, img_png in enumerate(page_images, start=1):
        vision_data = _google_vision_text_detection(img_png)
        page_text = _extract_text(vision_data)
        texts.append(page_text or "")
        if not page_text:
            raws.append({"page": idx, "raw": vision_data})

    full_text = "\n".join(texts).strip() or None

    return {
        "type": "pdf",
        "pages": len(page_images),
        "text": full_text,
        "page_texts": texts,
        "raw_pages_without_text": raws,
    }
