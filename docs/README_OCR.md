# OCR API (Image + PDF)

## Setup
pip install -r requirements.txt

## Config
Create a .env file (do not commit it):
GOOGLE_VISION_API_KEY=YOUR_KEY_HERE

## Run
uvicorn app.main:app --reload

## Test
http://127.0.0.1:8000/docs
