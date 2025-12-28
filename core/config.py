import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

# مهم: override=True عشان لو المتغير موجود وفاضي يستبدله
load_dotenv(dotenv_path=ENV_PATH, override=True)

GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "").strip()
