import logging
from pathlib import Path

import uvicorn

from api.app import app

# ── Logging setup ───────────────────────────────────────────────────
_API_DIR = Path(__file__).resolve().parent
_LOG_FILE = _API_DIR / "runtime_api.log"
_ERR_FILE = _API_DIR / "runtime_api.err"

_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

# File handler — all INFO+ messages
_fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_fh.setLevel(logging.INFO)
_fh.setFormatter(_fmt)

# Error file handler — WARNING+ only
_eh = logging.FileHandler(_ERR_FILE, encoding="utf-8")
_eh.setLevel(logging.WARNING)
_eh.setFormatter(_fmt)

# Apply to root so every logger (uvicorn, fastapi, watheq.*) writes here
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), _fh, _eh])


if __name__ == "__main__":
    # Default port aligned with dashboard BACKEND_BASE_URL
    uvicorn.run(app, host="0.0.0.0", port=8012)
