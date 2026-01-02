import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

# Try explicit load from project .env first (override existing empty values)
load_dotenv(dotenv_path=ENV_PATH, override=True)

# Fallback: let python-dotenv try to locate an env file if the above didn't work
if not os.getenv("GOOGLE_VISION_API_KEY"):
	alt = find_dotenv()
	if alt:
		load_dotenv(alt, override=False)

# Final fallback: attempt to parse the .env file manually (robustness on Windows)
if not os.getenv("GOOGLE_VISION_API_KEY"):
	try:
		with open(ENV_PATH, "r", encoding="utf-8") as f:
			for line in f:
				if line.strip().startswith("GOOGLE_VISION_API_KEY"):
					parts = line.split("=", 1)
					if len(parts) == 2:
						os.environ.setdefault("GOOGLE_VISION_API_KEY", parts[1].strip())
						break
	except Exception as e:
		logging.getLogger("core.config").debug("Failed to read .env manually: %s", e)

GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "").strip()
