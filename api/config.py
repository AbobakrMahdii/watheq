import os
from secrets import token_urlsafe

ENV = os.getenv("ENV", "development").lower()

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    if ENV == "production":
        raise RuntimeError("SECRET_KEY must be set in production")
    # Dev-only fallback to keep local setup working
    SECRET_KEY = token_urlsafe(32)

ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
