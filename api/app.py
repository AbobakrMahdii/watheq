from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from .routers.auth_router import router as auth_router
from .routers.admin_router import router as admin_router
from .routers.face_router import router as face_router
from .routers.ipfs_router import router as ipfs_router
from .routers.ledger_router import router as ledger_router
from .routers.ocr_router import router as ocr_router
from .security import get_current_user
from . import database as db_module


app = FastAPI(
    title="Watheq Unified Backend API",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(auth_router)
app.include_router(admin_router)

# Service routers (require authenticated user via Bearer token)
app.include_router(face_router, prefix="/api/v1", dependencies=[Depends(get_current_user)])
app.include_router(ipfs_router, prefix="/api/v1", dependencies=[Depends(get_current_user)])
app.include_router(ledger_router, prefix="/api/v1", dependencies=[Depends(get_current_user)])
app.include_router(ocr_router, prefix="/api/v1", dependencies=[Depends(get_current_user)])


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )

    # Add Bearer auth scheme for Swagger UI
    openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})
    openapi_schema["components"]["securitySchemes"]["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
    }

    # Protect service paths under /api/v1 (but not /api/v1/auth)
    for path, path_item in openapi_schema.get("paths", {}).items():
        if path.startswith("/api/v1/") and not path.startswith("/api/v1/auth"):
            for operation in path_item.values():
                if isinstance(operation, dict):
                    operation.setdefault("security", []).append({"BearerAuth": []})

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.on_event("startup")
async def startup_event():
    # connect DB
    try:
        await db_module.database.connect()
    except Exception:
        pass

    # ensure users table exists (simple schema)
    create_sql = """
    CREATE TABLE IF NOT EXISTS users (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      name VARCHAR(255),
      email VARCHAR(255) UNIQUE,
      password VARCHAR(255),
      role VARCHAR(50)
    ) ENGINE=InnoDB;
    """
    try:
        await db_module.database.execute(create_sql)
    except Exception:
        # non-fatal on startup
        pass


@app.on_event("shutdown")
async def shutdown_event():
    try:
        await db_module.database.disconnect()
    except Exception:
        pass