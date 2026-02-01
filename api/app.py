import logging
import os
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from .routers.auth_router import router as auth_router
from .routers.admin_router import router as admin_router
from .routers.face_router import router as face_router
from .routers.ipfs_router import router as ipfs_router
from .routers.ledger_router import router as ledger_router
from .routers.ocr_router import router as ocr_router
from .routers.document_router import router as document_router
from .routers.file_upload_router import router as file_upload_router
from .routers.document_type_router import router as document_type_router
from .routers.admin_document_type_router import router as admin_document_type_router
from .routers.admin_audit_router import router as admin_audit_router
from .routers.verification_router import router as verification_router
from .routers.admin_verification_router import router as admin_verification_router
from .security import get_current_user, get_current_admin
from . import database as db_module
from .services.audit_log_service import log_request_event

logger = logging.getLogger("watheq.api")

def get_allowed_origins() -> list[str]:
    env = os.getenv("ENV", "development").lower()
    raw = os.getenv("ALLOWED_ORIGINS", "").strip()
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    if env == "production":
        return []
    return ["http://localhost:3000", "http://127.0.0.1:3000"]


app = FastAPI(
    title="Watheq Unified Backend API",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
    except Exception as exc:
        if not getattr(request.state, "audit_logged", False):
            await log_request_event(request, status="failed", failure_reason=str(exc))
            request.state.audit_logged = True
        raise

    if getattr(request.state, "audit_logged", False):
        return response

    if response.status_code >= 400:
        await log_request_event(request, status="failed")
    else:
        await log_request_event(request, status="success")
    return response

# Include all routers
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(document_type_router)
app.include_router(admin_audit_router)

# Service routers (require authenticated user via Bearer token)
app.include_router(face_router, prefix="/api/v1", dependencies=[Depends(get_current_user)])
app.include_router(ipfs_router, prefix="/api/v1", dependencies=[Depends(get_current_user)])
app.include_router(ledger_router, prefix="/api/v1", dependencies=[Depends(get_current_user)])
app.include_router(ocr_router, prefix="/api/v1", dependencies=[Depends(get_current_user)])
app.include_router(document_router, prefix="/api/v1", dependencies=[Depends(get_current_user)])
app.include_router(file_upload_router, prefix="/api/v1", dependencies=[Depends(get_current_user)])
app.include_router(admin_document_type_router)
app.include_router(verification_router)
app.include_router(admin_verification_router)


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


@app.exception_handler(HTTPException)
async def audit_http_exception_handler(request: Request, exc: HTTPException):
    if not getattr(request.state, "audit_logged", False):
        detail = exc.detail
        if isinstance(detail, dict):
            reason = detail.get("message") or str(detail)
        else:
            reason = str(detail)
        await log_request_event(request, status="failed", failure_reason=reason)
        request.state.audit_logged = True
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def audit_validation_exception_handler(request: Request, exc: RequestValidationError):
    if not getattr(request.state, "audit_logged", False):
        reason = exc.errors()[0].get("msg") if exc.errors() else "Validation error"
        await log_request_event(request, status="failed", failure_reason=reason)
        request.state.audit_logged = True
    return await request_validation_exception_handler(request, exc)


@app.on_event("startup")
async def startup_event():
    # connect DB
    try:
        await db_module.database.connect()
    except Exception:
        logger.exception("Database connection failed on startup")

    # ensure users table exists (simple schema)
    create_sql = """
    CREATE TABLE IF NOT EXISTS users (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      name VARCHAR(255),
      username VARCHAR(255) UNIQUE,
      email VARCHAR(255) UNIQUE,
      password VARCHAR(255),
      role VARCHAR(50)
    ) ENGINE=InnoDB;
    """
    try:
        await db_module.database.execute(create_sql)
    except Exception:
        # non-fatal on startup
        logger.exception("Failed to ensure users table exists")

    # ensure document_types table exists
    doc_types_sql = """
    CREATE TABLE IF NOT EXISTS document_types (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      name VARCHAR(255) UNIQUE NOT NULL,
      is_active BOOLEAN DEFAULT TRUE,
      requires_back_image BOOLEAN DEFAULT FALSE,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB;
    """
    try:
        await db_module.database.execute(doc_types_sql)
    except Exception:
        logger.exception("Failed to ensure document_types table exists")

    # ensure audit_logs table exists
    audit_logs_sql = """
    CREATE TABLE IF NOT EXISTS audit_logs (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      operation_id CHAR(36) NOT NULL UNIQUE,
      operation_type VARCHAR(50) NOT NULL,
      status VARCHAR(20) NOT NULL,
      failure_reason TEXT,
      user_id BIGINT,
      user_name VARCHAR(255),
      user_email VARCHAR(255),
      user_role VARCHAR(50),
      ip_address VARCHAR(45),
      user_agent TEXT,
      service VARCHAR(100),
      module VARCHAR(100),
      path VARCHAR(255),
      method VARCHAR(10),
      file_name VARCHAR(255),
      file_ext VARCHAR(20),
      file_size BIGINT,
      file_cid VARCHAR(255),
      file_url VARCHAR(255),
      extra_data JSON,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      INDEX idx_audit_created_at (created_at),
      INDEX idx_audit_user_id (user_id),
      INDEX idx_audit_operation_type (operation_type),
      INDEX idx_audit_status (status)
    ) ENGINE=InnoDB;
    """
    try:
        await db_module.database.execute(audit_logs_sql)
    except Exception:
        logger.exception("Failed to ensure audit_logs table exists")

    verifications_sql = """
    CREATE TABLE IF NOT EXISTS verifications (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      user_id BIGINT,
      document_type_id BIGINT,
      status VARCHAR(20) NOT NULL,
      current_stage VARCHAR(20),
      error_message TEXT,
      start_time TIMESTAMP NULL,
      end_time TIMESTAMP NULL,
      result_data JSON,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      INDEX idx_verifications_user_id (user_id),
      INDEX idx_verifications_status (status)
    ) ENGINE=InnoDB;
    """
    try:
        await db_module.database.execute(verifications_sql)
    except Exception:
        logger.exception("Failed to ensure verifications table exists")

    verification_steps_sql = """
    CREATE TABLE IF NOT EXISTS verification_steps (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      verification_id BIGINT NOT NULL,
      step_name VARCHAR(100),
      stage VARCHAR(20) NOT NULL,
      status VARCHAR(20) NOT NULL,
      error_message TEXT,
      start_time TIMESTAMP NULL,
      end_time TIMESTAMP NULL,
      result_data JSON,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      INDEX idx_steps_verification_id (verification_id),
      INDEX idx_steps_stage (stage)
    ) ENGINE=InnoDB;
    """
    try:
        await db_module.database.execute(verification_steps_sql)
    except Exception:
        logger.exception("Failed to ensure verification_steps table exists")

    # Best-effort migrations for verifications/verification_steps on existing databases.
    verification_alter_statements = [
        "ALTER TABLE verifications ADD COLUMN current_stage VARCHAR(20)",
        "ALTER TABLE verifications ADD COLUMN error_message TEXT",
        "ALTER TABLE verifications ADD COLUMN start_time TIMESTAMP NULL",
        "ALTER TABLE verifications ADD COLUMN end_time TIMESTAMP NULL",
        "ALTER TABLE verifications ADD COLUMN result_data JSON",
        "ALTER TABLE verifications ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "ALTER TABLE verifications ADD COLUMN document_type_id BIGINT",
    ]
    for stmt in verification_alter_statements:
        try:
            await db_module.database.execute(stmt)
        except Exception:
            pass

    verification_steps_alter_statements = [
        "ALTER TABLE verification_steps ADD COLUMN step_name VARCHAR(100)",
        "ALTER TABLE verification_steps ADD COLUMN stage VARCHAR(20)",
        "ALTER TABLE verification_steps ADD COLUMN status VARCHAR(20)",
        "ALTER TABLE verification_steps ADD COLUMN verification_id BIGINT",
        "ALTER TABLE verification_steps ADD COLUMN error_message TEXT",
        "ALTER TABLE verification_steps ADD COLUMN start_time TIMESTAMP NULL",
        "ALTER TABLE verification_steps ADD COLUMN end_time TIMESTAMP NULL",
        "ALTER TABLE verification_steps ADD COLUMN result_data JSON",
        "ALTER TABLE verification_steps ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "ALTER TABLE verification_steps MODIFY COLUMN step_name VARCHAR(100) NULL",
    ]
    for stmt in verification_steps_alter_statements:
        try:
            await db_module.database.execute(stmt)
        except Exception:
            pass

    # Best-effort migrations for audit_logs on existing databases.
    audit_alter_statements = [
        "ALTER TABLE audit_logs ADD COLUMN operation_id CHAR(36) NULL",
        "ALTER TABLE audit_logs ADD COLUMN operation_type VARCHAR(50) NOT NULL",
        "ALTER TABLE audit_logs ADD COLUMN status VARCHAR(20) NOT NULL",
        "ALTER TABLE audit_logs ADD COLUMN failure_reason TEXT",
        "ALTER TABLE audit_logs ADD COLUMN user_id BIGINT",
        "ALTER TABLE audit_logs ADD COLUMN user_name VARCHAR(255)",
        "ALTER TABLE audit_logs ADD COLUMN user_email VARCHAR(255)",
        "ALTER TABLE audit_logs ADD COLUMN user_role VARCHAR(50)",
        "ALTER TABLE audit_logs ADD COLUMN ip_address VARCHAR(45)",
        "ALTER TABLE audit_logs ADD COLUMN user_agent TEXT",
        "ALTER TABLE audit_logs ADD COLUMN service VARCHAR(100)",
        "ALTER TABLE audit_logs ADD COLUMN module VARCHAR(100)",
        "ALTER TABLE audit_logs ADD COLUMN path VARCHAR(255)",
        "ALTER TABLE audit_logs ADD COLUMN method VARCHAR(10)",
        "ALTER TABLE audit_logs ADD COLUMN file_name VARCHAR(255)",
        "ALTER TABLE audit_logs ADD COLUMN file_ext VARCHAR(20)",
        "ALTER TABLE audit_logs ADD COLUMN file_size BIGINT",
        "ALTER TABLE audit_logs ADD COLUMN file_cid VARCHAR(255)",
        "ALTER TABLE audit_logs ADD COLUMN file_url VARCHAR(255)",
        "ALTER TABLE audit_logs ADD COLUMN extra_data JSON",
        "ALTER TABLE audit_logs ADD COLUMN created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP",
    ]
    for stmt in audit_alter_statements:
        try:
            await db_module.database.execute(stmt)
        except Exception:
            # Ignore if column already exists
            pass

    audit_index_statements = [
        "CREATE INDEX idx_audit_created_at ON audit_logs (created_at)",
        "CREATE INDEX idx_audit_user_id ON audit_logs (user_id)",
        "CREATE INDEX idx_audit_operation_type ON audit_logs (operation_type)",
        "CREATE INDEX idx_audit_status ON audit_logs (status)",
        "CREATE UNIQUE INDEX idx_audit_operation_id ON audit_logs (operation_id)",
    ]
    for stmt in audit_index_statements:
        try:
            await db_module.database.execute(stmt)
        except Exception:
            # Ignore if index already exists
            pass

    # Backfill legacy columns if needed (best-effort).
    try:
        await db_module.database.execute(
            "UPDATE audit_logs SET created_at = `timestamp` WHERE created_at IS NULL"
        )
    except Exception:
        pass
    try:
        await db_module.database.execute(
            "UPDATE audit_logs SET operation_id = UUID() WHERE operation_id IS NULL OR operation_id = ''"
        )
    except Exception:
        pass


@app.on_event("shutdown")
async def shutdown_event():
    try:
        await db_module.database.disconnect()
    except Exception:
        logger.exception("Database disconnect failed on shutdown")
