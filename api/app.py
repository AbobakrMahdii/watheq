import logging
import os
import aiomysql
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from .routers.auth_router import router as auth_router
from .routers.admin_router import router as admin_router
from .routers.face_router import router as face_router
from .routers.ipfs_router import router as ipfs_router
from .routers.ocr_router import router as ocr_router
from .routers.document_router import router as document_router
from .routers.file_upload_router import router as file_upload_router
from .routers.document_type_router import router as document_type_router
from .routers.admin_document_type_router import router as admin_document_type_router
from .routers.admin_audit_router import router as admin_audit_router
from .routers.verification_router import router as verification_router
from .routers.admin_verification_router import router as admin_verification_router
from .routers.blockchain_router import router as blockchain_router
from .routers.biometric_router import router as biometric_router
from .routers.notification_router import router as notification_router
from .routers.admin_citizen_router import router as admin_citizen_router
from .security import get_current_user, get_current_admin, get_password_hash
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
app.include_router(
    face_router, prefix="/api/v1", dependencies=[Depends(get_current_user)]
)
app.include_router(
    ipfs_router, prefix="/api/v1", dependencies=[Depends(get_current_user)]
)
app.include_router(
    ocr_router, prefix="/api/v1", dependencies=[Depends(get_current_user)]
)
app.include_router(
    document_router, prefix="/api/v1", dependencies=[Depends(get_current_user)]
)
app.include_router(
    file_upload_router, prefix="/api/v1", dependencies=[Depends(get_current_user)]
)
app.include_router(admin_document_type_router)
app.include_router(verification_router)
app.include_router(admin_verification_router)
app.include_router(blockchain_router)
app.include_router(biometric_router)
app.include_router(notification_router)
app.include_router(admin_citizen_router)


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
async def audit_validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    if not getattr(request.state, "audit_logged", False):
        reason = exc.errors()[0].get("msg") if exc.errors() else "Validation error"
        await log_request_event(request, status="failed", failure_reason=reason)
        request.state.audit_logged = True
    return await request_validation_exception_handler(request, exc)


@app.on_event("startup")
async def startup_event():
    # Auto-create the database if it doesn't exist yet
    try:
        conn = await aiomysql.connect(
            host=db_module.DB_HOST,
            port=db_module.DB_PORT,
            user=db_module.DB_USER,
            password=db_module.DB_PASSWORD,
        )
        async with conn.cursor() as cur:
            await cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_module.DB_NAME}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.close()
        logger.info("Ensured database '%s' exists", db_module.DB_NAME)
    except Exception:
        logger.exception("Failed to auto-create database '%s'", db_module.DB_NAME)

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
      role VARCHAR(50),
      is_active BOOLEAN DEFAULT TRUE,
      deleted_at TIMESTAMP NULL
    ) ENGINE=InnoDB;
    """
    try:
        await db_module.database.execute(create_sql)
    except Exception:
        # non-fatal on startup
        logger.exception("Failed to ensure users table exists")
    # Best-effort add columns on existing deployments
    for stmt in [
        "ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT TRUE",
        "ALTER TABLE users ADD COLUMN deleted_at TIMESTAMP NULL",
    ]:
        try:
            await db_module.database.execute(stmt)
        except Exception:
            pass

    # Seed default super admin if users table is empty (first run)
    try:
        row = await db_module.database.fetch_one("SELECT COUNT(*) AS cnt FROM users")
        if row and row["cnt"] == 0:
            await db_module.database.execute(
                """
                INSERT INTO users (name, username, email, password, role, is_active)
                VALUES (:name, :username, :email, :password, :role, :is_active)
                """,
                values={
                    "name": "Super Admin",
                    "username": "admin",
                    "email": "admin@admin.admin",
                    "password": get_password_hash("pass1234"),
                    "role": "super_admin",
                    "is_active": True,
                },
            )
            logger.info("Default super admin seeded (admin@admin.admin)")
    except Exception:
        logger.exception("Failed to seed default super admin")

    # ensure document_types table exists
    doc_types_sql = """
    CREATE TABLE IF NOT EXISTS document_types (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      name VARCHAR(255) UNIQUE NOT NULL,
      folder_name VARCHAR(255) NOT NULL DEFAULT 'identity',
      is_active BOOLEAN DEFAULT TRUE,
      requires_back_image BOOLEAN DEFAULT FALSE,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB;
    """
    try:
        await db_module.database.execute(doc_types_sql)
    except Exception:
        logger.exception("Failed to ensure document_types table exists")
    # Best-effort add folder_name on existing deployments
    try:
        await db_module.database.execute(
            "ALTER TABLE document_types ADD COLUMN folder_name VARCHAR(255) NOT NULL DEFAULT 'identity'"
        )
    except Exception:
        pass

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
      status VARCHAR(50) NOT NULL,
      current_stage VARCHAR(50),
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
      stage VARCHAR(50) NOT NULL,
      status VARCHAR(50) NOT NULL,
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

    # ensure document_hashes table exists
    document_hashes_sql = """
    CREATE TABLE IF NOT EXISTS document_hashes (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      document_id CHAR(36) NOT NULL,
      hash CHAR(64) NOT NULL UNIQUE,
      ipfs_cid VARCHAR(255) NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      INDEX idx_hash (hash),
      INDEX idx_doc_id (document_id)
    ) ENGINE=InnoDB;
    """
    try:
        await db_module.database.execute(document_hashes_sql)
    except Exception:
        logger.exception("Failed to ensure document_hashes table exists")

    # ensure biometric_audit_log table exists
    biometric_audit_sql = """
    CREATE TABLE IF NOT EXISTS biometric_audit_log (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      user_id BIGINT NOT NULL,
      document_id VARCHAR(255) NOT NULL,
      liveness_result VARCHAR(50) NOT NULL,
      match_result BOOLEAN NOT NULL,
      confidence_score DOUBLE NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      INDEX idx_bio_user (user_id),
      INDEX idx_bio_doc (document_id)
    ) ENGINE=InnoDB;
    """
    try:
        await db_module.database.execute(biometric_audit_sql)
    except Exception:
        logger.exception("Failed to ensure biometric_audit_log table exists")

    # ensure citizen_records table exists
    citizen_records_sql = """
    CREATE TABLE IF NOT EXISTS citizen_records (
      id INT AUTO_INCREMENT PRIMARY KEY,
      national_id VARCHAR(20) NOT NULL UNIQUE,
      full_name_ar VARCHAR(255),
      full_name_en VARCHAR(255),
      date_of_birth DATE,
      address TEXT,
      issue_date DATE,
      expiry_date DATE,
      gender VARCHAR(10),
      nationality VARCHAR(100),
      document_type VARCHAR(50),
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP NULL ON UPDATE CURRENT_TIMESTAMP,
      INDEX idx_citizen_national_id (national_id)
    ) ENGINE=InnoDB;
    """
    try:
        await db_module.database.execute(citizen_records_sql)
    except Exception:
        logger.exception("Failed to ensure citizen_records table exists")

    # ensure verification_notes table exists
    verification_notes_sql = """
    CREATE TABLE IF NOT EXISTS verification_notes (
      id INT AUTO_INCREMENT PRIMARY KEY,
      verification_id BIGINT NOT NULL,
      admin_id BIGINT NOT NULL,
      note_text TEXT NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      INDEX idx_notes_verification (verification_id),
      INDEX idx_notes_admin (admin_id)
    ) ENGINE=InnoDB;
    """
    try:
        await db_module.database.execute(verification_notes_sql)
    except Exception:
        logger.exception("Failed to ensure verification_notes table exists")

    # ensure notifications table exists (for admin failed-verification alerts)
    notifications_sql = """
    CREATE TABLE IF NOT EXISTS notifications (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      verification_id BIGINT NOT NULL,
      message TEXT NOT NULL,
      document_type_name VARCHAR(255),
      user_name VARCHAR(255),
      failure_stage VARCHAR(50),
      failure_reason_code VARCHAR(100),
      is_read BOOLEAN DEFAULT FALSE,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      INDEX idx_notif_is_read (is_read),
      INDEX idx_notif_created (created_at),
      INDEX idx_notif_verification (verification_id)
    ) ENGINE=InnoDB;
    """
    try:
        await db_module.database.execute(notifications_sql)
    except Exception:
        logger.exception("Failed to ensure notifications table exists")

    # Best-effort migrations for verifications/verification_steps on existing databases.
    verification_alter_statements = [
        "ALTER TABLE verifications MODIFY COLUMN current_stage VARCHAR(50)",
        "ALTER TABLE verifications MODIFY COLUMN status VARCHAR(50) NOT NULL",
        "ALTER TABLE verifications ADD COLUMN current_stage VARCHAR(50)",
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
        "ALTER TABLE verification_steps MODIFY COLUMN stage VARCHAR(50) NOT NULL",
        "ALTER TABLE verification_steps MODIFY COLUMN status VARCHAR(50) NOT NULL",
        "ALTER TABLE verification_steps ADD COLUMN step_name VARCHAR(100)",
        "ALTER TABLE verification_steps ADD COLUMN stage VARCHAR(50)",
        "ALTER TABLE verification_steps ADD COLUMN status VARCHAR(50)",
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

    # Backfill legacy operation_id if needed (best-effort).
    try:
        await db_module.database.execute(
            "UPDATE audit_logs SET operation_id = UUID() WHERE operation_id IS NULL OR operation_id = ''"
        )
    except Exception:
        pass

    # ── Recover stuck verifications (PENDING / RUNNING) from a previous crash ──
    # When the server stops (or crashes) while background tasks are processing
    # verifications, those records stay in PENDING or RUNNING forever. Mark them
    # as FAILED so users see the result and can retry.
    try:
        stuck = await db_module.database.fetch_one(
            "SELECT COUNT(*) AS cnt FROM verifications WHERE status IN ('PENDING', 'RUNNING')"
        )
        stuck_count = stuck["cnt"] if stuck else 0
        if stuck_count > 0:
            await db_module.database.execute(
                """
                UPDATE verifications
                   SET status       = 'FAILED',
                       error_message = 'توقف الخادم أثناء التحقق — يرجى إعادة المحاولة',
                       end_time      = NOW()
                 WHERE status IN ('PENDING', 'RUNNING')
                """
            )
            # Also mark any RUNNING steps as FAILED so the UI pipeline view is consistent
            await db_module.database.execute(
                """
                UPDATE verification_steps
                   SET status        = 'FAILED',
                       error_message = 'Server restarted',
                       end_time      = NOW()
                 WHERE status = 'RUNNING'
                """
            )
            logger.warning(
                "Recovered %d stuck verification(s) from previous server session",
                stuck_count,
            )
    except Exception:
        logger.exception("Failed to recover stuck verifications")


@app.on_event("shutdown")
async def shutdown_event():
    try:
        await db_module.database.disconnect()
    except Exception:
        logger.exception("Database disconnect failed on shutdown")
