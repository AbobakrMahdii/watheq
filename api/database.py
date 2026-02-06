from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
import asyncio
import aiomysql
from databases import Database
from typing import AsyncIterator, Dict, Any, Optional

# Load environment variables from api/.env if present
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path, override=True)

# =========================
# Config
# =========================
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "watheq_db")

DB_USER_ESC = quote_plus(DB_USER)
DB_PASSWORD_ESC = quote_plus(DB_PASSWORD)
DATABASE_URL = f"mysql+aiomysql://{DB_USER_ESC}:{DB_PASSWORD_ESC}@{DB_HOST}/{DB_NAME}"
database = Database(DATABASE_URL)

# =========================
# Users Collection
# =========================
class UsersCollection:
    def __init__(self, db: Database):
        self.db = db

    async def _ensure_connected(self) -> None:
        if not self.db.is_connected:
            await self.db.connect()

    async def find_one(self, filt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        await self._ensure_connected()
        if not filt:
            return None
        if "_id" in filt:
            q = "SELECT id as _id, name, username, email, password, role FROM users WHERE id = :id"
            row = await self.db.fetch_one(q, values={"id": int(filt["_id"])})
            return dict(row) if row else None
        if "username" in filt:
            q = "SELECT id as _id, name, username, email, password, role FROM users WHERE username = :username"
            row = await self.db.fetch_one(q, values={"username": filt["username"]})
            return dict(row) if row else None
        if "email" in filt:
            q = "SELECT id as _id, name, username, email, password, role FROM users WHERE email = :email"
            row = await self.db.fetch_one(q, values={"email": filt["email"]})
            return dict(row) if row else None
        return None

    async def insert_one(self, doc: Dict[str, Any]) -> int:
        await self._ensure_connected()
        username = doc.get("username")
        if username:
            q = "INSERT INTO users (name, username, email, password, role) VALUES (:name, :username, :email, :password, :role)"
        else:
            q = "INSERT INTO users (name, email, password, role) VALUES (:name, :email, :password, :role)"
        return await self.db.execute(q, values=doc)

    async def update_one(self, filt: Dict[str, Any], update: Dict[str, Any]) -> int:
        await self._ensure_connected()
        if not filt:
            return 0
        if "_id" in filt:
            user_id = int(filt["_id"])
        else:
            return 0
        if "$set" in update:
            sets = update["$set"]
            parts = []
            values = {"id": user_id}
            i = 0
            for k, v in sets.items():
                param = f"val{i}"
                parts.append(f"{k} = :{param}")
                values[param] = v
                i += 1
            q = "UPDATE users SET " + ", ".join(parts) + " WHERE id = :id"
            await self.db.execute(q, values=values)
            return 1
        return 0

    def find(self, filt: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        async def _iter():
            await self._ensure_connected()
            if filt and "role" in filt:
                if isinstance(filt["role"], dict) and "$in" in filt["role"]:
                    vals = list(filt["role"]["$in"] or [])
                    if not vals:
                        rows = []
                    else:
                        placeholders = ", ".join([f":v{i}" for i in range(len(vals))])
                        q = (
                            "SELECT id as _id, name, username, email, password, role "
                            f"FROM users WHERE role IN ({placeholders})"
                        )
                        rows = await self.db.fetch_all(
                            q,
                            values={f"v{i}": vals[i] for i in range(len(vals))},
                        )
                else:
                    q = "SELECT id as _id, name, username, email, password, role FROM users WHERE role = :role"
                    rows = await self.db.fetch_all(q, values={"role": filt["role"]})
            else:
                q = "SELECT id as _id, name, username, email, password, role FROM users"
                rows = await self.db.fetch_all(q)
            for r in rows:
                yield dict(r)
        return _iter()

# Instance
users = UsersCollection(database)

# Function to get collection
def get_user_collection():
    return users

# =========================
# Document Types Collection
# =========================
class DocumentTypesCollection:
    def __init__(self, db: Database):
        self.db = db

    async def _ensure_connected(self) -> None:
        if not self.db.is_connected:
            await self.db.connect()

    async def find_one(self, filt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        await self._ensure_connected()
        if not filt:
            return None
        if "_id" in filt:
            q = "SELECT id as id, name, is_active, requires_back_image, created_at FROM document_types WHERE id = :id"
            row = await self.db.fetch_one(q, values={"id": int(filt["_id"])})
            return dict(row) if row else None
        if "name" in filt:
            q = "SELECT id as id, name, is_active, requires_back_image, created_at FROM document_types WHERE name = :name"
            row = await self.db.fetch_one(q, values={"name": filt["name"]})
            return dict(row) if row else None
        return None

    async def find(self, filt: Dict[str, Any] = None) -> list[Dict[str, Any]]:
        await self._ensure_connected()
        query_parts = []
        values = {}
        if filt and "is_active" in filt:
            query_parts.append("is_active = :is_active")
            values["is_active"] = filt["is_active"]

        q = "SELECT id as id, name, is_active, requires_back_image, created_at FROM document_types"
        if query_parts:
            q += " WHERE " + " AND ".join(query_parts)
        q += " ORDER BY name"
        rows = await self.db.fetch_all(q, values=values)
        return [dict(row) for row in rows]

    async def insert_one(self, doc: Dict[str, Any]) -> int:
        await self._ensure_connected()
        q = """
            INSERT INTO document_types (name, is_active, requires_back_image, created_at)
            VALUES (:name, :is_active, :requires_back_image, :created_at)
        """
        return await self.db.execute(q, values=doc)

    async def update_one(self, doc_id: int, update_data: Dict[str, Any]) -> int:
        await self._ensure_connected()
        set_parts = []
        values = {"id": doc_id}
        for key, value in update_data.items():
            set_parts.append(f"{key} = :{key}")
            values[key] = value
        
        if not set_parts:
            return 0 # No update to perform

        q = f"UPDATE document_types SET {', '.join(set_parts)} WHERE id = :id"
        return await self.db.execute(q, values=values)

    async def delete_one(self, doc_id: int) -> int:
        await self._ensure_connected()
        q = "DELETE FROM document_types WHERE id = :id"
        return await self.db.execute(q, values={"id": doc_id})

_document_types_collection: Optional[DocumentTypesCollection] = None

def get_document_type_collection() -> DocumentTypesCollection:
    global _document_types_collection
    if _document_types_collection is None:
        _document_types_collection = DocumentTypesCollection(database)
    return _document_types_collection

# =========================
# Audit Logs Collection
# =========================
class AuditLogsCollection:
    def __init__(self, db: Database):
        self.db = db

    async def _ensure_connected(self) -> None:
        if not self.db.is_connected:
            await self.db.connect()

    async def insert_one(self, doc: Dict[str, Any]) -> int:
        await self._ensure_connected()
        q = """
            INSERT INTO audit_logs (
                operation_id,
                operation_type,
                status,
                failure_reason,
                user_id,
                user_name,
                user_email,
                user_role,
                ip_address,
                user_agent,
                service,
                module,
                path,
                method,
                file_name,
                file_ext,
                file_size,
                file_cid,
                file_url,
                extra_data,
                created_at
            ) VALUES (
                :operation_id,
                :operation_type,
                :status,
                :failure_reason,
                :user_id,
                :user_name,
                :user_email,
                :user_role,
                :ip_address,
                :user_agent,
                :service,
                :module,
                :path,
                :method,
                :file_name,
                :file_ext,
                :file_size,
                :file_cid,
                :file_url,
                :extra_data,
                :created_at
            )
        """
        return await self.db.execute(q, values=doc)

    async def list(
        self,
        filters: Dict[str, Any],
        limit: int,
        offset: int,
    ) -> list[Dict[str, Any]]:
        await self._ensure_connected()
        where_parts = []
        values: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }

        if filters.get("user_id") is not None:
            where_parts.append("user_id = :user_id")
            values["user_id"] = filters["user_id"]
        if filters.get("user_name"):
            where_parts.append("user_name LIKE :user_name")
            values["user_name"] = f"%{filters['user_name']}%"
        if filters.get("user_email"):
            where_parts.append("user_email LIKE :user_email")
            values["user_email"] = f"%{filters['user_email']}%"
        if filters.get("operation_type"):
            where_parts.append("operation_type = :operation_type")
            values["operation_type"] = filters["operation_type"]
        if filters.get("status"):
            where_parts.append("status = :status")
            values["status"] = filters["status"]
        if filters.get("date_from"):
            where_parts.append("created_at >= :date_from")
            values["date_from"] = filters["date_from"]
        if filters.get("date_to"):
            where_parts.append("created_at <= :date_to")
            values["date_to"] = filters["date_to"]
        if filters.get("query"):
            where_parts.append(
                "(user_name LIKE :query OR user_email LIKE :query OR operation_type LIKE :query OR module LIKE :query OR service LIKE :query)"
            )
            values["query"] = f"%{filters['query']}%"

        q = """
            SELECT
                id,
                operation_id,
                operation_type,
                status,
                failure_reason,
                user_id,
                user_name,
                user_email,
                user_role,
                ip_address,
                user_agent,
                service,
                module,
                path,
                method,
                file_name,
                file_ext,
                file_size,
                file_cid,
                file_url,
                extra_data,
                created_at
            FROM audit_logs
        """
        if where_parts:
            q += " WHERE " + " AND ".join(where_parts)
        q += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"

        rows = await self.db.fetch_all(q, values=values)
        return _normalize_audit_rows(rows)

    async def count(self, filters: Dict[str, Any]) -> int:
        await self._ensure_connected()
        where_parts = []
        values: Dict[str, Any] = {}

        if filters.get("user_id") is not None:
            where_parts.append("user_id = :user_id")
            values["user_id"] = filters["user_id"]
        if filters.get("user_name"):
            where_parts.append("user_name LIKE :user_name")
            values["user_name"] = f"%{filters['user_name']}%"
        if filters.get("user_email"):
            where_parts.append("user_email LIKE :user_email")
            values["user_email"] = f"%{filters['user_email']}%"
        if filters.get("operation_type"):
            where_parts.append("operation_type = :operation_type")
            values["operation_type"] = filters["operation_type"]
        if filters.get("status"):
            where_parts.append("status = :status")
            values["status"] = filters["status"]
        if filters.get("date_from"):
            where_parts.append("created_at >= :date_from")
            values["date_from"] = filters["date_from"]
        if filters.get("date_to"):
            where_parts.append("created_at <= :date_to")
            values["date_to"] = filters["date_to"]
        if filters.get("query"):
            where_parts.append(
                "(user_name LIKE :query OR user_email LIKE :query OR operation_type LIKE :query OR module LIKE :query OR service LIKE :query)"
            )
            values["query"] = f"%{filters['query']}%"

        q = "SELECT COUNT(*) as total FROM audit_logs"
        if where_parts:
            q += " WHERE " + " AND ".join(where_parts)
        row = await self.db.fetch_one(q, values=values)
        return int(row["total"]) if row else 0

    async def list_all(self, filters: Dict[str, Any]) -> list[Dict[str, Any]]:
        await self._ensure_connected()
        where_parts = []
        values: Dict[str, Any] = {}

        if filters.get("user_id") is not None:
            where_parts.append("user_id = :user_id")
            values["user_id"] = filters["user_id"]
        if filters.get("user_name"):
            where_parts.append("user_name LIKE :user_name")
            values["user_name"] = f"%{filters['user_name']}%"
        if filters.get("user_email"):
            where_parts.append("user_email LIKE :user_email")
            values["user_email"] = f"%{filters['user_email']}%"
        if filters.get("operation_type"):
            where_parts.append("operation_type = :operation_type")
            values["operation_type"] = filters["operation_type"]
        if filters.get("status"):
            where_parts.append("status = :status")
            values["status"] = filters["status"]
        if filters.get("date_from"):
            where_parts.append("created_at >= :date_from")
            values["date_from"] = filters["date_from"]
        if filters.get("date_to"):
            where_parts.append("created_at <= :date_to")
            values["date_to"] = filters["date_to"]
        if filters.get("query"):
            where_parts.append(
                "(user_name LIKE :query OR user_email LIKE :query OR operation_type LIKE :query OR module LIKE :query OR service LIKE :query)"
            )
            values["query"] = f"%{filters['query']}%"

        q = """
            SELECT
                id,
                operation_id,
                operation_type,
                status,
                failure_reason,
                user_id,
                user_name,
                user_email,
                user_role,
                ip_address,
                user_agent,
                service,
                module,
                path,
                method,
                file_name,
                file_ext,
                file_size,
                file_cid,
                file_url,
                extra_data,
                created_at
            FROM audit_logs
        """
        if where_parts:
            q += " WHERE " + " AND ".join(where_parts)
        q += " ORDER BY created_at DESC"

        rows = await self.db.fetch_all(q, values=values)
        return _normalize_audit_rows(rows)


def _normalize_audit_rows(rows: list[Any]) -> list[Dict[str, Any]]:
    items = []
    for row in rows:
        item = dict(row)
        extra = item.get("extra_data")
        if isinstance(extra, str):
            try:
                item["extra_data"] = json.loads(extra)
            except Exception:
                item["extra_data"] = None
        items.append(item)
    return items


_audit_logs_collection: Optional[AuditLogsCollection] = None


def get_audit_log_collection() -> AuditLogsCollection:
    global _audit_logs_collection
    if _audit_logs_collection is None:
        _audit_logs_collection = AuditLogsCollection(database)
    return _audit_logs_collection

# =========================
# Verifications Collection
# =========================
class VerificationsCollection:
    def __init__(self, db: Database):
        self.db = db

    async def _ensure_connected(self) -> None:
        if not self.db.is_connected:
            await self.db.connect()

    async def insert_one(self, doc: Dict[str, Any]) -> int:
        await self._ensure_connected()
        q = """
            INSERT INTO verifications (
                user_id,
                document_type_id,
                status,
                current_stage,
                error_message,
                start_time,
                end_time,
                result_data
            ) VALUES (
                :user_id,
                :document_type_id,
                :status,
                :current_stage,
                :error_message,
                :start_time,
                :end_time,
                :result_data
            )
        """
        return await self.db.execute(q, values=doc)

    async def update_one(self, verification_id: int, update_data: Dict[str, Any]) -> int:
        await self._ensure_connected()
        if not update_data:
            return 0
        parts = []
        values = {"id": verification_id}
        for key, value in update_data.items():
            parts.append(f"{key} = :{key}")
            values[key] = value
        q = f"UPDATE verifications SET {', '.join(parts)} WHERE id = :id"
        return await self.db.execute(q, values=values)

    async def find_one(self, verification_id: int) -> Optional[Dict[str, Any]]:
        await self._ensure_connected()
        q = """
            SELECT
                id,
                user_id,
                document_type_id,
                status,
                current_stage,
                error_message,
                start_time,
                end_time,
                result_data,
                created_at
            FROM verifications
            WHERE id = :id
        """
        row = await self.db.fetch_one(q, values={"id": verification_id})
        if not row:
            return None
        return _normalize_verification_row(row)

    async def list_by_user(self, user_id: int, limit: int, offset: int) -> list[Dict[str, Any]]:
        await self._ensure_connected()
        q = """
            SELECT
                id,
                user_id,
                document_type_id,
                status,
                current_stage,
                error_message,
                start_time,
                end_time,
                result_data,
                created_at
            FROM verifications
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """
        rows = await self.db.fetch_all(q, values={"user_id": user_id, "limit": limit, "offset": offset})
        return [_normalize_verification_row(row) for row in rows]

    async def list_all(self, limit: int, offset: int) -> list[Dict[str, Any]]:
        await self._ensure_connected()
        q = """
            SELECT
                id,
                user_id,
                document_type_id,
                status,
                current_stage,
                error_message,
                start_time,
                end_time,
                result_data,
                created_at
            FROM verifications
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """
        rows = await self.db.fetch_all(q, values={"limit": limit, "offset": offset})
        return [_normalize_verification_row(row) for row in rows]

    async def count(self, user_id: Optional[int] = None) -> int:
        await self._ensure_connected()
        if user_id is not None:
            row = await self.db.fetch_one(
                "SELECT COUNT(*) as total FROM verifications WHERE user_id = :user_id",
                values={"user_id": user_id},
            )
        else:
            row = await self.db.fetch_one("SELECT COUNT(*) as total FROM verifications")
        return int(row["total"]) if row else 0

    async def count_by_status(self, user_id: int) -> Dict[str, int]:
        await self._ensure_connected()
        q = """
            SELECT status, COUNT(*) as total
            FROM verifications
            WHERE user_id = :user_id
            GROUP BY status
        """
        rows = await self.db.fetch_all(q, values={"user_id": user_id})
        return {row["status"]: int(row["total"]) for row in rows}


class VerificationStepsCollection:
    def __init__(self, db: Database):
        self.db = db

    async def _ensure_connected(self) -> None:
        if not self.db.is_connected:
            await self.db.connect()

    async def insert_one(self, doc: Dict[str, Any]) -> int:
        await self._ensure_connected()
        q = """
            INSERT INTO verification_steps (
                verification_id,
                step_name,
                stage,
                status,
                error_message,
                start_time,
                end_time,
                result_data
            ) VALUES (
                :verification_id,
                :step_name,
                :stage,
                :status,
                :error_message,
                :start_time,
                :end_time,
                :result_data
            )
        """
        return await self.db.execute(q, values=doc)

    async def update_one(self, step_id: int, update_data: Dict[str, Any]) -> int:
        await self._ensure_connected()
        if not update_data:
            return 0
        parts = []
        values = {"id": step_id}
        for key, value in update_data.items():
            parts.append(f"{key} = :{key}")
            values[key] = value
        q = f"UPDATE verification_steps SET {', '.join(parts)} WHERE id = :id"
        return await self.db.execute(q, values=values)

    async def list_by_verification(self, verification_id: int) -> list[Dict[str, Any]]:
        await self._ensure_connected()
        q = """
            SELECT
                id,
                verification_id,
                stage,
                status,
                error_message,
                start_time,
                end_time,
                result_data,
                created_at
            FROM verification_steps
            WHERE verification_id = :verification_id
            ORDER BY id ASC
        """
        rows = await self.db.fetch_all(q, values={"verification_id": verification_id})
        items = []
        for row in rows:
            item = dict(row)
            item["result_data"] = _parse_json_field(item.get("result_data"))
            items.append(item)
        return items


_verifications_collection: Optional[VerificationsCollection] = None
_verification_steps_collection: Optional[VerificationStepsCollection] = None


def get_verifications_collection() -> VerificationsCollection:
    global _verifications_collection
    if _verifications_collection is None:
        _verifications_collection = VerificationsCollection(database)
    return _verifications_collection


def get_verification_steps_collection() -> VerificationStepsCollection:
    global _verification_steps_collection
    if _verification_steps_collection is None:
        _verification_steps_collection = VerificationStepsCollection(database)
    return _verification_steps_collection


def _parse_json_field(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return value


def _normalize_verification_row(row: Any) -> Dict[str, Any]:
    item = dict(row)
    item["result_data"] = _parse_json_field(item.get("result_data"))
    return item

# =========================
# Initialize DB + tables
# =========================
async def init_db():
    # Connect to MySQL server without specifying DB
    conn = await aiomysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    async with conn.cursor() as cur:
        await cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME};")
    conn.close()

    # Connect using databases
    await database.connect()

    # Create users table if not exists
    query = """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        username VARCHAR(100) UNIQUE,
        email VARCHAR(100) NOT NULL UNIQUE,
        password VARCHAR(255) NOT NULL,
        role VARCHAR(20) NOT NULL
    );
    """
    await database.execute(query)

    # Try to add username column on existing deployments (ignore if already exists)
    try:
        await database.execute("ALTER TABLE users ADD COLUMN username VARCHAR(100) UNIQUE;")
    except Exception:
        pass

    # Create document_types table if not exists
    query = """
    CREATE TABLE IF NOT EXISTS document_types (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) UNIQUE NOT NULL,
        is_active BOOLEAN DEFAULT TRUE,
        requires_back_image BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    await database.execute(query)

    # Create audit_logs table if not exists
    audit_query = """
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
    );
    """
    await database.execute(audit_query)

    # Create verifications table if not exists
    verification_query = """
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
    );
    """
    await database.execute(verification_query)

    # Create verification_steps table if not exists
    steps_query = """
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
    );
    """
    await database.execute(steps_query)




