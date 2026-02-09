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
# Hashes / Documents table
# =========================
class DocumentHashesCollection:
    """
    تخزين بصمة الملف (SHA-256) مع CID لتفادي التكرار.
    نستخدم قاعدة البيانات لتسريع التحقق قبل النشر على البلوكشين.
    """

    def __init__(self, db: Database):
        self.db = db

    async def _ensure_connected(self) -> None:
        if not self.db.is_connected:
            await self.db.connect()

    async def find_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        await self._ensure_connected()
        row = await self.db.fetch_one(
            "SELECT document_id, hash, ipfs_cid, created_at FROM document_hashes WHERE hash = :h",
            values={"h": file_hash},
        )
        return dict(row) if row else None

    async def insert_one(self, document_id: str, file_hash: str, ipfs_cid: str) -> None:
        await self._ensure_connected()
        await self.db.execute(
            """
            INSERT INTO document_hashes (document_id, hash, ipfs_cid, created_at)
            VALUES (:document_id, :hash, :ipfs_cid, NOW())
            """,
            values={
                "document_id": document_id,
                "hash": file_hash,
                "ipfs_cid": ipfs_cid,
            },
        )


_document_hashes_collection: Optional[DocumentHashesCollection] = None


def get_document_hashes_collection() -> DocumentHashesCollection:
    global _document_hashes_collection
    if _document_hashes_collection is None:
        _document_hashes_collection = DocumentHashesCollection(database)
    return _document_hashes_collection


# =========================
# Biometric audit log table
# =========================
class BiometricAuditCollection:
    """
    تسجيل نتائج التحقق البيومتري (حيوية + تطابق الوجه) مع عدم تخزين الصور الخام.
    """

    def __init__(self, db: Database):
        self.db = db

    async def _ensure_connected(self) -> None:
        if not self.db.is_connected:
            await self.db.connect()

    async def insert_one(
        self,
        user_id: int,
        document_id: str,
        liveness_result: str,
        match_result: bool,
        confidence_score: float,
    ) -> None:
        await self._ensure_connected()
        await self.db.execute(
            """
            INSERT INTO biometric_audit_log (
                user_id, document_id, liveness_result,
                match_result, confidence_score, created_at
            ) VALUES (
                :user_id, :document_id, :liveness_result,
                :match_result, :confidence_score, NOW()
            )
            """,
            values={
                "user_id": user_id,
                "document_id": document_id,
                "liveness_result": liveness_result,
                "match_result": int(bool(match_result)),
                "confidence_score": confidence_score,
            },
        )


_biometric_audit_collection: Optional[BiometricAuditCollection] = None


def get_biometric_audit_collection() -> BiometricAuditCollection:
    global _biometric_audit_collection
    if _biometric_audit_collection is None:
        _biometric_audit_collection = BiometricAuditCollection(database)
    return _biometric_audit_collection


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
            q = """
                SELECT id as _id, name, username, email, password, role, is_active, deleted_at
                FROM users
                WHERE id = :id AND (deleted_at IS NULL)
            """
            row = await self.db.fetch_one(q, values={"id": int(filt["_id"])})
            return dict(row) if row else None
        if "username" in filt:
            q = """
                SELECT id as _id, name, username, email, password, role, is_active, deleted_at
                FROM users
                WHERE username = :username AND (deleted_at IS NULL)
            """
            row = await self.db.fetch_one(q, values={"username": filt["username"]})
            return dict(row) if row else None
        if "email" in filt:
            q = """
                SELECT id as _id, name, username, email, password, role, is_active, deleted_at
                FROM users
                WHERE email = :email AND (deleted_at IS NULL)
            """
            row = await self.db.fetch_one(q, values={"email": filt["email"]})
            return dict(row) if row else None
        return None

    async def insert_one(self, doc: Dict[str, Any]) -> int:
        await self._ensure_connected()
        username = doc.get("username")
        if username:
            q = """
                INSERT INTO users (name, username, email, password, role, is_active, deleted_at)
                VALUES (:name, :username, :email, :password, :role, :is_active, :deleted_at)
            """
        else:
            q = """
                INSERT INTO users (name, email, password, role, is_active, deleted_at)
                VALUES (:name, :email, :password, :role, :is_active, :deleted_at)
            """
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
                            """
                            SELECT id as _id, name, username, email, password, role, is_active, deleted_at
                            FROM users
                            WHERE role IN ("""
                            + placeholders
                            + """) AND (deleted_at IS NULL)
                        """
                        )
                        rows = await self.db.fetch_all(
                            q,
                            values={f"v{i}": vals[i] for i in range(len(vals))},
                        )
                else:
                    q = """
                        SELECT id as _id, name, username, email, password, role, is_active, deleted_at
                        FROM users
                        WHERE role = :role AND (deleted_at IS NULL)
                    """
                    rows = await self.db.fetch_all(q, values={"role": filt["role"]})
            else:
                q = """
                    SELECT id as _id, name, username, email, password, role, is_active, deleted_at
                    FROM users
                    WHERE (deleted_at IS NULL)
                """
                rows = await self.db.fetch_all(q)
            for r in rows:
                yield dict(r)

        return _iter()

    async def get_first_super_admin_id(self) -> Optional[str]:
        await self._ensure_connected()
        row = await self.db.fetch_one(
            """
            SELECT id as _id
            FROM users
            WHERE role = 'super_admin' AND (deleted_at IS NULL)
            ORDER BY id ASC
            LIMIT 1
            """
        )
        return str(row["_id"]) if row else None


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
            q = "SELECT id as id, name, folder_name, is_active, requires_back_image, created_at FROM document_types WHERE id = :id"
            row = await self.db.fetch_one(q, values={"id": int(filt["_id"])})
            return dict(row) if row else None
        if "name" in filt:
            q = "SELECT id as id, name, folder_name, is_active, requires_back_image, created_at FROM document_types WHERE name = :name"
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

        q = "SELECT id as id, name, folder_name, is_active, requires_back_image, created_at FROM document_types"
        if query_parts:
            q += " WHERE " + " AND ".join(query_parts)
        q += " ORDER BY name"
        rows = await self.db.fetch_all(q, values=values)
        return [dict(row) for row in rows]

    async def insert_one(self, doc: Dict[str, Any]) -> int:
        await self._ensure_connected()
        # Omit created_at so MySQL DEFAULT CURRENT_TIMESTAMP is used
        clean = {k: v for k, v in doc.items() if k != "created_at"}
        q = """
            INSERT INTO document_types (name, folder_name, is_active, requires_back_image)
            VALUES (:name, :folder_name, :is_active, :requires_back_image)
        """
        return await self.db.execute(q, values=clean)

    async def update_one(self, doc_id: int, update_data: Dict[str, Any]) -> int:
        await self._ensure_connected()
        set_parts = []
        values = {"id": doc_id}
        for key, value in update_data.items():
            set_parts.append(f"{key} = :{key}")
            values[key] = value

        if not set_parts:
            return 0  # No update to perform

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
        sort_by: str = "created_at",
        sort_order: str = "desc",
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

        ALLOWED_AUDIT_SORT_COLS = {
            "created_at",
            "user_name",
            "operation_type",
            "status",
            "module",
            "id",
        }
        safe_sort_by = sort_by if sort_by in ALLOWED_AUDIT_SORT_COLS else "created_at"
        safe_sort_order = "ASC" if str(sort_order).upper() == "ASC" else "DESC"
        q += f" ORDER BY {safe_sort_by} {safe_sort_order} LIMIT :limit OFFSET :offset"

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

    async def update_one(
        self, verification_id: int, update_data: Dict[str, Any]
    ) -> int:
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
                v.id,
                v.user_id,
                v.document_type_id,
                v.status,
                v.current_stage,
                v.error_message,
                v.start_time,
                v.end_time,
                v.result_data,
                v.created_at,
                u.name AS user_name, u.email AS user_email,
                dt.name AS document_type_name
            FROM verifications v
            LEFT JOIN users u ON v.user_id = u.id
            LEFT JOIN document_types dt ON v.document_type_id = dt.id
            WHERE v.id = :id
        """
        row = await self.db.fetch_one(q, values={"id": verification_id})
        if not row:
            return None
        return _normalize_verification_row(row)

    async def list_by_user(
        self, user_id: int, limit: int, offset: int
    ) -> list[Dict[str, Any]]:
        await self._ensure_connected()
        q = """
            SELECT
                v.id,
                v.user_id,
                v.document_type_id,
                v.status,
                v.current_stage,
                v.error_message,
                v.start_time,
                v.end_time,
                v.result_data,
                v.created_at,
                dt.name AS document_type_name
            FROM verifications v
            LEFT JOIN document_types dt ON v.document_type_id = dt.id
            WHERE v.user_id = :user_id
            ORDER BY v.created_at DESC
            LIMIT :limit OFFSET :offset
        """
        rows = await self.db.fetch_all(
            q, values={"user_id": user_id, "limit": limit, "offset": offset}
        )
        return [_normalize_verification_row(row) for row in rows]

    async def list_all(
        self, limit: int, offset: int, **filters
    ) -> list[Dict[str, Any]]:
        """List verifications with optional filters: status, document_type_id, user_id,
        date_from, date_to, search, sort_by, sort_order."""
        await self._ensure_connected()
        where_clauses = []
        values: Dict[str, Any] = {"limit": limit, "offset": offset}

        if filters.get("status"):
            where_clauses.append("v.status = :status")
            values["status"] = filters["status"]
        if filters.get("document_type_id"):
            where_clauses.append("v.document_type_id = :document_type_id")
            values["document_type_id"] = filters["document_type_id"]
        if filters.get("user_id"):
            where_clauses.append("v.user_id = :user_id")
            values["user_id"] = filters["user_id"]
        if filters.get("date_from"):
            where_clauses.append("v.created_at >= :date_from")
            values["date_from"] = filters["date_from"]
        if filters.get("date_to"):
            where_clauses.append("v.created_at <= :date_to")
            values["date_to"] = filters["date_to"]
        if filters.get("search"):
            where_clauses.append(
                "(u.name LIKE :search OR u.email LIKE :search OR v.current_stage LIKE :search)"
            )
            values["search"] = f"%{filters['search']}%"
        if filters.get("user_name"):
            where_clauses.append("u.name LIKE :user_name")
            values["user_name"] = f"%{filters['user_name']}%"
        if filters.get("operation_type"):
            where_clauses.append("v.current_stage = :operation_type")
            values["operation_type"] = filters["operation_type"]

        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        sort_by = filters.get("sort_by", "created_at")
        if sort_by not in ("created_at", "status", "id"):
            sort_by = "created_at"
        sort_order = "ASC" if filters.get("sort_order", "").upper() == "ASC" else "DESC"

        q = f"""
            SELECT
                v.id, v.user_id, v.document_type_id, v.status,
                v.current_stage, v.error_message, v.start_time,
                v.end_time, v.result_data, v.created_at,
                u.name AS user_name, u.email AS user_email,
                dt.name AS document_type_name
            FROM verifications v
            LEFT JOIN users u ON v.user_id = u.id
            LEFT JOIN document_types dt ON v.document_type_id = dt.id
            {where_sql}
            ORDER BY v.{sort_by} {sort_order}
            LIMIT :limit OFFSET :offset
        """
        rows = await self.db.fetch_all(q, values=values)
        return [_normalize_verification_row(row) for row in rows]

    async def list_for_export(
        self, limit: int, offset: int, **filters
    ) -> list[Dict[str, Any]]:
        """Export verifications with optional filters, including latest admin note."""
        await self._ensure_connected()
        where_clauses = []
        values: Dict[str, Any] = {"limit": limit, "offset": offset}

        if filters.get("status"):
            where_clauses.append("v.status = :status")
            values["status"] = filters["status"]
        if filters.get("document_type_id"):
            where_clauses.append("v.document_type_id = :document_type_id")
            values["document_type_id"] = filters["document_type_id"]
        if filters.get("user_id"):
            where_clauses.append("v.user_id = :user_id")
            values["user_id"] = filters["user_id"]
        if filters.get("date_from"):
            where_clauses.append("v.created_at >= :date_from")
            values["date_from"] = filters["date_from"]
        if filters.get("date_to"):
            where_clauses.append("v.created_at <= :date_to")
            values["date_to"] = filters["date_to"]
        if filters.get("search"):
            where_clauses.append(
                "(u.name LIKE :search OR u.email LIKE :search OR v.current_stage LIKE :search)"
            )
            values["search"] = f"%{filters['search']}%"
        if filters.get("user_name"):
            where_clauses.append("u.name LIKE :user_name")
            values["user_name"] = f"%{filters['user_name']}%"
        if filters.get("operation_type"):
            where_clauses.append("v.current_stage = :operation_type")
            values["operation_type"] = filters["operation_type"]

        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        q = f"""
            SELECT
                v.id,
                v.user_id,
                u.name AS user_name,
                u.email AS user_email,
                v.current_stage,
                v.status,
                v.created_at,
                n.note_text AS supervisor_note
            FROM verifications v
            LEFT JOIN users u ON v.user_id = u.id
            LEFT JOIN (
                SELECT vn1.verification_id, vn1.note_text
                FROM verification_notes vn1
                INNER JOIN (
                    SELECT verification_id, MAX(created_at) AS max_created
                    FROM verification_notes
                    GROUP BY verification_id
                ) vn2
                ON vn1.verification_id = vn2.verification_id
                AND vn1.created_at = vn2.max_created
            ) n ON n.verification_id = v.id
            {where_sql}
            ORDER BY v.created_at DESC
            LIMIT :limit OFFSET :offset
        """
        rows = await self.db.fetch_all(q, values=values)
        return [dict(row) for row in rows]

    async def count_filtered(self, **filters) -> int:
        """Count verifications with optional filters."""
        await self._ensure_connected()
        where_clauses = []
        values: Dict[str, Any] = {}

        if filters.get("status"):
            where_clauses.append("v.status = :status")
            values["status"] = filters["status"]
        if filters.get("document_type_id"):
            where_clauses.append("v.document_type_id = :document_type_id")
            values["document_type_id"] = filters["document_type_id"]
        if filters.get("user_id"):
            where_clauses.append("v.user_id = :user_id")
            values["user_id"] = filters["user_id"]
        if filters.get("date_from"):
            where_clauses.append("v.created_at >= :date_from")
            values["date_from"] = filters["date_from"]
        if filters.get("date_to"):
            where_clauses.append("v.created_at <= :date_to")
            values["date_to"] = filters["date_to"]
        if filters.get("search"):
            where_clauses.append(
                "(u.name LIKE :search OR u.email LIKE :search OR v.current_stage LIKE :search)"
            )
            values["search"] = f"%{filters['search']}%"
        if filters.get("user_name"):
            where_clauses.append("u.name LIKE :user_name")
            values["user_name"] = f"%{filters['user_name']}%"
        if filters.get("operation_type"):
            where_clauses.append("v.current_stage = :operation_type")
            values["operation_type"] = filters["operation_type"]

        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        q = f"""
            SELECT COUNT(*) as total
            FROM verifications v
            LEFT JOIN users u ON v.user_id = u.id
            {where_sql}
        """
        row = await self.db.fetch_one(q, values=values)
        return int(row["total"]) if row else 0

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

    async def count_all_by_status(self) -> Dict[str, int]:
        """Admin-level: count verifications across all users by status."""
        await self._ensure_connected()
        q = "SELECT status, COUNT(*) as total FROM verifications GROUP BY status"
        rows = await self.db.fetch_all(q)
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
                step_name,
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
# Citizen Records Collection
# =========================
class CitizenRecordsCollection:
    """سجلات المواطنين للتحقق من بيانات الوثائق مقابل قاعدة البيانات."""

    def __init__(self, db: Database):
        self.db = db

    async def _ensure_connected(self) -> None:
        if not self.db.is_connected:
            await self.db.connect()

    async def get_by_national_id(self, national_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_connected()
        row = await self.db.fetch_one(
            "SELECT * FROM citizen_records WHERE national_id = :nid",
            values={"nid": national_id},
        )
        return dict(row) if row else None

    async def create(self, data: Dict[str, Any]) -> int:
        """Insert a new citizen record. Missing fields default to None."""
        await self._ensure_connected()
        all_columns = [
            "national_id",
            "full_name_ar",
            "full_name_en",
            "date_of_birth",
            "address",
            "issue_date",
            "expiry_date",
            "gender",
            "nationality",
            "document_type",
        ]
        safe_data = {col: data.get(col) for col in all_columns}
        q = """
            INSERT INTO citizen_records (
                national_id, full_name_ar, full_name_en, date_of_birth,
                address, issue_date, expiry_date, gender,
                nationality, document_type
            ) VALUES (
                :national_id, :full_name_ar, :full_name_en, :date_of_birth,
                :address, :issue_date, :expiry_date, :gender,
                :nationality, :document_type
            )
        """
        return await self.db.execute(q, values=safe_data)

    async def update(self, national_id: str, data: Dict[str, Any]) -> int:
        await self._ensure_connected()
        if not data:
            return 0
        parts = []
        values = {"nid": national_id}
        for key, value in data.items():
            parts.append(f"{key} = :{key}")
            values[key] = value
        q = f"UPDATE citizen_records SET {', '.join(parts)} WHERE national_id = :nid"
        return await self.db.execute(q, values=values)

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "id",
        sort_order: str = "DESC",
    ) -> list[Dict[str, Any]]:
        await self._ensure_connected()
        # sort_by and sort_order are validated by the caller
        rows = await self.db.fetch_all(
            f"SELECT * FROM citizen_records ORDER BY {sort_by} {sort_order} LIMIT :limit OFFSET :offset",
            values={"limit": limit, "offset": offset},
        )
        return [dict(row) for row in rows]


_citizen_records_collection: Optional[CitizenRecordsCollection] = None


def get_citizen_records_collection() -> CitizenRecordsCollection:
    global _citizen_records_collection
    if _citizen_records_collection is None:
        _citizen_records_collection = CitizenRecordsCollection(database)
    return _citizen_records_collection


# =========================
# Verification Notes Collection
# =========================
class VerificationNotesCollection:
    """ملاحظات المشرفين على عمليات التحقق."""

    def __init__(self, db: Database):
        self.db = db

    async def _ensure_connected(self) -> None:
        if not self.db.is_connected:
            await self.db.connect()

    async def add_note(
        self, verification_id: int, admin_id: int, note_text: str
    ) -> int:
        await self._ensure_connected()
        return await self.db.execute(
            """
            INSERT INTO verification_notes (verification_id, admin_id, note_text)
            VALUES (:verification_id, :admin_id, :note_text)
            """,
            values={
                "verification_id": verification_id,
                "admin_id": admin_id,
                "note_text": note_text,
            },
        )

    async def get_by_verification(self, verification_id: int) -> list[Dict[str, Any]]:
        await self._ensure_connected()
        rows = await self.db.fetch_all(
            """
            SELECT vn.*, u.name AS admin_name, u.email AS admin_email
            FROM verification_notes vn
            LEFT JOIN users u ON vn.admin_id = u.id
            WHERE vn.verification_id = :vid
            ORDER BY vn.created_at DESC
            """,
            values={"vid": verification_id},
        )
        return [dict(row) for row in rows]

    async def get_latest_by_verification(
        self, verification_id: int
    ) -> Optional[Dict[str, Any]]:
        await self._ensure_connected()
        row = await self.db.fetch_one(
            """
            SELECT vn.*, u.name AS admin_name, u.email AS admin_email
            FROM verification_notes vn
            LEFT JOIN users u ON vn.admin_id = u.id
            WHERE vn.verification_id = :vid
            ORDER BY vn.created_at DESC
            LIMIT 1
            """,
            values={"vid": verification_id},
        )
        return dict(row) if row else None


_verification_notes_collection: Optional[VerificationNotesCollection] = None


def get_verification_notes_collection() -> VerificationNotesCollection:
    global _verification_notes_collection
    if _verification_notes_collection is None:
        _verification_notes_collection = VerificationNotesCollection(database)
    return _verification_notes_collection


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
        role VARCHAR(20) NOT NULL,
        is_active BOOLEAN DEFAULT TRUE,
        deleted_at TIMESTAMP NULL
    );
    """
    await database.execute(query)

    # Best-effort migrations
    try:
        await database.execute(
            "ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT TRUE;"
        )
    except Exception:
        pass
    try:
        await database.execute(
            "ALTER TABLE users ADD COLUMN deleted_at TIMESTAMP NULL;"
        )
    except Exception:
        pass

    # Create document_types table if not exists
    query = """
    CREATE TABLE IF NOT EXISTS document_types (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) UNIQUE NOT NULL,
        folder_name VARCHAR(255) NOT NULL,
        is_active BOOLEAN DEFAULT TRUE,
        requires_back_image BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    await database.execute(query)

    # Try to add folder_name column on existing deployments (ignore if already exists)
    try:
        await database.execute(
            "ALTER TABLE document_types ADD COLUMN folder_name VARCHAR(255) NOT NULL DEFAULT 'identity';"
        )
    except Exception:
        pass

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

    # Create document_hashes table for deduplication
    doc_hash_query = """
    CREATE TABLE IF NOT EXISTS document_hashes (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        document_id CHAR(36) NOT NULL,
        hash CHAR(64) NOT NULL UNIQUE,
        ipfs_cid VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_hash (hash),
        INDEX idx_doc_id (document_id)
    );
    """
    await database.execute(doc_hash_query)

    # Create biometric audit log table
    biometric_audit_query = """
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
    );
    """
    await database.execute(biometric_audit_query)

    # Create citizen_records table
    citizen_records_query = """
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
    );
    """
    await database.execute(citizen_records_query)

    # Create verification_notes table
    verification_notes_query = """
    CREATE TABLE IF NOT EXISTS verification_notes (
        id INT AUTO_INCREMENT PRIMARY KEY,
        verification_id BIGINT NOT NULL,
        admin_id BIGINT NOT NULL,
        note_text TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_notes_verification (verification_id),
        INDEX idx_notes_admin (admin_id)
    );
    """
    await database.execute(verification_notes_query)
