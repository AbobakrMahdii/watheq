from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from fastapi import Request

from api.database import get_audit_log_collection, get_user_collection
from api.security import decode_token

EXCLUDED_PATH_PREFIXES = (
    "/api/v1/docs",
    "/api/v1/openapi.json",
    "/api/v1/redoc",
)

EXCLUDED_PATHS = {
    "/api/v1/auth/login",
    "/api/v1/auth/logout",
}


def _get_client_ip(request: Request) -> Optional[str]:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


def _get_user_agent(request: Request) -> Optional[str]:
    return request.headers.get("user-agent")


def _derive_module(path: str) -> Optional[str]:
    parts = [p for p in path.split("/") if p]
    if not parts:
        return None
    if parts[0] == "api":
        if len(parts) > 1 and parts[1] == "v1":
            return parts[2] if len(parts) > 2 else "v1"
        if len(parts) > 1 and parts[1] == "admin":
            return "admin"
        return parts[1] if len(parts) > 1 else "api"
    return parts[0]


def _derive_operation_type(path: str, method: str) -> str:
    lower_path = path.lower()
    if lower_path.endswith("/login"):
        return "Login"
    if lower_path.endswith("/logout"):
        return "Logout"
    if "verify" in lower_path:
        return "Verify"
    if "upload" in lower_path or "pin-file" in lower_path:
        return "Upload"
    if "ocr" in lower_path:
        return "OCR"

    if method == "POST":
        return "Create"
    if method in {"PUT", "PATCH"}:
        return "Update"
    if method == "DELETE":
        return "Delete"
    if method == "GET":
        return "Read"
    return "Action"


def _should_skip_path(path: str) -> bool:
    if path in EXCLUDED_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in EXCLUDED_PATH_PREFIXES)


async def _resolve_user(request: Request) -> dict[str, Any]:
    auth = request.headers.get("authorization") or ""
    if not auth.lower().startswith("bearer "):
        return {}
    token = auth.split(" ", 1)[1].strip()
    if not token:
        return {}
    try:
        payload = decode_token(token)
    except Exception:
        return {}

    user_id = payload.get("sub")
    role = payload.get("role")
    email = payload.get("email")
    name = None

    if user_id is not None:
        try:
            users = get_user_collection()
            user = await users.find_one({"_id": int(user_id)})
            if user:
                name = user.get("name")
                email = user.get("email") or email
                role = user.get("role") or role
        except Exception:
            pass

    return {
        "user_id": int(user_id) if str(user_id).isdigit() else None,
        "user_name": name,
        "user_email": email,
        "user_role": role,
    }


async def log_request_event(
    request: Request,
    status: str,
    failure_reason: Optional[str] = None,
) -> None:
    path = request.url.path
    if _should_skip_path(path):
        return

    user_info = await _resolve_user(request)
    operation_type = _derive_operation_type(path, request.method)
    module = _derive_module(path)

    doc = {
        "operation_id": str(uuid4()),
        "operation_type": operation_type,
        "status": status,
        "failure_reason": failure_reason,
        "user_id": user_info.get("user_id"),
        "user_name": user_info.get("user_name"),
        "user_email": user_info.get("user_email"),
        "user_role": user_info.get("user_role"),
        "ip_address": _get_client_ip(request),
        "user_agent": _get_user_agent(request),
        "service": "backend",
        "module": module,
        "path": path,
        "method": request.method,
        "file_name": None,
        "file_ext": None,
        "file_size": None,
        "file_cid": None,
        "file_url": None,
        "extra_data": None,
        "created_at": datetime.now(timezone.utc),
    }

    try:
        collection = get_audit_log_collection()
        await collection.insert_one(doc)
    except Exception:
        # Avoid breaking main request flow
        pass


async def log_auth_event(
    request: Request,
    operation_type: str,
    status: str,
    failure_reason: Optional[str] = None,
    user_id: Optional[int] = None,
    user_name: Optional[str] = None,
    user_email: Optional[str] = None,
    user_role: Optional[str] = None,
    user_identifier: Optional[str] = None,
) -> None:
    if _should_skip_path(request.url.path):
        return

    if user_name is None and user_email is None and user_identifier:
        if "@" in user_identifier:
            user_email = user_identifier
        else:
            user_name = user_identifier

    doc = {
        "operation_id": str(uuid4()),
        "operation_type": operation_type,
        "status": status,
        "failure_reason": failure_reason,
        "user_id": user_id,
        "user_name": user_name,
        "user_email": user_email,
        "user_role": user_role,
        "ip_address": _get_client_ip(request),
        "user_agent": _get_user_agent(request),
        "service": "backend",
        "module": "auth",
        "path": request.url.path,
        "method": request.method,
        "file_name": None,
        "file_ext": None,
        "file_size": None,
        "file_cid": None,
        "file_url": None,
        "extra_data": None,
        "created_at": datetime.now(timezone.utc),
    }

    try:
        collection = get_audit_log_collection()
        await collection.insert_one(doc)
    except Exception:
        pass


async def log_file_event(
    request: Request,
    *,
    status: str,
    failure_reason: Optional[str] = None,
    operation_type: str = "Upload",
    module: Optional[str] = None,
    file_name: Optional[str] = None,
    file_size: Optional[int] = None,
    file_cid: Optional[str] = None,
    file_url: Optional[str] = None,
    extra_data: Optional[dict[str, Any]] = None,
) -> None:
    if _should_skip_path(request.url.path):
        return

    user_info = await _resolve_user(request)
    if not module:
        module = _derive_module(request.url.path)

    ext = None
    if file_name and "," not in file_name and "." in file_name:
        ext = file_name.rsplit(".", 1)[-1].lower()

    doc = {
        "operation_id": str(uuid4()),
        "operation_type": operation_type,
        "status": status,
        "failure_reason": failure_reason,
        "user_id": user_info.get("user_id"),
        "user_name": user_info.get("user_name"),
        "user_email": user_info.get("user_email"),
        "user_role": user_info.get("user_role"),
        "ip_address": _get_client_ip(request),
        "user_agent": _get_user_agent(request),
        "service": "backend",
        "module": module,
        "path": request.url.path,
        "method": request.method,
        "file_name": file_name,
        "file_ext": ext,
        "file_size": file_size,
        "file_cid": file_cid,
        "file_url": file_url,
        "extra_data": json.dumps(extra_data) if isinstance(extra_data, dict) else extra_data,
        "created_at": datetime.now(timezone.utc),
    }

    try:
        collection = get_audit_log_collection()
        await collection.insert_one(doc)
    except Exception:
        pass
