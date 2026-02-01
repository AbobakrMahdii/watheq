from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import HTTPException, Request, UploadFile

from api.services.audit_log_service import log_file_event

ALLOWED_MIME_TYPES = {
    "application/pdf": "PDF",
    "application/vnd.ms-excel": "Excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "Excel",
}

ALLOWED_EXTENSIONS = {
    "pdf": "PDF",
    "xls": "Excel",
    "xlsx": "Excel",
}


def _sanitize_filename(name: Optional[str]) -> str:
    if not name:
        return "file"
    clean = Path(name).name
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", clean).strip("._")
    return clean or "file"


def _detect_file_type(filename: str, mime_type: Optional[str]) -> Optional[str]:
    if mime_type and mime_type in ALLOWED_MIME_TYPES:
        return ALLOWED_MIME_TYPES[mime_type]
    ext = Path(filename).suffix.lower().lstrip(".")
    return ALLOWED_EXTENSIONS.get(ext)


async def handle_file_upload(
    request: Request,
    file: UploadFile,
    *,
    module: str = "files",
    storage_dir: Optional[Path] = None,
) -> dict[str, Any]:
    filename = _sanitize_filename(file.filename)
    mime_type = file.content_type or None
    file_type = _detect_file_type(filename, mime_type)

    if not file_type:
        request.state.audit_logged = True
        await log_file_event(
            request,
            status="failed",
            failure_reason="Invalid file type",
            operation_type="FILE_UPLOAD",
            module=module,
            file_name=filename,
            extra_data={
                "file_name": filename,
                "mime_type": mime_type,
                "extension": Path(filename).suffix.lower().lstrip("."),
            },
        )
        raise HTTPException(status_code=400, detail="Invalid file type")

    data = await file.read()
    if not data:
        request.state.audit_logged = True
        await log_file_event(
            request,
            status="failed",
            failure_reason="Empty file upload",
            operation_type="FILE_UPLOAD",
            module=module,
            file_name=filename,
            extra_data={
                "file_name": filename,
                "file_type": file_type,
                "mime_type": mime_type,
            },
        )
        raise HTTPException(status_code=400, detail="Empty file upload")

    if storage_dir is None:
        storage_dir = Path(__file__).resolve().parents[2] / "storage" / "uploads"
    dated_dir = storage_dir / datetime.now(timezone.utc).strftime("%Y/%m")
    dated_dir.mkdir(parents=True, exist_ok=True)

    stored_name = f"{uuid4().hex}_{filename}"
    storage_path = dated_dir / stored_name
    storage_path.write_bytes(data)

    extra_data = {
        "file_name": filename,
        "file_type": file_type,
        "extension": Path(filename).suffix.lower().lstrip("."),
        "file_size": len(data),
        "storage_path": str(storage_path),
        "mime_type": mime_type,
    }

    request.state.audit_logged = True
    await log_file_event(
        request,
        status="success",
        operation_type="FILE_UPLOAD",
        module=module,
        file_name=filename,
        file_size=len(data),
        extra_data=extra_data,
    )

    return {
        "file_name": filename,
        "file_type": file_type,
        "file_size": len(data),
        "mime_type": mime_type,
        "storage_path": str(storage_path),
    }
