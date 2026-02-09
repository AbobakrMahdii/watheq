"""Admin Verification Router — واجهة إدارة التحققات

Endpoints for admin dashboard:
- List verifications with filters (status, doc type, user, date, search)
- Get verification detail and pipeline steps
- Admin notes (add/get) on individual verifications
"""

from __future__ import annotations

from datetime import datetime
import logging
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from api.database import (
    get_verifications_collection,
    get_verification_steps_collection,
    get_verification_notes_collection,
)
from api.models import (
    VerificationListResponse,
    VerificationPublic,
    VerificationStepPublic,
)
from api.security import get_current_admin

import csv
import io

router = APIRouter(
    prefix="/api/admin",
    tags=["Admin - Verifications"],
    dependencies=[Depends(get_current_admin)],
)

logger = logging.getLogger("watheq.admin.verifications")

MAX_EXPORT_ROWS = 5000
EXPORT_CHUNK_SIZE = 500


# ---------------------------------------------------------------------------
# قائمة التحققات مع فلاتر متقدمة (6.4)
# ---------------------------------------------------------------------------
@router.get("/verifications", response_model=VerificationListResponse)
async def list_verifications(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    status: Optional[str] = Query(None, description="Filter by status"),
    user_name: Optional[str] = Query(None, description="Filter by user name"),
    operation_type: Optional[str] = Query(None, description="Filter by current stage/module"),
    document_type_id: Optional[int] = Query(None),
    user_id: Optional[int] = Query(None),
    date_from: Optional[str] = Query(None, description="ISO date"),
    date_to: Optional[str] = Query(None, description="ISO date"),
    search: Optional[str] = Query(None, description="Search user name/email"),
    sort_by: Optional[str] = Query("created_at"),
    sort_order: Optional[str] = Query("desc"),
):
    collection = get_verifications_collection()
    offset = (page - 1) * page_size
    items = await collection.list_all(
        limit=page_size,
        offset=offset,
        status=status,
        user_name=user_name,
        operation_type=operation_type,
        document_type_id=document_type_id,
        user_id=user_id,
        date_from=date_from,
        date_to=date_to,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    total = await collection.count_filtered(
        status=status,
        user_name=user_name,
        operation_type=operation_type,
        document_type_id=document_type_id,
        user_id=user_id,
        date_from=date_from,
        date_to=date_to,
        search=search,
    )
    return {"total": total, "page": page, "page_size": page_size, "items": items}


# ---------------------------------------------------------------------------
# إحصائيات التحققات الخفيفة (للعدادات الحية في الداشبورد)
# ---------------------------------------------------------------------------
@router.get("/verifications/stats")
async def verification_stats():
    """Lightweight admin-wide verification counts by status."""
    collection = get_verifications_collection()
    status_counts = await collection.count_all_by_status()
    total = await collection.count()
    return {
        "SUCCESS": status_counts.get("SUCCESS", 0),
        "FAILED": status_counts.get("FAILED", 0),
        "RUNNING": status_counts.get("RUNNING", 0),
        "PENDING": status_counts.get("PENDING", 0),
        "total": total,
    }


@router.get("/verifications/export")
async def export_verifications(
    status: Optional[str] = Query(None, description="Filter by status"),
    user_name: Optional[str] = Query(None, description="Filter by user name"),
    operation_type: Optional[str] = Query(None, description="Filter by current stage/module"),
    date_from: Optional[str] = Query(None, description="ISO date"),
    date_to: Optional[str] = Query(None, description="ISO date"),
    search: Optional[str] = Query(None, description="Search user name/email"),
    format: Optional[str] = Query("csv", description="Export format: csv"),
):
    if format and format.lower() != "csv":
        raise HTTPException(status_code=400, detail="Unsupported export format")

    collection = get_verifications_collection()

    async def stream_csv():
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "verification_id",
                "user_name",
                "user_email",
                "status",
                "result",
                "supervisor_note",
                "created_at",
            ]
        )
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)

        exported = 0
        offset = 0
        while exported < MAX_EXPORT_ROWS:
            batch = await collection.list_for_export(
                limit=min(EXPORT_CHUNK_SIZE, MAX_EXPORT_ROWS - exported),
                offset=offset,
                status=status,
                user_name=user_name,
                operation_type=operation_type,
                date_from=date_from,
                date_to=date_to,
                search=search,
            )
            if not batch:
                break
            for row in batch:
                status_val = row.get("status") or ""
                if status_val == "SUCCESS":
                    result_val = "SUCCESS"
                elif status_val == "FAILED":
                    result_val = "FAILED"
                else:
                    result_val = "IN_PROGRESS"
                writer.writerow(
                    [
                        row.get("id"),
                        row.get("user_name") or f'#{row.get("user_id")}',
                        row.get("user_email") or "",
                        status_val,
                        result_val,
                        row.get("supervisor_note") or "",
                        row.get("created_at") or "",
                    ]
                )
            exported += len(batch)
            offset += len(batch)
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    filename = f'verifications_export_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv'
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(stream_csv(), media_type="text/csv", headers=headers)


def _format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _find_font_path() -> Optional[str]:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/arialuni.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _build_verification_pdf(payload: dict) -> bytes:
    from fpdf import FPDF

    def _pdf_write_line(text: str, line_height: float = 6.0) -> None:
        width = pdf.w - pdf.l_margin - pdf.r_margin
        text = (text or "").replace("\n", " ").strip()
        if not text:
            pdf.ln(line_height)
            return
        words = text.split(" ")
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if pdf.get_string_width(candidate) <= width:
                line = candidate
                continue
            if line:
                pdf.cell(0, line_height, line, ln=1)
                line = ""
            if pdf.get_string_width(word) <= width:
                line = word
            else:
                chunk = ""
                for ch in word:
                    if pdf.get_string_width(chunk + ch) <= width:
                        chunk += ch
                    else:
                        if chunk:
                            pdf.cell(0, line_height, chunk, ln=1)
                        chunk = ch
                line = chunk
        if line:
            pdf.cell(0, line_height, line, ln=1)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    font_path = _find_font_path()
    if font_path:
        pdf.add_font("ReportFont", "", font_path, uni=True)
        pdf.set_font("ReportFont", size=11)
    else:
        pdf.set_font("Helvetica", size=11)

    pdf.set_font_size(14)
    pdf.cell(0, 8, "Verification Report", ln=1)
    pdf.set_font_size(11)

    _pdf_write_line(f"Verification ID: {_format_value(payload.get('verification_id'))}")
    _pdf_write_line(
        f"User: {_format_value(payload.get('user_name'))} ({_format_value(payload.get('user_email'))})"
    )
    _pdf_write_line(f"Operation/Module: {_format_value(payload.get('operation_type'))}")
    _pdf_write_line(f"Status: {_format_value(payload.get('status'))}")
    _pdf_write_line(f"Result: {_format_value(payload.get('result'))}")
    _pdf_write_line(f"Supervisor Note: {_format_value(payload.get('supervisor_note'))}")
    _pdf_write_line(f"Verified At: {_format_value(payload.get('verified_at'))}")
    _pdf_write_line(f"Document Reference: {_format_value(payload.get('document_reference'))}")

    raw = pdf.output(dest="S")
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    return str(raw).encode("latin-1", errors="ignore")


def _build_verification_csv(payload: dict) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "verification_id",
            "user_name",
            "user_email",
            "operation_type",
            "status",
            "result",
            "supervisor_note",
            "verified_at",
            "document_reference",
        ]
    )
    writer.writerow(
        [
            payload.get("verification_id"),
            payload.get("user_name"),
            payload.get("user_email"),
            payload.get("operation_type"),
            payload.get("status"),
            payload.get("result"),
            payload.get("supervisor_note"),
            payload.get("verified_at"),
            payload.get("document_reference"),
        ]
    )
    return output.getvalue().encode("utf-8")


def _extract_document_reference(result_data: dict) -> str:
    if not isinstance(result_data, dict):
        return ""
    blockchain = result_data.get("BLOCKCHAIN") or {}
    if isinstance(blockchain, dict):
        return (
            blockchain.get("sha256")
            or blockchain.get("hash")
            or blockchain.get("cid")
            or blockchain.get("doc_id")
            or ""
        )
    return ""


@router.get("/verifications/{verification_id}/report")
async def get_verification_report(
    verification_id: int,
    format: str = Query("pdf"),
    single: Optional[bool] = Query(True),
):
    if not single:
        raise HTTPException(status_code=400, detail="Single report flag is required")
    verifications = get_verifications_collection()
    notes_col = get_verification_notes_collection()
    item = await verifications.find_one(verification_id)
    if not item:
        raise HTTPException(status_code=404, detail="Verification not found")

    latest_note = await notes_col.get_latest_by_verification(verification_id)
    result_data = item.get("result_data") or {}
    status_val = item.get("status") or ""
    if status_val == "SUCCESS":
        result_val = "SUCCESS"
    elif status_val == "FAILED":
        result_val = "FAILED"
    else:
        result_val = "IN_PROGRESS"

    payload = {
        "verification_id": item.get("id"),
        "user_name": item.get("user_name") or f'#{item.get("user_id")}',
        "user_email": item.get("user_email") or "",
        "operation_type": item.get("current_stage") or "VERIFICATION",
        "status": status_val,
        "result": result_val,
        "supervisor_note": (latest_note or {}).get("note_text") or "",
        "verified_at": item.get("created_at"),
        "document_reference": _extract_document_reference(result_data),
    }

    fmt = (format or "pdf").lower()
    if fmt == "csv":
        content = _build_verification_csv(payload)
        filename = f'verification_{verification_id}_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv'
        return Response(
            content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    if fmt == "pdf":
        content = _build_verification_pdf(payload)
        filename = f'verification_{verification_id}_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.pdf'
        return Response(
            content,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    raise HTTPException(status_code=400, detail="Unsupported report format")


@router.get("/verifications/{verification_id}", response_model=VerificationPublic)
async def get_verification(verification_id: int):
    collection = get_verifications_collection()
    item = await collection.find_one(verification_id)
    if not item:
        raise HTTPException(status_code=404, detail="Verification not found")
    return item


@router.get(
    "/verifications/{verification_id}/steps",
    response_model=list[VerificationStepPublic],
)
async def get_verification_steps(verification_id: int):
    collection = get_verification_steps_collection()
    return await collection.list_by_verification(verification_id)


# ---------------------------------------------------------------------------
# ملاحظات المشرف على التحققات (6.3)
# ---------------------------------------------------------------------------
class NoteCreate(BaseModel):
    text: str = Field(..., alias="note")
    verification_id: Optional[int] = None

    class Config:
        allow_population_by_field_name = True


@router.post("/verifications/{verification_id}/notes")
async def add_note(
    verification_id: int,
    body: NoteCreate,
    admin=Depends(get_current_admin),
):
    """إضافة ملاحظة مشرف على تحقق معين."""

    verifications = get_verifications_collection()
    item = await verifications.find_one(verification_id)
    if not item:
        raise HTTPException(status_code=404, detail="Verification not found")

    notes_col = get_verification_notes_collection()
    admin_id_raw = admin.get("sub") or admin.get("_id") or admin.get("id")
    try:
        admin_id = int(admin_id_raw)
    except (TypeError, ValueError):
        logger.warning("Invalid admin id in token: %r", admin_id_raw)
        raise HTTPException(status_code=401, detail="Invalid admin identity")

    note_text = (body.text or "").strip()
    if not note_text:
        raise HTTPException(status_code=400, detail="Note text is required")
    if body.verification_id is not None and int(body.verification_id) != verification_id:
        raise HTTPException(status_code=400, detail="verification_id mismatch")

    logger.info(
        "Admin %s adding note for verification %s (len=%s)",
        admin_id,
        verification_id,
        len(note_text),
    )
    try:
        note_id = await notes_col.add_note(
            verification_id=verification_id,
            admin_id=admin_id,
            note_text=note_text,
        )
    except Exception as exc:
        logger.exception(
            "Failed to add note for verification %s by admin %s",
            verification_id,
            admin_id,
        )
        raise HTTPException(status_code=500, detail=f"Failed to add note: {exc}")

    logger.info(
        "Admin %s added note %s for verification %s",
        admin_id,
        note_id,
        verification_id,
    )
    return {
        "message": "Note added",
        "note": {
            "id": note_id,
            "verification_id": verification_id,
            "admin_id": admin_id,
            "text": note_text,
        },
    }

@router.get("/verifications/{verification_id}/notes")
async def get_notes(verification_id: int):
    # Fetch admin notes for a verification.
    verifications = get_verifications_collection()
    item = await verifications.find_one(verification_id)
    if not item:
        raise HTTPException(status_code=404, detail="Verification not found")

    notes_col = get_verification_notes_collection()
    rows = await notes_col.get_by_verification(verification_id)
    notes = []
    for row in rows:
        notes.append(
            {
                "id": row.get("id"),
                "verification_id": row.get("verification_id"),
                "admin_id": row.get("admin_id"),
                "admin_name": row.get("admin_name"),
                "admin_email": row.get("admin_email"),
                "text": row.get("note_text"),
                "created_at": row.get("created_at"),
            }
        )
    return notes
