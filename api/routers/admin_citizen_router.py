"""Admin-only endpoints for managing citizen records (super_admin only)."""

from datetime import datetime
import csv
import io
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from ..database import get_citizen_records_collection
from ..security import get_current_super_admin

router = APIRouter(
    prefix="/api/v1/admin/citizens",
    tags=["Admin – Citizens"],
    dependencies=[Depends(get_current_super_admin)],
)


class CitizenUpdate(BaseModel):
    full_name_ar: Optional[str] = None
    full_name_en: Optional[str] = None
    date_of_birth: Optional[str] = None
    address: Optional[str] = None
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    gender: Optional[str] = None
    nationality: Optional[str] = None
    document_type: Optional[str] = None


MAX_EXPORT_ROWS = 5000
EXPORT_CHUNK_SIZE = 500


def _serialize_dates(row: dict) -> dict:
    for key in ("date_of_birth", "issue_date", "expiry_date", "created_at", "updated_at"):
        val = row.get(key)
        if val is not None and not isinstance(val, str):
            row[key] = str(val)
    return row


def _find_font_path() -> Optional[str]:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialuni.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            from pathlib import Path

            if Path(path).exists():
                return path
        except Exception:
            continue
    return None


def _build_citizens_pdf(rows: list[dict]) -> bytes:
    from fpdf import FPDF

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    font_path = _find_font_path()
    if font_path:
        pdf.add_font("ReportFont", "", font_path, uni=True)
        pdf.set_font("ReportFont", size=10)
    else:
        pdf.set_font("Helvetica", size=10)

    pdf.set_font_size(14)
    pdf.cell(0, 8, "Citizens Export", ln=1)
    pdf.set_font_size(10)

    for row in rows:
        line = (
            f"NID: {row.get('national_id','')} | "
            f"Name AR: {row.get('full_name_ar','')} | "
            f"Name EN: {row.get('full_name_en','')} | "
            f"Doc: {row.get('document_type','')} | "
            f"DOB: {row.get('date_of_birth','')}"
        )
        pdf.multi_cell(0, 5, line)

    return pdf.output(dest="S").encode("latin-1", errors="ignore")


@router.get("")
async def list_citizens(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("id"),
    sort_order: str = Query("desc"),
    _admin: dict = Depends(get_current_super_admin),
):
    """List all citizen records (paginated, sortable)."""
    ALLOWED_SORT_COLS = {
        "id",
        "national_id",
        "full_name_ar",
        "full_name_en",
        "date_of_birth",
        "gender",
        "created_at",
    }
    if sort_by not in ALLOWED_SORT_COLS:
        sort_by = "id"
    order = "ASC" if sort_order.upper() == "ASC" else "DESC"

    col = get_citizen_records_collection()
    rows = await col.list_all(
        limit=limit, offset=offset, sort_by=sort_by, sort_order=order
    )
    # Serialise date objects to strings for JSON
    for row in rows:
        _serialize_dates(row)
    return {"citizens": rows, "limit": limit, "offset": offset}


@router.get("/{national_id}")
async def get_citizen(
    national_id: str,
    _admin: dict = Depends(get_current_super_admin),
):
    """Get a single citizen record by national ID."""
    col = get_citizen_records_collection()
    row = await col.get_by_national_id(national_id)
    if not row:
        raise HTTPException(404, "Citizen record not found")
    _serialize_dates(row)
    return row


@router.put("/{national_id}")
async def update_citizen(
    national_id: str,
    body: CitizenUpdate,
    _admin: dict = Depends(get_current_super_admin),
):
    """Update a citizen record (super_admin only)."""
    col = get_citizen_records_collection()
    existing = await col.get_by_national_id(national_id)
    if not existing:
        raise HTTPException(404, "Citizen record not found")

    update_data = {k: v for k, v in body.model_dump().items() if v is not None}
    if not update_data:
        raise HTTPException(400, "No fields to update")

    await col.update(national_id, update_data)
    updated = await col.get_by_national_id(national_id)
    _serialize_dates(updated)
    return updated


@router.delete("/{national_id}")
async def delete_citizen(
    national_id: str,
    _admin: dict = Depends(get_current_super_admin),
):
    """Delete a citizen record (super_admin only)."""
    col = get_citizen_records_collection()
    existing = await col.get_by_national_id(national_id)
    if not existing:
        raise HTTPException(404, "Citizen record not found")

    db = col.db
    await db.execute(
        "DELETE FROM citizen_records WHERE national_id = :nid",
        values={"nid": national_id},
    )
    return {"deleted": True, "national_id": national_id}


@router.get("/export")
async def export_citizens(
    format: Optional[str] = Query("csv", description="Export format: csv or pdf"),
    _admin: dict = Depends(get_current_super_admin),
):
    fmt = (format or "csv").lower()
    if fmt not in ("csv", "pdf"):
        raise HTTPException(status_code=400, detail="Unsupported export format")

    col = get_citizen_records_collection()

    async def _fetch_all(limit: int = MAX_EXPORT_ROWS) -> list[dict]:
        rows: list[dict] = []
        offset = 0
        while len(rows) < limit:
            batch = await col.list_all(limit=min(EXPORT_CHUNK_SIZE, limit - len(rows)), offset=offset)
            if not batch:
                break
            for row in batch:
                _serialize_dates(row)
                rows.append(row)
            offset += len(batch)
        return rows

    if fmt == "csv":
        async def stream_csv():
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(
                [
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
                    "created_at",
                    "updated_at",
                ]
            )
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

            exported = 0
            offset = 0
            while exported < MAX_EXPORT_ROWS:
                batch = await col.list_all(
                    limit=min(EXPORT_CHUNK_SIZE, MAX_EXPORT_ROWS - exported),
                    offset=offset,
                )
                if not batch:
                    break
                for row in batch:
                    _serialize_dates(row)
                    writer.writerow(
                        [
                            row.get("national_id") or "",
                            row.get("full_name_ar") or "",
                            row.get("full_name_en") or "",
                            row.get("date_of_birth") or "",
                            row.get("address") or "",
                            row.get("issue_date") or "",
                            row.get("expiry_date") or "",
                            row.get("gender") or "",
                            row.get("nationality") or "",
                            row.get("document_type") or "",
                            row.get("created_at") or "",
                            row.get("updated_at") or "",
                        ]
                    )
                exported += len(batch)
                offset += len(batch)
                yield output.getvalue()
                output.seek(0)
                output.truncate(0)

        filename = f'citizens_export_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv'
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return StreamingResponse(stream_csv(), media_type="text/csv", headers=headers)

    rows = await _fetch_all()
    content = _build_citizens_pdf(rows)
    filename = f'citizens_export_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.pdf'
    return Response(
        content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
