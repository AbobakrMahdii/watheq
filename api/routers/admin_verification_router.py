"""Admin Verification Router — واجهة إدارة التحققات

Endpoints for admin dashboard:
- List verifications with filters (status, doc type, user, date, search)
- Get verification detail and pipeline steps
- Admin notes (add/get) on individual verifications
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

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

router = APIRouter(
    prefix="/api/admin",
    tags=["Admin - Verifications"],
    dependencies=[Depends(get_current_admin)],
)


# ---------------------------------------------------------------------------
# قائمة التحققات مع فلاتر متقدمة (6.4)
# ---------------------------------------------------------------------------
@router.get("/verifications", response_model=VerificationListResponse)
async def list_verifications(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    status: Optional[str] = Query(None, description="Filter by status"),
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
    text: str


@router.post("/verifications/{verification_id}/notes")
async def add_note(
    verification_id: int,
    body: NoteCreate,
    admin=Depends(get_current_admin),
):
    """إضافة ملاحظة مشرف على تحقق معين."""
    notes_col = get_verification_notes_collection()
    admin_id = admin.get("_id") or admin.get("id")
    await notes_col.add_note(
        verification_id=verification_id,
        admin_id=int(admin_id),
        note_text=body.text,
    )
    return {"message": "تمت إضافة الملاحظة"}


@router.get("/verifications/{verification_id}/notes")
async def get_notes(verification_id: int):
    """جلب ملاحظات المشرفين على تحقق معين."""
    notes_col = get_verification_notes_collection()
    return await notes_col.get_by_verification(verification_id)
