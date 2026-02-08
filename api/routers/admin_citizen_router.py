"""Admin-only endpoints for managing citizen records (super_admin only)."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

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


@router.get("")
async def list_citizens(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _admin: dict = Depends(get_current_super_admin),
):
    """List all citizen records (paginated)."""
    col = get_citizen_records_collection()
    rows = await col.list_all(limit=limit, offset=offset)
    # Serialise date objects to strings for JSON
    for row in rows:
        for key in (
            "date_of_birth",
            "issue_date",
            "expiry_date",
            "created_at",
            "updated_at",
        ):
            val = row.get(key)
            if val is not None and not isinstance(val, str):
                row[key] = str(val)
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
    for key in (
        "date_of_birth",
        "issue_date",
        "expiry_date",
        "created_at",
        "updated_at",
    ):
        val = row.get(key)
        if val is not None and not isinstance(val, str):
            row[key] = str(val)
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
    for key in (
        "date_of_birth",
        "issue_date",
        "expiry_date",
        "created_at",
        "updated_at",
    ):
        val = updated.get(key)
        if val is not None and not isinstance(val, str):
            updated[key] = str(val)
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
