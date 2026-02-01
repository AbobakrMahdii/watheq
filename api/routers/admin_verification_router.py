from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.database import get_verifications_collection, get_verification_steps_collection
from api.models import VerificationListResponse, VerificationPublic, VerificationStepPublic
from api.security import get_current_admin

router = APIRouter(
    prefix="/api/admin",
    tags=["Admin - Verifications"],
    dependencies=[Depends(get_current_admin)],
)


@router.get("/verifications", response_model=VerificationListResponse)
async def list_verifications(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
):
    collection = get_verifications_collection()
    offset = (page - 1) * page_size
    items = await collection.list_all(limit=page_size, offset=offset)
    total = await collection.count()
    return {"total": total, "page": page, "page_size": page_size, "items": items}


@router.get("/verifications/{verification_id}", response_model=VerificationPublic)
async def get_verification(verification_id: int):
    collection = get_verifications_collection()
    item = await collection.find_one(verification_id)
    if not item:
        raise HTTPException(status_code=404, detail="Verification not found")
    return item


@router.get("/verifications/{verification_id}/steps", response_model=list[VerificationStepPublic])
async def get_verification_steps(verification_id: int):
    collection = get_verification_steps_collection()
    return await collection.list_by_verification(verification_id)
